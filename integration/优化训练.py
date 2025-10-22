#!/usr/bin/env python3
"""
优化的训练脚本 - Optimized Training Script
包含课程学习、奖励塑形、改进的检查点保存等功能
"""

import os
import sys
import argparse
import yaml
import time
import json
from typing import Dict, Any, List
from pathlib import Path
import torch
import numpy as np

# Add paths
AFLOW_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'AFlow'))
VERL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'verl-agent'))

sys.path.insert(0, AFLOW_PATH)
sys.path.insert(0, VERL_PATH)
sys.path.insert(0, os.path.dirname(__file__))

# Import components
try:
    from scripts.shared_experience import SharedExperiencePool
    from scripts.logs import logger

    from unified_state import StateManager
    from trainable_qwen_policy import TrainableQwenPolicy
    from rl_trainer import RLTrainer
    from deep_workflow_env_with_meta import create_deep_workflow_env  # 使用元学习增强版本
    from workflow_prompt_manager import get_prompt_manager

    IMPORTS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import required components: {e}")
    import traceback
    traceback.print_exc()
    IMPORTS_AVAILABLE = False


class RewardShaper:
    """奖励塑形器 - 解决稀疏奖励问题"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.enabled = config.get('enable', True)
        self.partial_credit = config.get('partial_credit', True)
        self.improvement_bonus = config.get('improvement_bonus', 0.1)
        self.proximity_reward = config.get('proximity_reward', True)
        self.proximity_scale = config.get('proximity_scale', 0.5)
        self.baseline_reward = config.get('baseline_reward', 0.01)

        self.history = {}  # 记录每个问题的历史得分

    def shape_reward(self,
                    raw_reward: float,
                    expected_answer: Any,
                    actual_answer: Any,
                    problem_id: str) -> float:
        """
        对原始奖励进行塑形

        Args:
            raw_reward: 原始奖励（0或1）
            expected_answer: 期望答案
            actual_answer: 实际答案
            problem_id: 问题ID

        Returns:
            塑形后的奖励
        """
        if not self.enabled:
            return raw_reward

        shaped_reward = raw_reward

        # 1. 基础奖励：避免全0
        if raw_reward == 0:
            shaped_reward += self.baseline_reward

        # 2. 接近度奖励：答案接近正确答案也给奖励
        if self.proximity_reward and raw_reward == 0:
            try:
                expected_num = int(expected_answer)
                actual_num = int(actual_answer) if actual_answer else 0

                # 计算相对误差
                if expected_num != 0:
                    relative_error = abs(expected_num - actual_num) / abs(expected_num)
                    # 误差越小，奖励越高
                    proximity = max(0, 1 - relative_error)
                    shaped_reward += proximity * self.proximity_scale
            except (ValueError, TypeError):
                pass

        # 3. 进步奖励：比上一次好就给奖励
        if problem_id in self.history:
            last_reward = self.history[problem_id]
            if shaped_reward > last_reward:
                shaped_reward += self.improvement_bonus

        self.history[problem_id] = shaped_reward

        return shaped_reward


class CurriculumScheduler:
    """课程学习调度器"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.enabled = config.get('enable', False)
        self.stages = config.get('stages', [])

    def get_sample_size(self, epoch: int) -> int:
        """根据当前epoch返回应该使用的样本数量"""
        if not self.enabled:
            return 12  # 默认全部

        for stage in self.stages:
            epoch_range = stage['epoch']
            # 支持两种格式: 整数 (1, 2, 3) 或 字符串范围 ("1-5", "6-10")
            if isinstance(epoch_range, int):
                if epoch == epoch_range:
                    return stage['sample_size']
            elif isinstance(epoch_range, str) and '-' in epoch_range:
                start, end = map(int, epoch_range.split('-'))
                if start <= epoch <= end:
                    return stage['sample_size']

        return 12  # 默认


class OptimizedTrainer:
    """优化的训练器"""

    def __init__(self, config: Dict[str, Any]):
        """初始化"""
        if not IMPORTS_AVAILABLE:
            raise RuntimeError("Required imports not available")

        self.config = config
        self.device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))

        print("=" * 80)
        print("  优化的AIME训练系统 - Optimized AIME Training System")
        print("=" * 80)
        print(f"\nDevice: {self.device}")
        print(f"PyTorch: {torch.__version__}")
        print(f"CUDA: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")

        # 训练参数
        self.total_epochs = config.get('total_epochs', 20)
        self.episodes_per_epoch = config.get('episodes_per_epoch', 8)
        self.update_frequency = config.get('update_frequency', 2)
        self.save_frequency = config.get('save_frequency', 1)

        # 路径
        self.output_dir = Path(config.get('output_dir', './output/optimized'))
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.checkpoint_dir = self.output_dir / 'checkpoints'
        self.checkpoint_dir.mkdir(exist_ok=True)

        # 课程学习
        self.curriculum = CurriculumScheduler(config.get('curriculum', {}))
        logger.info(f"✓ 课程学习: {'启用' if self.curriculum.enabled else '禁用'}")

        # 奖励塑形
        reward_config = config.get('rl', {}).get('reward_shaping', {})
        self.reward_shaper = RewardShaper(reward_config)
        logger.info(f"✓ 奖励塑形: {'启用' if self.reward_shaper.enabled else '禁用'}")

        # 共享组件
        self.shared_experience_pool = SharedExperiencePool(
            max_size=config.get('experience_pool_size', 5000),
            eviction_strategy='lowest_score'
        )
        self.state_manager = StateManager()
        self.prompt_manager = get_prompt_manager()

        # 环境配置
        self.env_config = config.get('environment', {})
        self.train_datasets = self.env_config.get('train_datasets', ['AIME'])

        # RL配置
        self.rl_config = config.get('rl', {})

        # 加载策略
        print("\n" + "=" * 80)
        print("加载Qwen策略模型")
        print("=" * 80)
        self._load_policy()

        # 创建RL trainer
        print("\n" + "=" * 80)
        print("创建RL Trainer")
        print("=" * 80)
        self._create_rl_trainer()

        # 环境
        self.train_envs = {}

        # 统计
        self.stats = {
            'epoch': 0,
            'best_score': 0.0,
            'best_scores': [],  # 每个epoch的最佳分数
            'avg_scores': [],
            'rewards_shaped': [],  # 塑形后的奖励
            'checkpoints_saved': []
        }

        # 早停
        early_stop_config = config.get('early_stopping', {})
        self.early_stop_enabled = early_stop_config.get('enable', True)
        self.early_stop_patience = early_stop_config.get('patience', 5)
        self.early_stop_min_improvement = early_stop_config.get('min_improvement', 0.01)
        self.epochs_without_improvement = 0

        logger.info("✅ 优化训练器初始化完成")

    def _load_policy(self):
        """加载策略"""
        policy_config = self.rl_config.get('policy', {})
        model_path = policy_config.get('model_path')

        if not model_path:
            raise ValueError("必须指定model_path")

        self.policy = TrainableQwenPolicy(
            model_path=model_path,
            device=str(self.device),
            torch_dtype=torch.bfloat16,
            use_lora=policy_config.get('use_lora', True),
            lora_r=policy_config.get('lora_r', 32),
            lora_alpha=policy_config.get('lora_alpha', 64),
            value_head_hidden_dim=policy_config.get('value_head_dim', 2048)
        )

        self.policy.system_prompt = self.prompt_manager.get_system_prompt()

        print(f"✓ 模型加载完成: {model_path}")
        print(f"✓ LoRA: r={policy_config.get('lora_r', 32)}, alpha={policy_config.get('lora_alpha', 64)}")

    def _create_rl_trainer(self):
        """创建RL trainer"""
        self.rl_trainer = RLTrainer(
            policy=self.policy,
            learning_rate=self.rl_config.get('learning_rate', 0.0003),
            value_coef=self.rl_config.get('value_coef', 1.0),
            entropy_coef=self.rl_config.get('entropy_coef', 0.1),
            max_grad_norm=self.rl_config.get('gradient_clip', 0.5),
            gamma=self.rl_config.get('gamma', 0.99),
            gae_lambda=self.rl_config.get('gae_lambda', 0.95),
            ppo_epochs=self.rl_config.get('ppo_epochs', 6),
            ppo_clip=self.rl_config.get('ppo_clip', 0.25),
            batch_size=self.rl_config.get('batch_size', 8),
            use_gigpo=self.rl_config.get('gigpo', {}).get('enable', True),
            gigpo_config={k: v for k, v in self.rl_config.get('gigpo', {}).items() if k != 'enable'},
            device=str(self.device)
        )

        print(f"✓ RL Trainer创建完成")

    def _create_environments(self):
        """创建环境"""
        print("\n" + "=" * 80)
        print("创建训练环境")
        print("=" * 80)

        for dataset in self.train_datasets:
            logger.info(f"创建{dataset}环境...")

            env = create_deep_workflow_env(
                dataset=dataset,
                opt_llm_config=self.env_config.get('opt_llm_config', {}),
                exec_llm_config=self.env_config.get('exec_llm_config', {}),
                operators=self.env_config.get('operators', []),
                env_num=self.env_config.get('env_num', 1),
                sample=self.env_config.get('sample', 12),
                max_rounds=self.env_config.get('max_rounds', 8),
                workspace_path=str(self.output_dir / 'workflows' / dataset)
            )

            self.train_envs[dataset] = env
            logger.info(f"✓ {dataset}环境创建完成")

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """训练一个epoch"""
        logger.info(f"\n{'=' * 80}")
        logger.info(f"Epoch {epoch}/{self.total_epochs}")

        # 课程学习：根据epoch调整样本数量
        sample_size = self.curriculum.get_sample_size(epoch)
        logger.info(f"📚 当前样本数: {sample_size} (课程学习)")
        logger.info(f"{'=' * 80}")

        epoch_stats = {
            'total_episodes': 0,
            'avg_score': 0.0,
            'avg_reward_shaped': 0.0,
            'num_updates': 0
        }

        # 训练
        for dataset, env in self.train_envs.items():
            # 动态调整环境的sample大小
            env.sample = sample_size

            logger.info(f"\n训练 {dataset}...")

            for update_iter in range(self.episodes_per_epoch // self.update_frequency):
                print(f"\n[{update_iter + 1}/{self.episodes_per_epoch // self.update_frequency}] 收集数据...")

                # 收集rollout
                collection_stats = self.rl_trainer.collect_rollout(
                    env=env,
                    num_episodes=self.update_frequency,
                    max_steps_per_episode=self.env_config.get('max_rounds', 8)
                )

                # 更新策略
                print("更新策略...")
                update_stats = self.rl_trainer.update()

                # 统计
                epoch_stats['total_episodes'] += collection_stats['num_episodes']
                epoch_stats['avg_score'] += collection_stats.get('avg_reward', 0.0)
                epoch_stats['num_updates'] += 1

                print(f"收集: {collection_stats}")
                print(f"更新: {update_stats}")

        # 平均统计
        if epoch_stats['num_updates'] > 0:
            epoch_stats['avg_score'] /= epoch_stats['num_updates']
            epoch_stats['avg_reward_shaped'] /= epoch_stats['num_updates']

        # 更新全局统计
        self.stats['epoch'] = epoch
        self.stats['avg_scores'].append(epoch_stats['avg_score'])

        logger.info(f"\n{'=' * 80}")
        logger.info(f"Epoch {epoch} 总结:")
        logger.info(f"  平均分数: {epoch_stats['avg_score']:.4f}")
        logger.info(f"  更新次数: {epoch_stats['num_updates']}")
        logger.info(f"{'=' * 80}")

        # 打印元学习统计
        logger.info(f"\n{'=' * 80}")
        logger.info("元学习进度报告 - Meta Learning Progress")
        logger.info(f"{'=' * 80}")
        for dataset, env in self.train_envs.items():
            if hasattr(env, 'print_meta_statistics'):
                logger.info(f"\n{dataset}:")
                env.print_meta_statistics()

        return epoch_stats

    def save_checkpoint(self, epoch: int, score: float, is_best: bool = False):
        """保存检查点"""
        # 保存策略
        checkpoint_name = f"best.pt" if is_best else f"epoch_{epoch}.pt"
        policy_path = self.checkpoint_dir / f"{checkpoint_name.replace('.pt', '_policy.pt')}"
        self.policy.save_checkpoint(str(policy_path))

        # 保存trainer
        trainer_path = self.checkpoint_dir / f"{checkpoint_name.replace('.pt', '_trainer.pt')}"
        self.rl_trainer.save_checkpoint(str(trainer_path))

        # 保存统计
        stats_path = self.checkpoint_dir / f"{checkpoint_name.replace('.pt', '_stats.json')}"
        with open(stats_path, 'w') as f:
            json.dump(self.stats, f, indent=2)

        self.stats['checkpoints_saved'].append({
            'epoch': epoch,
            'score': score,
            'is_best': is_best,
            'path': str(policy_path)
        })

        logger.info(f"✓ 检查点已保存: {checkpoint_name} (分数: {score:.4f})")

        # 保存元学习检查点
        for dataset, env in self.train_envs.items():
            if hasattr(env, 'save_meta_checkpoint'):
                env.save_meta_checkpoint()
                logger.info(f"✓ {dataset} 元学习检查点已保存")

    def train(self):
        """主训练循环"""
        logger.info("\n" + "=" * 80)
        logger.info("开始优化训练")
        logger.info("=" * 80)

        # 创建环境
        self._create_environments()

        # 训练循环
        for epoch in range(1, self.total_epochs + 1):
            # 训练
            epoch_stats = self.train_epoch(epoch)

            # 保存检查点
            if epoch % self.save_frequency == 0:
                self.save_checkpoint(epoch, epoch_stats['avg_score'])

            # 检查是否是最佳
            if epoch_stats['avg_score'] > self.stats['best_score']:
                improvement = epoch_stats['avg_score'] - self.stats['best_score']
                self.stats['best_score'] = epoch_stats['avg_score']
                self.save_checkpoint(epoch, epoch_stats['avg_score'], is_best=True)
                logger.info(f"🎉 新最佳分数: {self.stats['best_score']:.4f} (+{improvement:.4f})")
                self.epochs_without_improvement = 0
            else:
                self.epochs_without_improvement += 1

            # 早停检查
            if self.early_stop_enabled and self.epochs_without_improvement >= self.early_stop_patience:
                logger.info(f"\n⚠️ 早停触发: {self.early_stop_patience}个epoch无改进")
                break

        logger.info("\n" + "=" * 80)
        logger.info("训练完成!")
        logger.info(f"总Epoch数: {self.stats['epoch']}")
        logger.info(f"最佳分数: {self.stats['best_score']:.4f}")
        logger.info(f"保存的检查点数: {len(self.stats['checkpoints_saved'])}")
        logger.info("=" * 80)


def main():
    """主入口"""
    parser = argparse.ArgumentParser(description="优化的AIME训练")
    parser.add_argument('--config', type=str, default='优化运行.yaml', help="配置文件路径")
    args = parser.parse_args()

    # 加载配置
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # 替换环境变量
    def replace_env_vars(obj):
        if isinstance(obj, dict):
            return {k: replace_env_vars(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [replace_env_vars(item) for item in obj]
        elif isinstance(obj, str) and obj.startswith('${') and obj.endswith('}'):
            env_var = obj[2:-1]
            return os.environ.get(env_var, obj)
        return obj

    config = replace_env_vars(config)

    # 创建trainer
    trainer = OptimizedTrainer(config)

    # 训练
    trainer.train()


if __name__ == "__main__":
    main()
