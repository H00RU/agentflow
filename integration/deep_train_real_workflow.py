"""
Real Workflow Training Script - 真正的Workflow深度集成训练
True deep integration training with real workflow generation and execution
"""

import os
import sys
import argparse
import yaml
import time
from typing import Dict, Any
from pathlib import Path
import torch

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
    from deep_workflow_env import create_deep_workflow_env
    from workflow_prompt_manager import get_prompt_manager

    IMPORTS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import required components: {e}")
    import traceback
    traceback.print_exc()
    IMPORTS_AVAILABLE = False


class RealWorkflowTrainer:
    """
    真实Workflow训练器

    这是REAL实现：
    1. Qwen生成workflow描述
    2. 解析成workflow代码
    3. 执行真实的HumanEval测试
    4. 使用真实pass@k作为reward
    5. 训练Qwen学习workflow优化
    """

    def __init__(self, config: Dict[str, Any], resume_from: str = None):
        """初始化trainer"""
        if not IMPORTS_AVAILABLE:
            raise RuntimeError("Required imports not available. Check dependencies.")

        self.config = config
        self.resume_from = resume_from
        self.start_epoch = 1  # Will be updated if resuming
        self.device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))

        print("=" * 80)
        print("  REAL WORKFLOW DEEP INTEGRATION TRAINING")
        print("  真实Workflow深度集成训练")
        print("=" * 80)
        print(f"\nDevice: {self.device}")
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA device: {torch.cuda.get_device_name(0)}")
            print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

        # Training parameters
        self.total_epochs = config.get('total_epochs', 10)
        self.episodes_per_epoch = config.get('episodes_per_epoch', 10)
        self.update_frequency = config.get('update_frequency', 5)

        # Paths
        self.output_dir = Path(config.get('output_dir', './output/real_workflow'))
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.checkpoint_dir = self.output_dir / 'checkpoints'
        self.checkpoint_dir.mkdir(exist_ok=True)

        self.log_dir = self.output_dir / 'logs'
        self.log_dir.mkdir(exist_ok=True)

        self.workflow_dir = self.output_dir / 'workflows_generated'
        self.workflow_dir.mkdir(exist_ok=True)

        # Shared components
        self.shared_experience_pool = SharedExperiencePool(
            max_size=config.get('experience_pool_size', 10000),
            eviction_strategy=config.get('experience_eviction', 'lowest_score')
        )

        self.state_manager = StateManager()

        # Get prompt manager
        self.prompt_manager = get_prompt_manager()

        # Environment configuration
        self.env_config = config.get('environment', {})
        # 从配置读取训练数据集，不提供默认值（强制用户在配置文件中指定）
        self.train_datasets = self.env_config.get('train_datasets', [])
        if not self.train_datasets:
            raise ValueError("Please specify 'train_datasets' in config file under 'environment' section")

        # RL configuration
        self.rl_config = config.get('rl', {})

        # Load trainable policy
        print("\n" + "=" * 80)
        print("Loading Trainable Qwen Policy with Workflow Prompt")
        print("=" * 80)
        self._load_trainable_policy()

        # Create RL trainer
        print("\n" + "=" * 80)
        print("Creating RL Trainer")
        print("=" * 80)
        self._create_rl_trainer()

        # Environment (will be created lazily)
        self.train_envs = {}

        # Statistics
        self.stats = {
            'epoch': 0,
            'total_episodes': 0,
            'total_updates': 0,
            'best_score': 0.0,
            'avg_scores': [],
            'policy_losses': [],
            'value_losses': [],
            'workflow_history': []
        }

        logger.info("RealWorkflowTrainer initialized successfully")
        logger.info("✅ READY FOR REAL WORKFLOW TRAINING")

    def _load_trainable_policy(self):
        """加载trainable Qwen policy"""
        policy_config = self.rl_config.get('policy', {})
        model_path = policy_config.get('model_path')

        if model_path is None:
            raise ValueError("model_path must be specified in config")

        self.policy = TrainableQwenPolicy(
            model_path=model_path,
            device=str(self.device),
            torch_dtype=torch.bfloat16,
            freeze_base=policy_config.get('freeze_base', False),
            use_lora=policy_config.get('use_lora', True),
            lora_r=policy_config.get('lora_r', 16),
            lora_alpha=policy_config.get('lora_alpha', 32),
            value_head_hidden_dim=policy_config.get('value_head_dim', 1024)
        )

        # Set system prompt for workflow generation
        self.policy.system_prompt = self.prompt_manager.get_system_prompt()

        print(f"\n✓ Trainable policy loaded")
        print(f"✓ Model: {model_path}")
        print(f"✓ LoRA enabled: {policy_config.get('use_lora', True)}")
        print(f"✓ System prompt for workflow generation: SET")

    def _create_rl_trainer(self):
        """创建RL trainer"""
        rl_config = self.rl_config

        self.rl_trainer = RLTrainer(
            policy=self.policy,
            learning_rate=rl_config.get('learning_rate', 1e-5),
            value_coef=rl_config.get('value_coef', 0.5),
            entropy_coef=rl_config.get('entropy_coef', 0.01),
            max_grad_norm=rl_config.get('gradient_clip', 1.0),
            gamma=rl_config.get('gamma', 0.99),
            gae_lambda=rl_config.get('gae_lambda', 0.95),
            ppo_epochs=rl_config.get('ppo_epochs', 4),
            ppo_clip=rl_config.get('ppo_clip', 0.2),
            batch_size=rl_config.get('batch_size', 32),
            use_gigpo=rl_config.get('gigpo', {}).get('enable', True),
            gigpo_config={k: v for k, v in rl_config.get('gigpo', {}).items() if k != 'enable'},
            device=str(self.device)
        )

        print(f"\n✓ RL trainer created")

    def _evaluate_on_test_set(self, env):
        """
        在测试集上评估训练好的policy生成的workflow

        改进: 让训练好的policy生成新的workflow并在测试集上评估
        这样才能真正测试policy是否学会了workflow设计能力

        Args:
            env: environment实例

        Returns:
            测试集上的平均分数
        """
        logger.info("[Trainer] 🧪 Evaluating trained policy on TEST set...")
        logger.info("[Trainer] Policy will generate a NEW workflow for test evaluation")

        from workflow_parser import WorkflowParser
        import importlib.util
        import asyncio

        parser = WorkflowParser()
        test_dataset = self.train_datasets[0] if self.train_datasets else "AIME"

        # 计算测试集大小（通用方式：使用配置的train_test_split）
        train_test_split = self.env_config.get('train_test_split', 0.8)
        total_problems = len(env.evaluator.problems)
        train_size = int(total_problems * train_test_split)
        test_size = total_problems - train_size
        # 允许config覆盖
        num_test_problems = self.env_config.get('test_problems', test_size)
        logger.info(f"[Trainer] Dataset: {total_problems} total, {train_size} train, {test_size} test")
        logger.info(f"[Trainer] Will evaluate on {num_test_problems} test problems")

        try:
            # 步骤1: 构造测试observation (包含训练摘要但不泄露测试集信息)
            test_obs = self._construct_test_observation(env, test_dataset)
            logger.info(f"[Trainer] Test observation constructed (length: {len(test_obs)} chars)")

            # 步骤2: 使用训练好的policy生成workflow描述
            logger.info("[Trainer] Generating workflow using trained policy...")
            workflow_desc, _, _, _ = self.policy.get_action_and_value(
                obs=test_obs,
                max_new_tokens=300,
                temperature=0.7  # 保持一定随机性
            )
            logger.info(f"[Trainer] Policy generated workflow description:")
            logger.info(f"[Trainer] {workflow_desc[:200]}...")  # 打印前200字符

            # 步骤3: 解析workflow描述
            logger.info("[Trainer] Parsing workflow description...")
            workflow_spec = parser.parse_qwen_output(
                workflow_desc,
                dataset_type=test_dataset,
                sample_count=self.env_config.get('workflow_sample_count')
            )

            if workflow_spec is None:
                logger.warning("[Trainer] Failed to parse policy output, falling back to best_workflow")
                return self._evaluate_fallback_workflow(env, num_test_problems)

            logger.info(f"[Trainer] ✓ Parsed workflow: {len(workflow_spec.operators)} operators, {len(workflow_spec.steps)} steps")
            logger.info(f"[Trainer]   Operators: {workflow_spec.operators}")

            # 步骤4: 保存workflow到文件
            test_workflow_path = parser.save_workflow_to_file(
                workflow_spec,
                "policy_test_eval",
                str(self.workflow_dir / "test_evaluation")
            )
            logger.info(f"[Trainer] ✓ Workflow saved to {test_workflow_path}")

            # 步骤5: 导入并实例化workflow
            spec = importlib.util.spec_from_file_location("test_workflow", test_workflow_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            WorkflowClass = module.Workflow
            workflow = WorkflowClass(
                name="PolicyTestEvalWorkflow",
                llm_config=self.env_config['exec_llm_config'],
                dataset=test_dataset
            )

            # 步骤6: 在测试集上执行
            logger.info(f"[Trainer] Executing policy-generated workflow on {num_test_problems} TEST problems...")
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            result = loop.run_until_complete(
                env.evaluator.evaluate_workflow(
                    workflow,
                    num_problems=num_test_problems,
                    use_test_set=True,
                    random_sample=False  # 固定取前N个测试集问题
                )
            )

            loop.close()

            test_score = result.get('pass_at_k', 0.0)
            logger.info(f"[Trainer] ✅ Policy-generated workflow TEST score: {test_score:.4f}")

            return test_score

        except Exception as e:
            logger.error(f"[Trainer] ❌ Error during policy test evaluation: {e}")
            logger.warning("[Trainer] Falling back to best_workflow evaluation")
            import traceback
            traceback.print_exc()
            return self._evaluate_fallback_workflow(env, num_test_problems)

    def _construct_test_observation(self, env, dataset: str) -> str:
        """
        构造测试observation

        包含训练摘要信息但不泄露测试集内容，让policy能够基于训练经验生成workflow

        Args:
            env: environment实例
            dataset: 数据集名称

        Returns:
            测试observation字符串
        """
        # 获取训练统计信息
        best_score = getattr(env, 'best_score', 0.0)
        best_operators = []
        if hasattr(env, 'best_workflow') and env.best_workflow:
            best_operators = env.best_workflow.operators

        # 可用的operators
        available_operators = self.env_config.get('operators', ['Custom', 'ScEnsemble'])

        # 构造observation
        obs = f"""Dataset: {dataset}
Task: Design an optimized workflow for TEST evaluation

Training Summary:
- Best training score achieved: {best_score:.4f}
- Best performing operators: {', '.join(best_operators) if best_operators else 'N/A'}
- Available operators: {', '.join(available_operators)}

Your task: Generate a high-quality workflow that generalizes well to unseen test problems.
Focus on designing a robust workflow that can handle the complexity of {dataset} problems.

IMPORTANT: Output your workflow in the required XML format with <workflow_modification>, <operators>, and <workflow_steps>.
"""

        return obs

    def _evaluate_fallback_workflow(self, env, num_test_problems: int):
        """
        Fallback方法: 使用训练中找到的best_workflow进行测试

        这是原始的测试方法，当policy生成失败时使用

        Args:
            env: environment实例
            num_test_problems: 测试问题数量

        Returns:
            测试集上的分数
        """
        logger.info("[Trainer] Using fallback: evaluating best_workflow from training")

        if env.best_workflow is None:
            logger.warning("[Trainer] No best workflow found, returning 0.0")
            return 0.0

        from workflow_parser import WorkflowParser
        import importlib.util
        import asyncio

        parser = WorkflowParser()
        test_dataset = self.train_datasets[0] if self.train_datasets else "AIME"

        # 保存临时workflow
        test_workflow_path = parser.save_workflow_to_file(
            env.best_workflow,
            "fallback_test_eval",
            str(self.workflow_dir / "temp")
        )

        # 导入并测试
        spec = importlib.util.spec_from_file_location("test_workflow", test_workflow_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        WorkflowClass = module.Workflow
        workflow = WorkflowClass(
            name="FallbackTestEvalWorkflow",
            llm_config=self.env_config['exec_llm_config'],
            dataset=test_dataset
        )

        # 在测试集上评估
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        result = loop.run_until_complete(
            env.evaluator.evaluate_workflow(
                workflow,
                num_problems=num_test_problems,
                use_test_set=True,
                random_sample=False
            )
        )

        loop.close()

        return result.get('pass_at_k', 0.0)

    def _create_environments(self):
        """创建训练环境"""
        print("\n" + "=" * 80)
        print("Creating REAL Workflow Environments")
        print("=" * 80)

        opt_llm_config = self.env_config.get('opt_llm_config', {})
        exec_llm_config = self.env_config.get('exec_llm_config', {})
        operators = self.env_config.get('operators', ['Custom', 'CustomCodeGenerate', 'ScEnsemble', 'Test'])

        env_num = self.env_config.get('env_num', 2)
        sample = self.env_config.get('sample', 3)
        max_rounds = self.env_config.get('max_rounds', 10)
        workflow_sample_count = self.env_config.get('workflow_sample_count')
        train_test_split = self.env_config.get('train_test_split', 0.8)

        # Create training environments
        for dataset in self.train_datasets:
            logger.info(f"Creating REAL workflow environment for {dataset}")

            # Create DEEP WORKFLOW environment (真正的workflow执行)
            env = create_deep_workflow_env(
                dataset=dataset,
                opt_llm_config=opt_llm_config,
                exec_llm_config=exec_llm_config,
                operators=operators,
                env_num=env_num,
                sample=sample,
                max_rounds=max_rounds,
                workspace_path=str(self.workflow_dir / dataset),
                workflow_sample_count=workflow_sample_count,
                train_test_split=train_test_split
            )

            logger.info(f"✅ REAL Workflow Environment created")
            logger.info(f"   Dataset: {dataset}")
            logger.info(f"   Workflow generation: Qwen → Parser → Python code")
            logger.info(f"   Evaluation: Real HumanEval execution")
            logger.info(f"   Reward: Real pass@k scores")

            self.train_envs[dataset] = env

        print(f"\n✓ Created {len(self.train_envs)} REAL workflow environments")

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """训练一个epoch"""
        logger.info(f"\n{'=' * 80}")
        logger.info(f"Epoch {epoch}/{self.total_epochs}")
        logger.info(f"{'=' * 80}")

        epoch_stats = {
            'total_episodes': 0,
            'avg_score': 0.0,
            'avg_reward': 0.0,
            'num_updates': 0,
            'policy_loss': 0.0,
            'value_loss': 0.0
        }

        # Train on each dataset
        for dataset, env in self.train_envs.items():
            logger.info(f"\nTraining on {dataset} with REAL workflow execution")

            # Collect rollouts and update
            for update_iter in range(self.episodes_per_epoch // self.update_frequency):
                # Collect rollouts
                print(f"\n[{update_iter + 1}/{self.episodes_per_epoch // self.update_frequency}] Collecting rollouts...")
                print("Qwen will generate workflow descriptions")
                print("→ Parser will convert to workflow code")
                print("→ Real HumanEval tests will run")
                print("→ Real pass@k will be returned as reward")

                collection_stats = self.rl_trainer.collect_rollout(
                    env=env,
                    num_episodes=self.update_frequency,
                    max_steps_per_episode=self.env_config.get('max_rounds', 10)
                )

                # Update policy
                print(f"\nUpdating policy with real workflow rewards...")
                update_stats = self.rl_trainer.update()

                # Record stats
                epoch_stats['total_episodes'] += collection_stats['num_episodes']
                epoch_stats['avg_reward'] += collection_stats['avg_reward']
                epoch_stats['avg_score'] += collection_stats.get('avg_reward', 0.0)
                epoch_stats['num_updates'] += 1

                if 'policy_loss' in update_stats:
                    epoch_stats['policy_loss'] += update_stats['policy_loss']
                    epoch_stats['value_loss'] += update_stats['value_loss']

                print(f"\nCollection stats: {collection_stats}")
                print(f"Update stats: {update_stats}")

        # Average stats
        if epoch_stats['num_updates'] > 0:
            epoch_stats['policy_loss'] /= epoch_stats['num_updates']
            epoch_stats['value_loss'] /= epoch_stats['num_updates']
            epoch_stats['avg_reward'] /= epoch_stats['num_updates']
            epoch_stats['avg_score'] /= epoch_stats['num_updates']

        # Update global statistics
        self.stats['epoch'] = epoch
        self.stats['total_episodes'] += epoch_stats['total_episodes']
        self.stats['total_updates'] += epoch_stats['num_updates']
        self.stats['avg_scores'].append(epoch_stats['avg_score'])
        self.stats['policy_losses'].append(epoch_stats['policy_loss'])
        self.stats['value_losses'].append(epoch_stats['value_loss'])

        logger.info(f"\n{'=' * 80}")
        logger.info(f"Epoch {epoch} Summary:")
        logger.info(f"  Total episodes: {epoch_stats['total_episodes']}")
        logger.info(f"  Avg real workflow score: {epoch_stats['avg_score']:.4f}")
        logger.info(f"  Updates: {epoch_stats['num_updates']}")
        logger.info(f"  Policy loss: {epoch_stats['policy_loss']:.4f}")
        logger.info(f"  Value loss: {epoch_stats['value_loss']:.4f}")
        logger.info(f"{'=' * 80}")

        # Evaluate on TEST set at end of epoch
        logger.info(f"\n🧪 Evaluating on TEST set...")
        test_score = self._evaluate_on_test_set(env)
        epoch_stats['test_score'] = test_score
        logger.info(f"📊 TEST Set Score: {test_score:.4f}")
        logger.info(f"{'=' * 80}\n")

        return epoch_stats

    def save_checkpoint(self, epoch: int, best: bool = False):
        """保存checkpoint"""
        checkpoint_name = f"best.pt" if best else f"epoch_{epoch}.pt"
        checkpoint_path = self.checkpoint_dir / checkpoint_name

        # Save policy
        policy_path = self.checkpoint_dir / checkpoint_name.replace('.pt', '_policy.pt')
        self.policy.save_checkpoint(str(policy_path))

        # Save trainer
        trainer_path = self.checkpoint_dir / checkpoint_name.replace('.pt', '_trainer.pt')
        self.rl_trainer.save_checkpoint(str(trainer_path))

        # Save stats
        checkpoint = {
            'epoch': epoch,
            'stats': self.stats,
            'config': self.config
        }
        torch.save(checkpoint, checkpoint_path)

        logger.info(f"✓ Checkpoint saved to {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path: str):
        """从checkpoint恢复训练"""
        logger.info(f"📂 Loading checkpoint from {checkpoint_path}")

        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        # Load main checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.start_epoch = checkpoint['epoch'] + 1  # Continue from next epoch
        self.stats = checkpoint.get('stats', self.stats)

        # Load policy
        policy_path = str(checkpoint_path).replace('.pt', '_policy.pt')
        if Path(policy_path).exists():
            self.policy.load_checkpoint(policy_path)
            logger.info(f"✓ Policy loaded from {policy_path}")
        else:
            logger.warning(f"⚠️ Policy checkpoint not found: {policy_path}")

        # Load trainer
        trainer_path = str(checkpoint_path).replace('.pt', '_trainer.pt')
        if Path(trainer_path).exists():
            self.rl_trainer.load_checkpoint(trainer_path)
            logger.info(f"✓ Trainer loaded from {trainer_path}")
        else:
            logger.warning(f"⚠️ Trainer checkpoint not found: {trainer_path}")

        logger.info(f"✅ Resume from epoch {self.start_epoch} (completed: {checkpoint['epoch']})")
        logger.info(f"   Best score so far: {self.stats['best_score']:.4f}")
        logger.info(f"   Total episodes: {self.stats['total_episodes']}")
        logger.info(f"   Total updates: {self.stats['total_updates']}")

    def train(self):
        """主训练循环"""
        logger.info("\n" + "=" * 80)
        logger.info("Starting REAL Workflow Training")
        logger.info("=" * 80)

        # Create environments
        self._create_environments()

        # Resume from checkpoint if specified
        if self.resume_from:
            self.load_checkpoint(self.resume_from)
            logger.info(f"🔄 Resuming training from epoch {self.start_epoch}")

        # Training loop
        for epoch in range(self.start_epoch, self.total_epochs + 1):
            # Train
            epoch_stats = self.train_epoch(epoch)

            # Save checkpoint
            if epoch % self.config.get('save_frequency', 5) == 0:
                self.save_checkpoint(epoch)

            # Save best
            if epoch_stats['avg_score'] > self.stats['best_score']:
                self.stats['best_score'] = epoch_stats['avg_score']
                self.save_checkpoint(epoch, best=True)
                logger.info(f"🎉 NEW BEST SCORE: {self.stats['best_score']:.4f}")

        logger.info("\n" + "=" * 80)
        logger.info("REAL Workflow Training Completed!")
        logger.info(f"Total epochs: {self.total_epochs}")
        logger.info(f"Total episodes: {self.stats['total_episodes']}")
        logger.info(f"Total updates: {self.stats['total_updates']}")
        logger.info(f"Best real workflow score: {self.stats['best_score']:.4f}")
        logger.info("=" * 80)


def main():
    """主入口"""
    parser = argparse.ArgumentParser(description="Real Workflow Deep Integration Training")
    parser.add_argument('--config', type=str, required=True, help="Path to config file")
    parser.add_argument('--resume', type=str, default=None,
                        help="Resume from checkpoint (e.g., 'epoch_2.pt' or 'best.pt')")
    args = parser.parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Replace environment variables in config
    import os
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

    # Handle resume checkpoint path
    resume_checkpoint = None
    if args.resume:
        # If relative path, assume it's in the checkpoint directory
        resume_path = Path(args.resume)
        if not resume_path.is_absolute():
            output_dir = Path(config.get('output_dir', './output/real_workflow'))
            resume_checkpoint = str(output_dir / 'checkpoints' / args.resume)
        else:
            resume_checkpoint = args.resume

    # Create trainer
    trainer = RealWorkflowTrainer(config, resume_from=resume_checkpoint)

    # Train
    trainer.train()


if __name__ == "__main__":
    main()
