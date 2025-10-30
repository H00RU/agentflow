"""
Schema 1 MCTS Optimization Script - 方案一纯MCTS优化
Pure MCTS-based workflow optimization using native AFlow
No RL fusion, clean separation of concerns

架构（职责分离）:
- MCTS optimizer (AFlow): 找最优workflow
- Environment: 执行MCTS并返回真实评估结果
"""

import os
import sys
import argparse
import yaml
import time
from typing import Dict, Any
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
    from scripts.logs import logger
    from deep_workflow_env import create_deep_workflow_env

    IMPORTS_AVAILABLE = True
except ImportError as e:
    print(f"Error: Could not import required components: {e}")
    import traceback
    traceback.print_exc()
    IMPORTS_AVAILABLE = False


class RealWorkflowTrainer:
    """
    Schema 1 MCTS Optimization器

    纯粹的MCTS-based优化：
    1. 创建MCTS优化环境
    2. 多轮MCTS树搜索
    3. 评估workflow真实性能
    4. 保存最佳workflow
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

        # Paths - 使用本地路径（避免Python import问题）
        self.output_dir = Path(config.get('output_dir', './output/real_workflow'))
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.checkpoint_dir = self.output_dir / 'checkpoints'
        self.checkpoint_dir.mkdir(exist_ok=True)

        self.log_dir = self.output_dir / 'logs'
        self.log_dir.mkdir(exist_ok=True)

        self.workflow_dir = self.output_dir / 'workflows_generated'
        self.workflow_dir.mkdir(exist_ok=True)

        # Environment configuration (Schema 1: only MCTS, no RL)
        self.env_config = config.get('environment', {})
        self.dataset = self.env_config.get('dataset', 'AIME')

        logger.info(f"[Schema 1 Config] Dataset: {self.dataset}")
        logger.info(f"[Schema 1 Config] Max rounds: {self.env_config.get('max_rounds', 10)}")
        logger.info(f"[Schema 1 Config] Sample size: {self.env_config.get('sample', 3)}")

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
            'workflow_history': []
        }

        logger.info("RealWorkflowTrainer initialized successfully")
        logger.info("✅ READY FOR SCHEMA 1 MCTS OPTIMIZATION")

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

        import importlib.util
        import asyncio
        import os

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

            # 步骤2: 使用训练好的policy生成workflow代码（无Parser）
            logger.info("[Trainer] Generating workflow CODE using trained policy...")
            workflow_output, _, _, _, _ = self.policy.get_action_and_value(
                obs=test_obs,
                max_new_tokens=800,  # 增加token数以容纳完整代码
                temperature=0.7
            )
            logger.info(f"[Trainer] Policy generated workflow:")
            logger.info(f"[Trainer] {workflow_output[:300]}...")  # 打印前300字符

            # 步骤3: 提取代码（无Parser）
            logger.info("[Trainer] Extracting code from policy output...")
            extraction_result = env._extract_code_from_qwen(workflow_output)

            if extraction_result is None:
                logger.warning("[Trainer] Failed to extract code from policy output, falling back to best_workflow")
                return self._evaluate_fallback_workflow(env, num_test_problems)

            graph_code = extraction_result['graph']
            modification = extraction_result['modification']
            prompt_code = extraction_result.get('prompt', '')

            logger.info(f"[Trainer] ✓ Extracted workflow code")
            logger.info(f"[Trainer]   Modification: {modification}")
            logger.info(f"[Trainer]   Code length: {len(graph_code)} chars")

            # 步骤3.5: 验证语法
            if not env._validate_python_syntax(graph_code):
                logger.warning("[Trainer] Syntax error in generated code, falling back to best_workflow")
                return self._evaluate_fallback_workflow(env, num_test_problems)

            # 步骤4: 保存workflow到文件（使用AFlow方式）
            test_workflow_path = env._save_workflow_code_aflow_style(
                graph_code=graph_code,
                prompt_code=prompt_code,
                round_id="policy_test_eval",
                modification=modification
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

## Output Format:
Generate a COMPLETE Python workflow using the XML format below:

<modification>
Brief description of your workflow design and why it should work well on test problems.
Example: "Use ensemble of 15 samples with custom reasoning and review steps"
</modification>

<graph>
class Workflow:
    def __init__(self, name: str, llm_config, dataset: str) -> None:
        from scripts.async_llm import create_llm_instance
        self.name = name
        self.dataset = dataset
        self.llm = create_llm_instance(llm_config)

        # Initialize operators
        from scripts import operator
        self.custom = operator.Custom(self.llm)
        self.sc_ensemble = operator.ScEnsemble(self.llm)
        # Add more operators as needed

    async def __call__(self, problem: str, entry_point=None):
        # YOUR COMPLETE WORKFLOW LOGIC HERE
        # MUST return (solution, cost) tuple
        return solution, 0.0
</graph>

<prompt>
# Custom prompts if needed (optional)
</prompt>

CRITICAL: Code must be syntactically correct Python and return (solution, cost) tuple.
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

        import importlib.util
        import asyncio

        test_dataset = self.train_datasets[0] if self.train_datasets else "AIME"

        # best_workflow现在是字典：{'graph': code, 'modification': str, ...}
        # 保存临时workflow（使用AFlow方式）
        test_workflow_path = env._save_workflow_code_aflow_style(
            graph_code=env.best_workflow['graph'],
            prompt_code=env.best_workflow.get('prompt', ''),
            round_id="fallback_test_eval",
            modification=env.best_workflow.get('modification', 'Best workflow from training')
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

        # MCTS + GRPO 配置
        validation_rounds = self.env_config.get('validation_rounds', 3)
        rl_weight = self.env_config.get('rl_weight', 0.5)

        # MCTS + Qwen代码生成配置
        use_qwen_code_generation = self.env_config.get('use_qwen_code_generation', False)
        qwen_max_retries = self.env_config.get('qwen_max_retries', 2)

        # Mini-batch配置
        mini_batch_size = self.env_config.get('mini_batch_size', None)

        # Create training environment (Schema 1: 单数据集)
        logger.info(f"Creating MCTS Optimization environment for {self.dataset}")

        # Create MCTS environment
        env = create_deep_workflow_env(
            dataset=self.dataset,
            opt_llm_config=opt_llm_config,
            exec_llm_config=exec_llm_config,
            operators=operators,
            env_num=env_num,
            sample=sample,
            max_rounds=max_rounds,
            workspace_path=str(self.workflow_dir),
            train_test_split=train_test_split,
            validation_rounds=validation_rounds,
            # Schema 1: 以下参数不使用，但保持兼容性
            rl_weight=0.0,
            use_qwen_code_generation=False,
            qwen_code_generator=None,
            qwen_max_retries=2,
            mini_batch_size=mini_batch_size
        )

        logger.info(f"✅ MCTS Optimization Environment created (Schema 1)")
        logger.info(f"   Dataset: {self.dataset}")
        logger.info(f"   Mode: Pure MCTS (no RL fusion)")
        logger.info(f"   Max rounds: {max_rounds}")
        logger.info(f"   Sample size: {sample}")
        logger.info(f"   Evaluation: Real {self.dataset} execution")
        logger.info(f"   Reward: Real pass@k scores")

        self.train_envs[self.dataset] = env
        print(f"\n✓ Created MCTS Optimization environment")

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Schema 1: MCTS优化的一个epoch

        纯粹的MCTS树搜索，不涉及RL policy训练
        """
        logger.info(f"\n{'=' * 80}")
        logger.info(f"Epoch {epoch}/{self.total_epochs} (Schema 1 - Pure MCTS)")
        logger.info(f"{'=' * 80}")

        epoch_stats = {
            'total_episodes': 0,
            'avg_score': 0.0,
            'max_score': 0.0,
        }

        # Run MCTS optimization on each dataset
        for dataset, env in self.train_envs.items():
            logger.info(f"\n🔍 MCTS optimization on {dataset}")
            logger.info(f"   Running {self.episodes_per_epoch} episodes with MCTS tree search")

            episode_scores = []

            # Run multiple optimization rounds
            for episode in range(1, self.episodes_per_epoch + 1):
                logger.info(f"\n[Episode {episode}/{self.episodes_per_epoch}] Running MCTS optimization...")

                # Reset environment
                obs, info = env.reset()
                logger.info(f"  Environment reset for MCTS")

                # Dummy action (Schema 1: MCTS doesn't use external actions)
                actions = ["Run MCTS optimization"] * len(obs)

                # Execute MCTS step
                try:
                    next_obs, rewards, dones, info_dict = env.step(actions)

                    avg_reward = float(np.mean(rewards)) if rewards else 0.0
                    episode_scores.append(avg_reward)

                    logger.info(f"  ✅ MCTS complete - Score: {avg_reward:.4f}")
                    epoch_stats['total_episodes'] += 1

                except Exception as e:
                    logger.error(f"  ❌ Error in MCTS step: {e}")
                    import traceback
                    traceback.print_exc()

            # Calculate epoch statistics
            if episode_scores:
                epoch_stats['avg_score'] = float(np.mean(episode_scores))
                epoch_stats['max_score'] = float(np.max(episode_scores))
            else:
                epoch_stats['avg_score'] = 0.0
                epoch_stats['max_score'] = 0.0

        # Update global statistics
        self.stats['epoch'] = epoch
        self.stats['total_episodes'] += epoch_stats['total_episodes']
        self.stats['avg_scores'].append(epoch_stats['avg_score'])

        logger.info(f"\n{'=' * 80}")
        logger.info(f"Epoch {epoch} Summary:")
        logger.info(f"  Total episodes: {epoch_stats['total_episodes']}")
        logger.info(f"  Avg MCTS score: {epoch_stats['avg_score']:.4f}")
        logger.info(f"  Max MCTS score: {epoch_stats['max_score']:.4f}")
        logger.info(f"{'=' * 80}\n")

        return epoch_stats

    def save_checkpoint(self, epoch: int, best: bool = False):
        """保存checkpoint (Schema 1 - 仅保存统计信息)"""
        checkpoint_name = f"best.pt" if best else f"epoch_{epoch}.pt"
        checkpoint_path = self.checkpoint_dir / checkpoint_name

        # Save stats
        checkpoint = {
            'epoch': epoch,
            'stats': self.stats,
            'config': self.config
        }
        torch.save(checkpoint, checkpoint_path)

        logger.info(f"✓ Checkpoint saved to {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path: str):
        """从checkpoint恢复训练 (Schema 1)"""
        logger.info(f"📂 Loading checkpoint from {checkpoint_path}")

        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        # Load main checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.start_epoch = checkpoint['epoch'] + 1  # Continue from next epoch
        self.stats = checkpoint.get('stats', self.stats)

        logger.info(f"✅ Resume from epoch {self.start_epoch} (completed: {checkpoint['epoch']})")
        logger.info(f"   Best score so far: {self.stats['best_score']:.4f}")
        logger.info(f"   Total episodes: {self.stats['total_episodes']}")

    def train(self):
        """主训练循环 - Schema 1 MCTS优化"""
        logger.info("\n" + "=" * 80)
        logger.info("Starting Schema 1 MCTS Optimization Training")
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
        logger.info("Schema 1 MCTS Optimization Training Completed!")
        logger.info(f"Total epochs: {self.total_epochs}")
        logger.info(f"Total episodes: {self.stats['total_episodes']}")
        logger.info(f"Best MCTS workflow score: {self.stats['best_score']:.4f}")
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
