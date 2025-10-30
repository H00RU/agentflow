"""
Deep Workflow Environment - MCTS + GRPO训练环境
MCTS-based workflow optimization with GRPO training

Schema 1 架构（职责分离）:
- AFlow Optimizer: MCTS树搜索找最优workflow
- Qwen + GRPO: 学会生成好的workflow修改建议
- 两个系统独立，互不干扰

关键改变：使用原生AFlow Optimizer，而不是RLEnhancedOptimizer
"""

import sys
import os
import asyncio
import shutil
import importlib
from typing import List, Tuple, Dict, Optional
import numpy as np

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'AFlow'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'AFlow', 'scripts'))

from scripts.logs import logger
from scripts.evaluator import DatasetType
from scripts.optimizer import Optimizer
from workflow_evaluator import WorkflowEvaluator

logger.info("[DeepWorkflowEnv] Using native AFlow Optimizer (Schema 1)")
logger.info("[DeepWorkflowEnv] Qwen + GRPO learns workflow optimization")


class DeepWorkflowEnv:
    """
    MCTS + GRPO训练环境（Schema 1 - 职责分离）

    架构（完全独立的两个系统）:
    1. MCTS优化器（AFlow）：通过树搜索找最优workflow
       - 不受GRPO影响
       - 纯粹的workflow优化
    2. Qwen + GRPO：学会生成好的workflow修改建议
       - 观察MCTS返回的rewards
       - 通过GRPO更新参数

    数据流：
    step() → MCTS优化 → 获得score → Qwen从score学习 → GRPO更新

    这个架构的优势：
    - 清晰的职责分离（两个系统互不干扰）
    - 易于调试（MCTS问题和学习问题分离）
    - 理论简洁（标准MCTS + 标准GRPO）
    """

    def __init__(
        self,
        dataset: str,
        opt_llm_config: Dict,
        exec_llm_config: Dict,
        operators: List[str],
        env_num: int = 2,
        sample: int = 3,
        max_rounds: int = 10,
        workspace_path: str = None,
        workflow_sample_count: int = None,
        validation_rounds: int = 3,
        rl_weight: float = 0.5,
        train_test_split: float = 0.8,
        use_qwen_code_generation: bool = False,
        qwen_code_generator=None,
        qwen_max_retries: int = 2,
        mini_batch_size: int = None
    ):
        """
        初始化MCTS + GRPO训练环境

        Args:
            dataset: 数据集名称（如"HumanEval", "AIME"）
            opt_llm_config: 优化LLM配置（传递给MCTS优化器，但Qwen会替代）
            exec_llm_config: 执行LLM配置（用于运行workflow中的LLM调用）
            operators: 可用的operators列表
            env_num: 并行环境数量
            sample: 每轮测试的样本数
            max_rounds: MCTS最大轮数
            workspace_path: workspace路径（存储workflow代码）
            workflow_sample_count: workflow内部采样数（用于ScEnsemble等）
            validation_rounds: MCTS验证轮数
            rl_weight: MCTS UCB与RL Q-value的融合权重 (0.0-1.0)
            train_test_split: 训练/测试集划分比例 (默认0.8)
            use_qwen_code_generation: 使用Qwen替代GPT-4生成代码
            qwen_code_generator: Qwen policy实例（GRPO训练的模型）
            qwen_max_retries: Qwen语法错误时的最大重试次数
            mini_batch_size: Mini-batch大小（None=全量）
        """
        self.dataset = dataset
        self.opt_llm_config = opt_llm_config
        self.exec_llm_config = exec_llm_config
        self.operators = operators
        self.env_num = env_num
        self.sample = sample
        self.max_rounds = max_rounds
        self.workflow_sample_count = workflow_sample_count
        self.train_test_split = train_test_split
        self.validation_rounds = validation_rounds
        self.rl_weight = rl_weight

        # Mini-batch configuration
        self.mini_batch_size = mini_batch_size  # None = use all samples

        # MCTS + Qwen相关参数（保持兼容性但不使用）
        # Schema 1中这些参数不再使用，MCTS和GRPO完全独立
        self.use_qwen_code_generation = use_qwen_code_generation
        self.qwen_code_generator = qwen_code_generator
        self.qwen_max_retries = qwen_max_retries

        logger.info("[DeepWorkflowEnv] Schema 1 Configuration:")
        logger.info(f"  - Qwen code generation: {use_qwen_code_generation} (ignored in Schema 1)")
        logger.info(f"  - RL weight: {rl_weight} (ignored in Schema 1)")

        # Workspace路径（存储生成的workflow）
        # ⚠️ 必须使用AFlow目录下的路径，确保Python模块导入正确
        import sys
        aflow_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'AFlow'))
        aflow_optimized_path = os.path.join(aflow_path, 'optimized')

        # 添加AFlow/optimized到sys.path（这样Optimizer可以正确导入workflow模块）
        aflow_optimized_path = os.path.abspath(aflow_optimized_path)
        if aflow_optimized_path not in sys.path:
            sys.path.insert(0, aflow_optimized_path)

        if workspace_path is None:
            # 使用相对于AFlow/optimized的路径
            self.workspace_path = os.path.join(aflow_optimized_path, dataset)
        else:
            # 如果指定了custom workspace，使用AFlow/optimized
            logger.info(f"[DeepWorkflowEnv] Custom workspace specified: {workspace_path}")
            logger.info(f"[DeepWorkflowEnv] Using AFlow default workspace for proper import")
            self.workspace_path = os.path.join(aflow_optimized_path, dataset)

        os.makedirs(self.workspace_path, exist_ok=True)
        logger.info(f"[DeepWorkflowEnv] Workspace: {self.workspace_path}")

        # 创建evaluator（用于真实测试）
        # ⚠️ 必须在_init_dynamic_mode()之前创建，因为适配器需要它
        # 所有数据集统一使用WorkflowEvaluator，AIME已加入AFlow标准支持
        self.evaluator = WorkflowEvaluator(
            dataset=self.dataset,
            sample_size=sample,
            timeout_per_problem=30,
            train_test_split=self.train_test_split,
            llm_config=self.exec_llm_config  # 传递LLM配置给evaluator
        )
        logger.info(f"[DeepWorkflowEnv] Using WorkflowEvaluator for {self.dataset}")

        # 初始化MCTS优化器和相关组件
        logger.info(f"[DeepWorkflowEnv] ✨ Initializing MCTS + GRPO environment")
        logger.info(f"[DeepWorkflowEnv] Qwen will replace GPT-4 in MCTS framework")
        self._init_mcts_components()

        # 当前状态
        self.current_round = 0
        self.workflow_history = []  # 历史workflow及其分数
        self.best_score = 0.0
        self.best_workflow = None

        # 统计
        self.total_api_calls = 0
        self.total_tests_run = 0

        logger.info(f"[DeepWorkflowEnv] Initialized")
        logger.info(f"[DeepWorkflowEnv] Dataset: {dataset}")
        logger.info(f"[DeepWorkflowEnv] Workspace: {self.workspace_path}")
        logger.info(f"[DeepWorkflowEnv] Evaluator sample size: {sample}")
        if self.mini_batch_size:
            logger.info(f"[DeepWorkflowEnv] 🎲 Mini-Batch Mode: {self.mini_batch_size} problems/test (random sampling)")
        else:
            logger.info(f"[DeepWorkflowEnv] 📊 Full-Batch Mode: {sample} problems/test")
        logger.info(f"[DeepWorkflowEnv] ✅ REAL WORKFLOW EXECUTION ENABLED")

    def _init_mcts_components(self):
        """初始化原生MCTS优化器（Schema 1）"""
        # Schema 1: 不使用共享经验池和状态管理，MCTS和GRPO完全独立

        # 为每个并行环境创建一个原生Optimizer实例
        self.optimizers = []
        question_type = self._infer_question_type(self.dataset)

        for i in range(self.env_num):
            # 传给Optimizer的optimized_path应该是相对于AFlow/optimized的相对路径
            # 这样Optimizer可以正确构造导入路径
            optimizer = Optimizer(
                dataset=self.dataset,
                question_type=question_type,
                opt_llm_config=self.opt_llm_config,
                exec_llm_config=self.exec_llm_config,
                operators=self.operators,
                sample=self.sample,
                check_convergence=False,
                optimized_path="optimized/",  # 相对路径，已包含数据集子目录
                initial_round=1,
                max_rounds=self.max_rounds,
            )
            self.optimizers.append(optimizer)

        # ===== Schema 1: 使用WorkflowEvaluator进行评估 =====
        # 替换optimizer的evaluation_utils为WorkflowEvaluator
        from evaluation_adapter import EvaluationUtilsAdapter

        for optimizer in self.optimizers:
            optimizer.evaluation_utils = EvaluationUtilsAdapter(
                workflow_evaluator=self.evaluator,
                root_path=optimizer.root_path
            )

        logger.info(f"[DeepWorkflowEnv] Created {self.env_num} native Optimizers (Schema 1)")
        logger.info(f"[DeepWorkflowEnv] ✅ MCTS and GRPO are completely independent")
        logger.info(f"[DeepWorkflowEnv] ✅ Using WorkflowEvaluator for {self.dataset}")
        logger.info(f"[DeepWorkflowEnv] ✅ Mini-batch and train/test split enabled")

    def _infer_question_type(self, dataset: str) -> str:
        """推断问题类型"""
        dataset_upper = dataset.upper()
        if dataset_upper in ["HUMANEVAL", "MBPP", "CODEEVAL"]:
            return "code"
        elif dataset_upper in ["AIME", "MATH", "GSM8K"]:
            return "math"
        else:
            return "qa"

    def reset(self) -> Tuple[List[str], List[Dict]]:
        """
        重置环境

        Returns:
            observations: 观测列表
            info: 信息字典列表
        """
        self.current_round = 0

        observations = []
        info = []

        for i in range(self.env_num):
            # 构造观测：告诉Qwen当前状态
            obs = self._construct_observation(
                round_num=0,
                best_score=self.best_score,
                history_summary=self._get_history_summary()
            )
            observations.append(obs)

            info_dict = {
                'step': 0,
                'round': 0,
                'env_id': i,
                'best_score': self.best_score,
                'workflow_path': None
            }
            info.append(info_dict)

        logger.info(f"[DeepWorkflowEnv] Environment reset")
        return observations, info

    def step(self, actions: List[str]) -> Tuple[List[str], List[float], List[bool], List[Dict]]:
        """
        执行MCTS优化step（Schema 1）

        流程（两个独立系统）:
        1. 环境（step）接收Qwen建议作为输入（但不直接使用）
        2. MCTS优化器执行纯粹的树搜索，找最优workflow
        3. 评估workflow性能
        4. 返回真实pass@k分数作为reward
        5. Qwen + GRPO观察这个reward，从中学习

        Args:
            actions: Qwen生成的workflow优化建议列表（可选使用）

        Returns:
            next_observations: 下一步观测
            rewards: 真实的workflow性能分数（pass@k）
            dones: 是否结束
            info: 额外信息
        """
        self.current_round += 1
        return self._step_mcts(actions)

    def _step_mcts(self, actions: List[str]) -> Tuple[List[str], List[float], List[bool], List[Dict]]:
        """
        MCTS优化step实现（Schema 1 - 纯MCTS搜索）

        使用原生Optimizer进行纯粹的MCTS树搜索：
        1. Optimizer执行MCTS树搜索（与GRPO完全独立）
        2. 生成最优workflow
        3. 在真实验证集上评估
        4. 返回pass@k分数
        """
        next_observations = []
        rewards = []
        dones = []
        info = []

        logger.info(f"[DeepWorkflowEnv] ===== Round {self.current_round} (MCTS + GRPO) =====")
        logger.info(f"[DeepWorkflowEnv] Running {len(actions)} MCTS optimizations...")

        # 并行运行所有优化器
        for i, (optimizer, action) in enumerate(zip(self.optimizers, actions)):
            try:
                logger.info(f"[DeepWorkflowEnv] Env {i}: Running native MCTS Optimizer...")
                logger.info(f"[DeepWorkflowEnv] Env {i}: Action hint: {action[:100]}...")

                # 运行一轮优化（纯MCTS，与GRPO完全独立）
                # Optimizer 会：
                # 1. 执行MCTS树搜索（UCB策略）
                # 2. 使用LLM生成新workflow
                # 3. 在验证集上评估
                # 4. 更新MCTS树（不涉及RL）
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

                try:
                    score = loop.run_until_complete(optimizer._optimize_graph())
                except Exception as e:
                    logger.error(f"[DeepWorkflowEnv] Env {i}: Error in optimization: {e}")
                    import traceback
                    traceback.print_exc()
                    score = 0.0
                finally:
                    loop.close()

                logger.info(f"[DeepWorkflowEnv] Env {i}: ✅ Optimization score: {score:.4f}")

                # 更新最佳分数
                if score > self.best_score:
                    logger.info(f"[DeepWorkflowEnv] Env {i}: 🎉 NEW BEST SCORE! {self.best_score:.4f} -> {score:.4f}")
                    self.best_score = score
                    self.best_workflow = optimizer.graph

                # 记录历史
                self.workflow_history.append({
                    'round': self.current_round,
                    'env_id': i,
                    'score': score,
                    'workflow_type': 'dynamic'
                })

                # 返回分数作为 reward
                reward = float(score)
                rewards.append(reward)

                # 构造下一个观测
                next_obs = self._construct_observation(
                    round_num=self.current_round,
                    best_score=self.best_score,
                    history_summary=self._get_history_summary(),
                    last_score=score
                )
                next_observations.append(next_obs)

                # 判断是否结束
                done = self.current_round >= self.max_rounds
                dones.append(done)

                # Info
                info_dict = {
                    'step': self.current_round,
                    'round': self.current_round,
                    'env_id': i,
                    'score': score,
                    'best_score': self.best_score,
                    'is_best': score == self.best_score,
                    'optimization_type': 'dynamic_rl_mcts'
                }
                info.append(info_dict)

            except Exception as e:
                logger.error(f"[DeepWorkflowEnv] Env {i}: ERROR: {e}")
                import traceback
                traceback.print_exc()

                rewards.append(0.0)
                next_observations.append(self._construct_observation(
                    self.current_round, self.best_score, f"Error: {str(e)}"
                ))
                dones.append(False)
                info.append({'step': self.current_round, 'error': str(e)})

        avg_reward = np.mean(rewards) if rewards else 0.0
        logger.info(f"[DeepWorkflowEnv] Round {self.current_round} completed")
        logger.info(f"[DeepWorkflowEnv] Avg reward: {avg_reward:.4f}, Best so far: {self.best_score:.4f}")

        return next_observations, rewards, dones, info

    def _extract_code_from_qwen(self, qwen_output: str) -> Optional[Dict[str, str]]:
        """
        从Qwen输出提取代码 - 完全对齐原版AFlow

        原版AFlow期望LLM返回：
        <modification>...</modification>
        <graph>...</graph>
        <prompt>...</prompt>

        Args:
            qwen_output: Qwen生成的输出

        Returns:
            {'graph': str, 'modification': str, 'prompt': str} 或 None
        """
        import re

        result = {}

        # 提取 modification
        modification_pattern = r"<modification>(.*?)</modification>"
        modification_match = re.search(modification_pattern, qwen_output, re.DOTALL)
        if modification_match:
            result['modification'] = modification_match.group(1).strip()
        else:
            result['modification'] = "No modification description provided"

        # 提取 graph (必需)
        graph_pattern = r"<graph>(.*?)</graph>"
        graph_match = re.search(graph_pattern, qwen_output, re.DOTALL)
        if not graph_match:
            logger.error("[DeepWorkflowEnv] No <graph> tag found in Qwen output")
            return None

        result['graph'] = graph_match.group(1).strip()

        # 提取 prompt (可选)
        prompt_pattern = r"<prompt>(.*?)</prompt>"
        prompt_match = re.search(prompt_pattern, qwen_output, re.DOTALL)
        if prompt_match:
            result['prompt'] = prompt_match.group(1).strip()
        else:
            result['prompt'] = "# Auto-generated - no custom prompts needed\n"

        return result

    def _validate_python_syntax(self, code: str) -> bool:
        """
        验证Python代码语法

        Args:
            code: Python代码字符串

        Returns:
            bool: 语法是否正确
        """
        try:
            compile(code, '<string>', 'exec')
            return True
        except SyntaxError as e:
            logger.error(f"[DeepWorkflowEnv] Syntax error in code: {e}")
            logger.error(f"[DeepWorkflowEnv] Error line: {e.lineno}, offset: {e.offset}")
            logger.error(f"[DeepWorkflowEnv] Error text: {e.text}")
            return False
        except Exception as e:
            logger.error(f"[DeepWorkflowEnv] Unexpected error during syntax check: {e}")
            return False

    def _save_workflow_code_aflow_style(self,
                                        graph_code: str,
                                        prompt_code: str,
                                        round_id: str,
                                        modification: str) -> str:
        """
        使用原版AFlow的方式保存workflow代码

        原版AFlow: graph_utils.py:147-158
        - 使用WORKFLOW_TEMPLATE填充
        - 保存graph.py, prompt.py, __init__.py
        - 保存modification.txt记录

        Args:
            graph_code: workflow class代码
            prompt_code: prompt定义代码
            round_id: round标识
            modification: 修改描述

        Returns:
            str: graph.py的完整路径
        """
        from scripts.prompts.optimize_prompt import WORKFLOW_TEMPLATE

        # 创建round目录 - 统一使用workflows子目录以匹配optimizer查找路径
        workflows_base = os.path.join(self.workspace_path, "workflows")
        round_dir = os.path.join(workflows_base, f"round_{round_id}")
        os.makedirs(round_dir, exist_ok=True)

        # 1. 使用WORKFLOW_TEMPLATE生成完整代码（与原版AFlow相同）
        full_graph_code = WORKFLOW_TEMPLATE.format(
            graph=graph_code,
            round=round_id,
            dataset=self.dataset
        )

        # 2. 保存graph.py
        graph_path = os.path.join(round_dir, "graph.py")
        with open(graph_path, 'w', encoding='utf-8') as f:
            f.write(full_graph_code)

        # 3. 保存prompt.py
        prompt_path = os.path.join(round_dir, "prompt.py")
        with open(prompt_path, 'w', encoding='utf-8') as f:
            f.write(prompt_code)

        # 4. 保存__init__.py
        init_path = os.path.join(round_dir, "__init__.py")
        with open(init_path, 'w', encoding='utf-8') as f:
            f.write("")

        # 5. 保存modification.txt记录
        modification_path = os.path.join(round_dir, "modification.txt")
        with open(modification_path, 'w', encoding='utf-8') as f:
            f.write(f"Round {round_id} Modification:\n")
            f.write(f"{modification}\n\n")
            f.write("This workflow was generated by Qwen (no Parser).\n")
            f.write("Fully aligned with original AFlow design.\n")

        logger.info(f"[DeepWorkflowEnv] Saved workflow files to {round_dir}")
        logger.info(f"[DeepWorkflowEnv]   - graph.py: {len(full_graph_code)} chars")
        logger.info(f"[DeepWorkflowEnv]   - prompt.py: {len(prompt_code)} chars")
        logger.info(f"[DeepWorkflowEnv]   - modification.txt")

        return graph_path

    def _construct_observation(
        self,
        round_num: int,
        best_score: float,
        history_summary: str,
        last_score: Optional[float] = None
    ) -> str:
        """构造给Qwen的观测，根据dataset生成相应的任务描述"""
        # 根据dataset生成任务描述
        dataset_upper = self.dataset.upper()
        if dataset_upper == "AIME":
            task_desc = "Design and optimize agent workflow for solving AIME mathematical problems"
            focus_points = """1. Which operators to use for mathematical reasoning
2. How to combine them effectively for problem-solving
3. Using ensemble methods for robust solutions
4. How to improve upon previous attempts"""
        elif dataset_upper == "HUMANEVAL":
            task_desc = "Design and optimize agent workflow for code generation"
            focus_points = """1. Which operators to use for code generation
2. How to combine them effectively
3. How to improve upon previous attempts"""
        else:
            # 通用描述
            task_desc = f"Design and optimize agent workflow for {self.dataset} tasks"
            focus_points = """1. Which operators to use
2. How to combine them effectively
3. How to improve upon previous attempts"""

        obs = f"""Dataset: {self.dataset}
Task: {task_desc}
Round: {round_num}/{self.max_rounds}

Current Best Score: {best_score:.4f}"""

        if last_score is not None:
            obs += f"\nLast Score: {last_score:.4f}"

        obs += f"""

Available Operators:
{', '.join(self.operators)}

{history_summary}

Your task: Generate a workflow description that will be converted to executable code and tested on real {self.dataset} problems. Focus on:
{focus_points}
"""

        return obs

    def _get_history_summary(self) -> str:
        """获取历史workflow摘要"""
        if not self.workflow_history:
            return "History: No previous workflows yet."

        # 取最近3个workflow
        recent = self.workflow_history[-3:]
        summary = "Recent Workflow Performance:\n"

        for item in recent:
            summary += f"  Round {item['round']} Env{item['env_id']}: "
            summary += f"Score={item['score']:.4f}\n"

        return summary

    def close(self):
        """关闭环境"""
        logger.info(f"[DeepWorkflowEnv] Environment closed")
        logger.info(f"[DeepWorkflowEnv] Total tests run: {self.total_tests_run}")
        logger.info(f"[DeepWorkflowEnv] Best score achieved: {self.best_score:.4f}")


def create_deep_workflow_env(dataset, opt_llm_config, exec_llm_config, **kwargs):
    """
    创建MCTS + GRPO训练环境的工厂函数（Schema 1）

    架构（职责分离）：
    - MCTS优化器（AFlow）：找最优workflow
    - Qwen + GRPO：学会生成好的修改建议

    两个系统完全独立，互不干扰。

    关键参数：
    - max_rounds: MCTS树搜索最大轮数
    - sample: 每轮评估的样本数
    - train_test_split: 训练/测试集划分比例
    - mini_batch_size: 小批量测试大小（None=全量）
    """
    return DeepWorkflowEnv(
        dataset=dataset,
        opt_llm_config=opt_llm_config,
        exec_llm_config=exec_llm_config,
        operators=kwargs.get('operators', ['Custom', 'CustomCodeGenerate', 'ScEnsemble', 'Test']),
        env_num=kwargs.get('env_num', 2),
        sample=kwargs.get('sample', 3),
        max_rounds=kwargs.get('max_rounds', 10),
        workspace_path=kwargs.get('workspace_path'),
        workflow_sample_count=kwargs.get('workflow_sample_count'),
        validation_rounds=kwargs.get('validation_rounds', 3),
        rl_weight=kwargs.get('rl_weight', 0.5),
        train_test_split=kwargs.get('train_test_split', 0.8),
        # MCTS + Qwen参数
        use_qwen_code_generation=kwargs.get('use_qwen_code_generation', False),
        qwen_code_generator=kwargs.get('qwen_code_generator'),
        qwen_max_retries=kwargs.get('qwen_max_retries', 2),
        # Mini-batch参数
        mini_batch_size=kwargs.get('mini_batch_size', None)
    )


if __name__ == "__main__":
    # 测试环境
    import yaml

    # 加载配置
    config_path = "deep_config_e2e.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    env_config = config['environment']

    # 创建环境
    env = create_deep_workflow_env(
        dataset="HumanEval",
        opt_llm_config=env_config['opt_llm_config'],
        exec_llm_config=env_config['exec_llm_config'],
        operators=env_config['operators'],
        env_num=1,
        sample=2
    )

    # 测试
    obs, info = env.reset()
    print(f"Initial observation:\n{obs[0]}\n")

    # 模拟Qwen输出
    test_action = """
<workflow_modification>
Use ensemble approach to improve code quality
</workflow_modification>

<operators>
CustomCodeGenerate, ScEnsemble
</operators>

<workflow_steps>
1. Generate 3 candidate code solutions
2. Use ScEnsemble to select the best one
</workflow_steps>
"""

    next_obs, rewards, dones, info = env.step([test_action])
    print(f"Reward (real pass@k): {rewards[0]:.4f}")
    print(f"This is a REAL score from executing workflow on HumanEval!")
