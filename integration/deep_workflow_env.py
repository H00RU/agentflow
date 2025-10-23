"""
Deep Workflow Environment - 真正的AFlow Workflow执行环境
Real AFlow workflow execution environment with actual code testing
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
from workflow_parser import WorkflowParser, WorkflowSpec, DatasetClassifier
from workflow_evaluator import WorkflowEvaluator


class DeepWorkflowEnv:
    """
    深度集成的Workflow环境

    功能：
    1. 接收Qwen生成的workflow描述
    2. 解析并生成可执行的workflow代码
    3. 执行真实的HumanEval测试
    4. 返回真实的pass@k分数作为reward
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
        workflow_sample_count: int = None
    ):
        """
        初始化真实workflow环境

        Args:
            dataset: 数据集名称（如"HumanEval"）
            opt_llm_config: 优化LLM配置（GPT-4o，用于workflow生成）
            exec_llm_config: 执行LLM配置（用于运行workflow中的LLM调用）
            operators: 可用的operators列表
            env_num: 并行环境数量
            sample: 每轮测试的样本数
            max_rounds: 最大轮数
            workspace_path: workspace路径（存储workflow代码）
            workflow_sample_count: workflow内部采样数（用于ScEnsemble等）
        """
        self.dataset = dataset
        self.opt_llm_config = opt_llm_config
        self.exec_llm_config = exec_llm_config
        self.operators = operators
        self.env_num = env_num
        self.sample = sample
        self.max_rounds = max_rounds
        self.workflow_sample_count = workflow_sample_count

        # Workspace路径（存储生成的workflow）
        if workspace_path is None:
            aflow_path = os.path.join(os.path.dirname(__file__), '..', 'AFlow')
            self.workspace_path = os.path.join(aflow_path, 'workspace', dataset, 'workflows_rl')
        else:
            self.workspace_path = workspace_path

        os.makedirs(self.workspace_path, exist_ok=True)

        # 创建workflow解析器
        self.workflow_parser = WorkflowParser()

        # 创建evaluator（用于真实测试，根据dataset类型动态选择）
        # 优先尝试使用特定数据集的evaluator，否则使用通用evaluator
        if self._has_custom_evaluator(self.dataset):
            self.evaluator = self._create_custom_evaluator(self.dataset, self.exec_llm_config)
            logger.info(f"[DeepWorkflowEnv] Using custom evaluator for {self.dataset}")
        else:
            self.evaluator = WorkflowEvaluator(
                dataset=self.dataset,
                sample_size=sample,
                timeout_per_problem=30
            )
            logger.info(f"[DeepWorkflowEnv] Using WorkflowEvaluator for {self.dataset}")

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
        logger.info(f"[DeepWorkflowEnv] ✅ REAL WORKFLOW EXECUTION ENABLED")

    def _has_custom_evaluator(self, dataset: str) -> bool:
        """
        检查数据集是否有自定义evaluator

        Args:
            dataset: 数据集名称

        Returns:
            是否有自定义evaluator
        """
        dataset_upper = dataset.upper()
        # 目前只有AIME有自定义evaluator
        return dataset_upper == "AIME"

    def _create_custom_evaluator(self, dataset: str, llm_config: Dict):
        """
        创建数据集特定的自定义evaluator

        Args:
            dataset: 数据集名称
            llm_config: LLM配置

        Returns:
            自定义evaluator实例
        """
        dataset_upper = dataset.upper()

        if dataset_upper == "AIME":
            from aime_evaluator import AIMEEvaluator
            return AIMEEvaluator(
                llm_config=llm_config,
                dataset_path="/content/agentflow/AFlow/data/AIME_2024.jsonl",
                sample_size=self.sample  # 传递配置的 sample 参数
            )
        else:
            raise ValueError(f"No custom evaluator for dataset: {dataset}")

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
        执行step - 这里是真正的workflow执行！

        Args:
            actions: Qwen生成的workflow描述列表

        Returns:
            next_observations: 下一步观测
            rewards: 真实的workflow性能分数（pass@k）
            dones: 是否结束
            info: 额外信息
        """
        self.current_round += 1

        next_observations = []
        rewards = []
        dones = []
        info = []

        logger.info(f"[DeepWorkflowEnv] ===== Round {self.current_round} =====")
        logger.info(f"[DeepWorkflowEnv] Processing {len(actions)} workflow proposals...")

        for i, qwen_action in enumerate(actions):
            try:
                logger.info(f"[DeepWorkflowEnv] Env {i}: Processing Qwen output...")
                logger.info(f"[DeepWorkflowEnv] Env {i}: Action preview: {qwen_action[:200]}...")

                # 1. 解析Qwen输出为workflow规格 (传递dataset类型和采样数)
                workflow_spec = self.workflow_parser.parse_qwen_output(
                    qwen_action,
                    dataset_type=self.dataset,
                    sample_count=self.workflow_sample_count
                )

                if workflow_spec is None:
                    logger.error(f"[DeepWorkflowEnv] Env {i}: Failed to parse Qwen output!")
                    rewards.append(0.0)
                    next_observations.append(self._construct_observation(
                        self.current_round, self.best_score, "Parse failed"
                    ))
                    dones.append(False)
                    info.append({'step': self.current_round, 'error': 'parse_failed'})
                    continue

                logger.info(f"[DeepWorkflowEnv] Env {i}: Parsed workflow:")
                logger.info(f"[DeepWorkflowEnv] Env {i}:   Modification: {workflow_spec.modification}")
                logger.info(f"[DeepWorkflowEnv] Env {i}:   Operators: {workflow_spec.operators}")

                # 2. 保存workflow代码到文件
                round_id = f"{self.current_round}_env{i}"
                workflow_path = self.workflow_parser.save_workflow_to_file(
                    workflow_spec,
                    round_id,
                    self.workspace_path
                )

                logger.info(f"[DeepWorkflowEnv] Env {i}: Workflow code saved to {workflow_path}")

                # 3. 执行真实的workflow测试！
                logger.info(f"[DeepWorkflowEnv] Env {i}: ⚡ EXECUTING REAL WORKFLOW TEST...")
                score = self._execute_workflow_test(round_id, workflow_path)

                self.total_tests_run += 1

                logger.info(f"[DeepWorkflowEnv] Env {i}: ✅ Real test score: {score:.4f}")
                logger.info(f"[DeepWorkflowEnv] Env {i}: This is a REAL pass@k score from HumanEval!")

                # 4. 更新最佳workflow
                if score > self.best_score:
                    logger.info(f"[DeepWorkflowEnv] Env {i}: 🎉 NEW BEST SCORE! {self.best_score:.4f} -> {score:.4f}")
                    self.best_score = score
                    self.best_workflow = workflow_spec

                # 5. 记录历史
                self.workflow_history.append({
                    'round': self.current_round,
                    'env_id': i,
                    'score': score,
                    'workflow_spec': workflow_spec,
                    'workflow_path': workflow_path
                })

                # 6. 返回真实分数作为reward
                reward = float(score)
                rewards.append(reward)

                # 7. 构造下一个观测
                next_obs = self._construct_observation(
                    round_num=self.current_round,
                    best_score=self.best_score,
                    history_summary=self._get_history_summary(),
                    last_score=score
                )
                next_observations.append(next_obs)

                # 8. 判断是否结束
                done = self.current_round >= self.max_rounds
                dones.append(done)

                # 9. Info
                info_dict = {
                    'step': self.current_round,
                    'round': self.current_round,
                    'env_id': i,
                    'score': score,
                    'best_score': self.best_score,
                    'workflow_path': workflow_path,
                    'operators': workflow_spec.operators,
                    'modification': workflow_spec.modification,
                    'is_best': score == self.best_score
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
        logger.info(f"[DeepWorkflowEnv] Total tests run: {self.total_tests_run}")

        return next_observations, rewards, dones, info

    def _execute_workflow_test(self, round_id: str, workflow_path: str) -> float:
        """
        执行真实的workflow测试

        Args:
            round_id: round ID
            workflow_path: workflow代码路径

        Returns:
            真实的pass@k分数（0.0-1.0）
        """
        try:
            # 导入workflow模块
            round_dir = os.path.dirname(workflow_path)
            module_name = f"workspace.{self.dataset}.workflows_rl.round_{round_id}.graph"

            # 动态导入
            spec = importlib.util.spec_from_file_location(module_name, workflow_path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)

            # 获取Workflow类
            WorkflowClass = module.Workflow

            # 创建workflow实例
            workflow = WorkflowClass(
                name=f"RL_Workflow_R{round_id}",
                llm_config=self.exec_llm_config,
                dataset=self.dataset
            )

            # 使用evaluator执行测试
            # 这会真正运行测试任务并返回pass@k
            logger.info(f"[DeepWorkflowEnv] Running real {self.dataset} test with sample={self.sample}...")

            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            # 如果是AIMEEvaluator，需要先初始化
            if hasattr(self.evaluator, 'initialize') and self._has_custom_evaluator(self.dataset):
                if not self.evaluator.problems:  # 只在第一次初始化
                    logger.info(f"[DeepWorkflowEnv] Initializing AIMEEvaluator...")
                    loop.run_until_complete(self.evaluator.initialize())
                    logger.info(f"[DeepWorkflowEnv] AIMEEvaluator initialized with {len(self.evaluator.problems)} problems")

            # 执行评估
            result = loop.run_until_complete(
                self.evaluator.evaluate_workflow(workflow)
            )

            loop.close()

            # result是评估结果dict，提取pass@k分数
            score = result['pass_at_k'] if result and 'pass_at_k' in result else 0.0
            return float(score)

        except Exception as e:
            logger.error(f"[DeepWorkflowEnv] Workflow execution error: {e}")
            import traceback
            traceback.print_exc()
            return 0.0

    def _construct_observation(
        self,
        round_num: int,
        best_score: float,
        history_summary: str,
        last_score: Optional[float] = None
    ) -> str:
        """构造给Qwen的观测"""
        obs = f"""Dataset: {self.dataset}
Task: Design and optimize agent workflow for code generation
Round: {round_num}/{self.max_rounds}

Current Best Score: {best_score:.4f}"""

        if last_score is not None:
            obs += f"\nLast Score: {last_score:.4f}"

        obs += f"""

Available Operators:
{', '.join(self.operators)}

{history_summary}

Your task: Generate a workflow description that will be converted to executable code and tested on real {self.dataset} problems. Focus on:
1. Which operators to use
2. How to combine them effectively
3. How to improve upon previous attempts
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
            summary += f"Score={item['score']:.4f}, "
            summary += f"Operators={item['workflow_spec'].operators}\n"

        return summary

    def close(self):
        """关闭环境"""
        logger.info(f"[DeepWorkflowEnv] Environment closed")
        logger.info(f"[DeepWorkflowEnv] Total tests run: {self.total_tests_run}")
        logger.info(f"[DeepWorkflowEnv] Best score achieved: {self.best_score:.4f}")


def create_deep_workflow_env(dataset, opt_llm_config, exec_llm_config, **kwargs):
    """
    创建深度workflow环境的工厂函数

    这是真正的AFlow集成！
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
        workflow_sample_count=kwargs.get('workflow_sample_count')
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
