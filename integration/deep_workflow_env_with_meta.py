"""
Deep Workflow Environment with Meta Learning - 集成元学习的workflow环境
Enhanced version of deep_workflow_env.py with meta-learning integration

主要改进：
1. 使用元学习选择器替代硬编码的parser
2. 自动记录每次执行经验
3. 支持越练越强
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
from workflow_parser import WorkflowParser, WorkflowSpec
from workflow_evaluator import WorkflowEvaluator

# 导入元学习组件
from meta_learning_integration import MetaLearningIntegration


class DeepWorkflowEnvWithMeta:
    """
    深度集成的Workflow环境 + 元学习

    新功能：
    1. 使用元学习选择器智能选择operators
    2. 自动记录执行经验
    3. 越练越强的operator选择
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
        enable_meta_learning: bool = True,  # 新增：是否启用元学习
        meta_save_dir: str = None  # 新增：元学习保存目录
    ):
        """
        初始化真实workflow环境 + 元学习

        Args:
            dataset: 数据集名称（如"HumanEval"）
            opt_llm_config: 优化LLM配置
            exec_llm_config: 执行LLM配置
            operators: 可用的operators列表（元学习模式下作为参考）
            env_num: 并行环境数量
            sample: 每轮测试的样本数
            max_rounds: 最大轮数
            workspace_path: workspace路径
            enable_meta_learning: 是否启用元学习
            meta_save_dir: 元学习数据保存目录
        """
        self.dataset = dataset
        self.opt_llm_config = opt_llm_config
        self.exec_llm_config = exec_llm_config
        self.operators = operators
        self.env_num = env_num
        self.sample = sample
        self.max_rounds = max_rounds
        self.enable_meta_learning = enable_meta_learning

        # Workspace路径
        if workspace_path is None:
            aflow_path = os.path.join(os.path.dirname(__file__), '..', 'AFlow')
            self.workspace_path = os.path.join(aflow_path, 'workspace', dataset, 'workflows_rl')
        else:
            self.workspace_path = workspace_path

        os.makedirs(self.workspace_path, exist_ok=True)

        # 元学习集成（新增）
        if self.enable_meta_learning:
            self.meta_integration = MetaLearningIntegration(
                save_dir=meta_save_dir,
                enable_meta_learning=True,
                enable_adaptation=False  # 环境中不使用自适应，只使用选择器
            )
            logger.info("[DeepWorkflowEnvWithMeta] ✅ Meta learning ENABLED")
        else:
            self.meta_integration = None
            logger.info("[DeepWorkflowEnvWithMeta] ⚠️  Meta learning DISABLED")

        # 保留原始parser作为fallback
        self.fallback_parser = WorkflowParser()

        # 创建evaluator
        if self.dataset.upper() == "AIME":
            from aime_evaluator import AIMEEvaluator
            import asyncio
            self.evaluator = AIMEEvaluator(
                llm_config=self.exec_llm_config,
                dataset_path="/content/agentflow/AFlow/data/AIME_2024.jsonl"
            )
            # AIMEEvaluator需要异步初始化加载数据集
            asyncio.run(self.evaluator.initialize())
            logger.info(f"[DeepWorkflowEnvWithMeta] Using AIMEEvaluator")
        else:
            self.evaluator = WorkflowEvaluator(
                dataset=self.dataset,
                sample_size=sample,
                timeout_per_problem=30
            )
            logger.info(f"[DeepWorkflowEnvWithMeta] Using WorkflowEvaluator for {self.dataset}")

        # 当前状态
        self.current_round = 0
        self.workflow_history = []
        self.best_score = 0.0
        self.best_workflow = None

        # 统计（新增：用于记录元学习信息）
        self.execution_history = []  # 记录每次执行的详细信息

        # 统计
        self.total_api_calls = 0
        self.total_tests_run = 0

        logger.info(f"[DeepWorkflowEnvWithMeta] Initialized")
        logger.info(f"[DeepWorkflowEnvWithMeta] Dataset: {dataset}")
        logger.info(f"[DeepWorkflowEnvWithMeta] Workspace: {self.workspace_path}")

    def reset(self) -> Tuple[List[str], List[Dict]]:
        """重置环境"""
        self.current_round = 0

        observations = []
        info = []

        for i in range(self.env_num):
            obs = self._construct_observation(
                round_num=0,
                best_score=self.best_score,
                history=self._get_history_summary()
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

        logger.info(f"[DeepWorkflowEnvWithMeta] Environment reset")
        return observations, info

    def step(self, actions: List[str]) -> Tuple[List[str], List[float], List[bool], List[Dict]]:
        """
        执行step - 使用元学习选择器

        主要改进：
        1. 使用元学习选择operators（而不是Qwen输出）
        2. 记录执行经验
        """
        import time
        self.current_round += 1

        next_observations = []
        rewards = []
        dones = []
        info = []

        logger.info(f"[DeepWorkflowEnvWithMeta] ===== Round {self.current_round} =====")

        for i, qwen_action in enumerate(actions):
            start_time = time.time()

            try:
                logger.info(f"[DeepWorkflowEnvWithMeta] Env {i}: Processing action...")

                # ===== 关键改进：使用元学习选择器 =====
                if self.enable_meta_learning:
                    # 方案1：直接从Qwen action中提取问题描述（如果有）
                    # 或者使用一个通用的问题描述
                    problem_desc = self._extract_problem_desc(qwen_action)

                    # 使用元学习选择器选择operators
                    selection = self.meta_integration.parse_and_select_operators(
                        problem=problem_desc,
                        dataset_type=self.dataset
                    )

                    logger.info(f"[DeepWorkflowEnvWithMeta] Env {i}: Meta selector chose:")
                    logger.info(f"  Strategy: {selection['strategy_name']}")
                    logger.info(f"  Operators: {selection['operators']}")
                    logger.info(f"  Confidence: {selection['confidence']:.3f}")

                    # 使用选择的operators生成workflow代码
                    workflow_code = selection.get('workflow_code')
                    if workflow_code:
                        workflow_spec = WorkflowSpec(
                            modification=f"Meta-learned: {selection['strategy_name']}",
                            operators=selection['operators'],
                            steps=[selection.get('description', '')],
                            workflow_code=workflow_code
                        )
                    else:
                        # Fallback：使用传统parser
                        logger.warning(f"[DeepWorkflowEnvWithMeta] No workflow code, using fallback")
                        workflow_spec = self.fallback_parser.parse_qwen_output(
                            qwen_action,
                            dataset_type=self.dataset
                        )
                else:
                    # 不使用元学习，退回到传统parser
                    workflow_spec = self.fallback_parser.parse_qwen_output(
                        qwen_action,
                        dataset_type=self.dataset
                    )

                if workflow_spec is None:
                    logger.error(f"[DeepWorkflowEnvWithMeta] Env {i}: Failed to create workflow!")
                    rewards.append(0.0)
                    next_observations.append(self._construct_observation(
                        self.current_round, self.best_score, "Workflow creation failed"
                    ))
                    dones.append(False)
                    info.append({'step': self.current_round, 'error': 'creation_failed'})
                    continue

                # 2. 保存workflow代码
                round_id = f"{self.current_round}_env{i}"
                workflow_path = self._save_workflow(workflow_spec, round_id)

                logger.info(f"[DeepWorkflowEnvWithMeta] Env {i}: Workflow saved to {workflow_path}")

                # 3. 执行workflow并获取真实分数
                score, cost = self._execute_workflow(workflow_path)
                execution_time = time.time() - start_time

                logger.info(f"[DeepWorkflowEnvWithMeta] Env {i}: Score = {score:.4f}, Cost = ${cost:.4f}")

                # ===== 关键改进：记录经验到元学习系统 =====
                if self.enable_meta_learning:
                    # 记录这次执行的经验
                    self.meta_integration.record_execution(
                        problem=problem_desc,
                        dataset_type=self.dataset,
                        solution=f"operators: {workflow_spec.operators}",
                        expected_answer=None,  # 环境级别没有单个答案
                        actual_answer=score,
                        score=score,
                        execution_time=execution_time
                    )

                # 记录到历史
                self.workflow_history.append({
                    'round': self.current_round,
                    'env_id': i,
                    'operators': workflow_spec.operators,
                    'score': score,
                    'cost': cost,
                    'workflow_path': workflow_path
                })

                # 更新最佳
                if score > self.best_score:
                    self.best_score = score
                    self.best_workflow = workflow_spec
                    logger.info(f"[DeepWorkflowEnvWithMeta] 🎉 New best score: {score:.4f}")

                rewards.append(score)

                # 构造下一个观测
                obs = self._construct_observation(
                    self.current_round,
                    self.best_score,
                    self._get_history_summary()
                )
                next_observations.append(obs)

                # 判断是否结束
                done = (self.current_round >= self.max_rounds) or (score >= 0.95)
                dones.append(done)

                info.append({
                    'step': self.current_round,
                    'round': self.current_round,
                    'env_id': i,
                    'score': score,
                    'cost': cost,
                    'best_score': self.best_score,
                    'workflow_path': workflow_path,
                    'operators': workflow_spec.operators
                })

            except Exception as e:
                logger.error(f"[DeepWorkflowEnvWithMeta] Env {i} error: {e}")
                import traceback
                traceback.print_exc()

                rewards.append(0.0)
                next_observations.append(self._construct_observation(
                    self.current_round, self.best_score, f"Error: {str(e)}"
                ))
                dones.append(False)
                info.append({'step': self.current_round, 'error': str(e)})

        return next_observations, rewards, dones, info

    def _extract_problem_desc(self, qwen_action: str) -> str:
        """
        从Qwen的action中提取问题描述

        如果Qwen输出中包含问题描述，提取它；
        否则返回一个通用描述
        """
        # 简单版本：使用数据集类型作为问题描述
        # 更复杂的版本可以解析Qwen输出
        return f"Optimize workflow for {self.dataset} tasks"

    def _save_workflow(self, workflow_spec: WorkflowSpec, round_id: str) -> str:
        """保存workflow到文件"""
        round_dir = os.path.join(self.workspace_path, f"round_{round_id}")
        os.makedirs(round_dir, exist_ok=True)

        graph_path = os.path.join(round_dir, "graph.py")
        with open(graph_path, 'w') as f:
            f.write(workflow_spec.workflow_code)

        return graph_path

    def _execute_workflow(self, workflow_path: str) -> Tuple[float, float]:
        """执行workflow并返回分数"""
        try:
            # 动态加载workflow
            workflow_dir = os.path.dirname(workflow_path)
            sys.path.insert(0, workflow_dir)

            import importlib.util
            spec = importlib.util.spec_from_file_location("graph", workflow_path)
            graph_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(graph_module)

            # 创建workflow实例 - 使用泛用的字符串类型
            dataset_type = self._str_to_dataset_type(self.dataset)

            workflow = graph_module.Workflow(
                name=f"workflow_{self.current_round}",
                llm_config=self.exec_llm_config,
                dataset=dataset_type  # 现在是str而非DatasetType枚举
            )

            # 使用evaluator评估
            # 不同evaluator的接口可能不同，需要适配
            import asyncio
            import inspect

            eval_result = self.evaluator.evaluate_workflow(workflow)

            # 如果是coroutine，需要await
            if inspect.iscoroutine(eval_result):
                eval_result = asyncio.run(eval_result)

            # 适配不同evaluator的返回格式
            if isinstance(eval_result, dict):
                # AIMEEvaluator返回字典
                score = eval_result.get('pass_at_k', 0.0)
                cost = 0.0  # AIME evaluator不追踪cost
            elif isinstance(eval_result, tuple) and len(eval_result) == 2:
                # WorkflowEvaluator返回(score, cost)元组
                score, cost = eval_result
            else:
                logger.warning(f"[DeepWorkflowEnvWithMeta] Unexpected evaluator result format: {type(eval_result)}")
                score, cost = 0.0, 0.0

            sys.path.remove(workflow_dir)

            return score, cost

        except Exception as e:
            logger.error(f"[DeepWorkflowEnvWithMeta] Workflow execution error: {e}")
            import traceback
            traceback.print_exc()
            return 0.0, 0.0

    def _construct_observation(self, round_num: int, best_score: float, history: str) -> str:
        """构造观测"""
        obs = f"""Current Round: {round_num}/{self.max_rounds}
Best Score So Far: {best_score:.4f}
Dataset: {self.dataset}

History: {history}

Your task: Propose operator combinations to improve workflow performance.
Available operators: {', '.join(self.operators)}
"""
        return obs

    def _get_history_summary(self) -> str:
        """获取历史摘要"""
        if not self.workflow_history:
            return "No history yet"

        recent = self.workflow_history[-3:]
        summary = []
        for h in recent:
            summary.append(f"Round {h['round']}: operators={h['operators']}, score={h['score']:.3f}")

        return "; ".join(summary)

    def _str_to_dataset_type(self, dataset: str):
        """
        字符串转DatasetType - 泛用版本

        直接返回规范化的数据集名称，无需硬编码所有可能的数据集。
        这允许系统自动支持新数据集（如AIME）而无需修改此函数。
        """
        # 规范化：首字母大写，其余保持原样
        # 例如: "aime" -> "AIME", "humaneval" -> "HumanEval", "gsm8k" -> "GSM8K"
        dataset_name = dataset.strip()

        # 特殊处理一些常见的命名约定
        special_cases = {
            'humaneval': 'HumanEval',
            'hotpotqa': 'HotpotQA',
            'livecodebench': 'LiveCodeBench',
        }

        lower_name = dataset_name.lower()
        if lower_name in special_cases:
            return special_cases[lower_name]

        # 对于其他数据集（包括AIME），使用全大写或保持原样
        # 这支持任意新数据集而不需要修改代码
        if lower_name.isupper() or len(dataset_name) <= 5:
            return dataset_name.upper()

        return dataset_name

    def print_meta_statistics(self):
        """打印元学习统计"""
        if self.enable_meta_learning:
            self.meta_integration.print_statistics()

    def save_meta_checkpoint(self):
        """保存元学习检查点"""
        if self.enable_meta_learning:
            self.meta_integration.save_checkpoint()


# 向后兼容的工厂函数
def create_deep_workflow_env(*args, **kwargs):
    """
    创建环境的工厂函数

    自动启用元学习（如果未指定）
    """
    # 如果没有指定，默认启用元学习
    if 'enable_meta_learning' not in kwargs:
        kwargs['enable_meta_learning'] = True

    # 如果没有指定保存目录，使用默认
    if 'meta_save_dir' not in kwargs:
        if os.path.exists("/content/drive/MyDrive"):
            kwargs['meta_save_dir'] = "/content/drive/MyDrive/agentflow/meta_learning"
        else:
            kwargs['meta_save_dir'] = None  # 使用默认

    return DeepWorkflowEnvWithMeta(*args, **kwargs)
