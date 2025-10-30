"""
评估系统适配器 - 让AFlow Optimizer使用WorkflowEvaluator

将WorkflowEvaluator适配为AFlow期望的evaluation_utils接口
这样可以保持单一评估系统，同时兼容AFlow的optimizer
"""

import asyncio
from typing import Dict, Any
from scripts.logs import logger


class EvaluationUtilsAdapter:
    """
    适配器模式：让AFlow的optimizer能够使用WorkflowEvaluator

    替换optimizer的evaluation_utils，所有评估调用转发到WorkflowEvaluator
    """

    def __init__(self, workflow_evaluator, root_path: str = None):
        """
        Args:
            workflow_evaluator: WorkflowEvaluator实例（你设计的评估器）
            root_path: AFlow兼容性参数（不使用）
        """
        self.workflow_evaluator = workflow_evaluator
        self.root_path = root_path

    async def evaluate_initial_round(self, optimizer, graph_path, directory, validation_n, data):
        """
        初始轮次评估 - 适配AFlow接口

        Args:
            optimizer: RLEnhancedOptimizer实例
            graph_path: workflow文件路径
            directory: 工作目录
            validation_n: 验证轮数（我们的evaluator不需要这个）
            data: 数据累积列表

        Returns:
            data: 更新后的数据列表（保持AFlow兼容）
        """
        logger.info("[EvaluationAdapter] ===== Initial Round Evaluation =====")
        logger.info(f"[EvaluationAdapter] Using WorkflowEvaluator (mini-batch + train/test support)")
        logger.info(f"[EvaluationAdapter] Graph path: {graph_path}")

        # 评估当前workflow
        score = await self._evaluate_workflow_from_path(graph_path, directory)

        # 构造AFlow期望的数据格式
        result_data = {
            "round": optimizer.round,
            "score": score,
            "avg_cost": 0.0,  # WorkflowEvaluator不跟踪cost
            "total_cost": 0.0
        }
        data.append(result_data)

        logger.info(f"[EvaluationAdapter] Initial score: {score:.4f}")
        return data

    async def evaluate_graph(self, optimizer, directory, validation_n, data, initial=False):
        """
        标准图评估 - 适配AFlow接口

        Args:
            optimizer: RLEnhancedOptimizer实例
            directory: 工作目录
            validation_n: 验证轮数
            data: 数据累积列表
            initial: 是否是初始评估

        Returns:
            avg_score: 平均分数
        """
        logger.info(f"[EvaluationAdapter] ===== Graph Evaluation (Round {optimizer.round}) =====")
        logger.info(f"[EvaluationAdapter] Using WorkflowEvaluator")

        # 从optimizer的workspace获取graph路径
        graph_path = f"{optimizer.root_path}/{optimizer.dataset}/workflows/round_{optimizer.round}"

        # 评估workflow
        score = await self._evaluate_workflow_from_path(graph_path, directory)

        # 如果需要累积数据
        if data is not None:
            result_data = {
                "round": optimizer.round,
                "score": score,
                "avg_cost": 0.0,
                "total_cost": 0.0
            }
            data.append(result_data)

        logger.info(f"[EvaluationAdapter] Round {optimizer.round} score: {score:.4f}")
        return score

    async def evaluate_graph_test(self, optimizer, directory, is_test=True):
        """
        测试集评估 - 适配AFlow接口

        Args:
            optimizer: RLEnhancedOptimizer实例
            directory: 工作目录
            is_test: 是否使用测试集

        Returns:
            (score, avg_cost, total_cost): 三元组
        """
        logger.info(f"[EvaluationAdapter] ===== Test Set Evaluation =====")
        logger.info(f"[EvaluationAdapter] Using WorkflowEvaluator with test set")

        # 从optimizer的workspace获取graph路径
        graph_path = f"{optimizer.root_path}/{optimizer.dataset}/workflows/round_{optimizer.round}"

        # 评估workflow，使用测试集
        score = await self._evaluate_workflow_from_path(
            graph_path,
            directory,
            use_test_set=is_test
        )

        logger.info(f"[EvaluationAdapter] Test score: {score:.4f}")

        # 返回AFlow期望的三元组
        return (score, 0.0, 0.0)  # (score, avg_cost, total_cost)

    async def _evaluate_workflow_from_path(
        self,
        graph_path: str,
        directory: str,
        use_test_set: bool = False
    ) -> float:
        """
        从文件路径加载并评估workflow

        Args:
            graph_path: workflow目录路径
            directory: 工作目录
            use_test_set: 是否使用测试集

        Returns:
            score: 评估分数
        """
        import importlib.util
        import os

        # 加载workflow模块
        graph_file = os.path.join(graph_path, "graph.py")

        if not os.path.exists(graph_file):
            logger.warning(f"[EvaluationAdapter] Graph file not found: {graph_file}")
            return 0.0

        try:
            # 动态加载workflow模块
            spec = importlib.util.spec_from_file_location("workflow_module", graph_file)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            WorkflowClass = module.Workflow

            # 创建workflow实例
            workflow = WorkflowClass(
                name=f"EvaluationWorkflow_Round_{os.path.basename(graph_path)}",
                llm_config=self.workflow_evaluator.llm_config,  # 使用evaluator的配置
                dataset=self.workflow_evaluator.dataset
            )

            # 使用WorkflowEvaluator评估
            # 关键：这里会使用mini-batch和train/test划分！
            result = await self.workflow_evaluator.evaluate_workflow(
                workflow,
                num_problems=None,  # 使用evaluator的默认配置
                use_test_set=use_test_set,
                random_sample=True  # mini-batch模式下随机采样
            )

            score = result.get('pass_at_k', 0.0)
            logger.info(f"[EvaluationAdapter] Workflow evaluated: {score:.4f}")
            logger.info(f"[EvaluationAdapter]   - Passed: {result.get('num_passed', 0)}/{result.get('num_total', 0)}")

            return score

        except Exception as e:
            logger.error(f"[EvaluationAdapter] Error evaluating workflow: {e}")
            import traceback
            traceback.print_exc()
            return 0.0
