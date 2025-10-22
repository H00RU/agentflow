"""
Meta Learning Integration - 将元学习系统集成到现有训练流程
Integration script for connecting meta-learning to existing training pipeline

使用方法：
1. 在训练脚本中导入 MetaLearningIntegration
2. 替换原有的 workflow_parser 或直接使用 AdaptiveWorkflow
3. 训练过程中自动记录经验，越练越强
"""

import os
import sys
from typing import Dict, Any, List, Optional
import asyncio

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'AFlow'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'AFlow', 'scripts'))

from meta_operator_selector import MetaOperatorSelector, ProblemFeatureExtractor
from adaptive_workflow import AdaptiveWorkflow
from workflow_parser import WorkflowParser


class MetaLearningIntegration:
    """
    元学习集成类 - 桥接现有系统和元学习系统

    核心功能：
    1. 替换 WorkflowParser 使用元学习选择
    2. 包装 workflow 执行，自动记录经验
    3. 提供统计和监控接口
    """

    def __init__(
        self,
        save_dir: str = None,
        enable_meta_learning: bool = True,
        enable_adaptation: bool = True
    ):
        """
        初始化元学习集成

        Args:
            save_dir: 保存目录（默认自动检测）
            enable_meta_learning: 是否启用元学习（False时退回到原始行为）
            enable_adaptation: 是否启用自适应workflow
        """
        self.enable_meta_learning = enable_meta_learning
        self.enable_adaptation = enable_adaptation

        if enable_meta_learning:
            self.meta_selector = MetaOperatorSelector(save_dir=save_dir)
            print("[MetaLearningIntegration] Meta learning ENABLED")
        else:
            self.meta_selector = None
            print("[MetaLearningIntegration] Meta learning DISABLED (using fallback)")

        # 保留原始 parser 作为fallback
        self.fallback_parser = WorkflowParser()

        # 执行计数器
        self.execution_count = 0
        self.success_count = 0

    def create_workflow(
        self,
        name: str,
        llm_config: Dict,
        dataset: str
    ) -> Any:
        """
        创建 workflow 实例

        Args:
            name: workflow名称
            llm_config: LLM配置
            dataset: 数据集名称

        Returns:
            Workflow实例（AdaptiveWorkflow或传统Workflow）
        """
        if self.enable_meta_learning:
            # 使用自适应workflow
            from scripts.evaluator import DatasetType

            # 转换dataset字符串为DatasetType
            dataset_type = self._str_to_dataset_type(dataset)

            workflow = AdaptiveWorkflow(
                name=name,
                llm_config=llm_config,
                dataset=dataset_type,
                meta_selector=self.meta_selector,
                enable_adaptation=self.enable_adaptation
            )

            print(f"[MetaLearningIntegration] Created AdaptiveWorkflow for {dataset}")
            return workflow
        else:
            # 使用传统方式（需要手动创建）
            print(f"[MetaLearningIntegration] Using traditional workflow for {dataset}")
            # 这里返回None，让调用者使用传统方式创建
            return None

    def parse_and_select_operators(
        self,
        problem: str,
        dataset_type: str
    ) -> Dict:
        """
        解析问题并选择operators

        Args:
            problem: 问题文本
            dataset_type: 数据集类型

        Returns:
            {
                'operators': List[str],
                'strategy_name': str,
                'confidence': float,
                'workflow_code': str (可选)
            }
        """
        if self.enable_meta_learning:
            # 使用元学习选择
            selection = self.meta_selector.select_operators(
                problem=problem,
                dataset_type=dataset_type,
                use_exploration=True,
                exploration_rate=0.1
            )

            result = {
                'operators': selection['operators'],
                'strategy_name': selection['strategy_name'],
                'confidence': selection['confidence'],
                'description': selection['description']
            }

            # 可选：生成workflow代码（用于与原系统兼容）
            if hasattr(self.fallback_parser, '_generate_workflow_code'):
                result['workflow_code'] = self.fallback_parser._generate_workflow_code(
                    operators=selection['operators'],
                    steps=[selection['description']],
                    dataset_type=dataset_type
                )

            return result
        else:
            # 使用传统parser
            spec = self.fallback_parser.parse_qwen_output(problem, dataset_type)
            return {
                'operators': spec.operators if spec else [],
                'strategy_name': 'fallback',
                'confidence': 0.5,
                'workflow_code': spec.workflow_code if spec else None
            }

    def record_execution(
        self,
        problem: str,
        dataset_type: str,
        solution: str,
        expected_answer: Any,
        actual_answer: Any,
        score: float,
        execution_time: float = 0.0
    ):
        """
        记录一次执行结果

        Args:
            problem: 问题文本
            dataset_type: 数据集类型
            solution: 生成的解决方案
            expected_answer: 期望答案
            actual_answer: 实际答案
            score: 得分（0-1）
            execution_time: 执行时间
        """
        self.execution_count += 1

        if not self.enable_meta_learning:
            return

        # 判断是否成功
        success = (score > 0.5) or (str(expected_answer) == str(actual_answer))
        if success:
            self.success_count += 1

        # 推断使用的策略
        strategy_name = self._infer_strategy_from_solution(solution)
        operators_used = self._infer_operators_from_solution(solution)

        # 记录到元学习器
        self.meta_selector.record_experience(
            problem=problem,
            dataset_type=dataset_type,
            strategy_name=strategy_name,
            operators_used=operators_used,
            workflow_structure="adaptive",
            success=success,
            score=score,
            execution_time=execution_time
        )

        # 每10次执行打印一次统计
        if self.execution_count % 10 == 0:
            self.print_statistics()

    def _infer_strategy_from_solution(self, solution: str) -> str:
        """从解决方案推断使用的策略"""
        solution_lower = solution.lower()

        # 简单启发式推断
        if 'def ' in solution or 'function' in solution_lower:
            return 'code_simple'
        elif any(word in solution_lower for word in ['step 1', 'step 2', 'therefore', 'because']):
            return 'math_ensemble'
        else:
            return 'math_direct'

    def _infer_operators_from_solution(self, solution: str) -> List[str]:
        """从解决方案推断使用的operators"""
        operators = []

        if 'def ' in solution or 'function' in solution.lower():
            operators.append('CustomCodeGenerate')
        else:
            operators.append('Custom')

        return operators

    def _str_to_dataset_type(self, dataset: str):
        """将字符串转换为DatasetType枚举"""
        from scripts.evaluator import DatasetType

        dataset_upper = dataset.upper()

        if 'HUMANEVAL' in dataset_upper:
            return DatasetType.HumanEval
        elif 'MBPP' in dataset_upper:
            return DatasetType.MBPP
        elif 'GSM8K' in dataset_upper:
            return DatasetType.GSM8K
        elif 'MATH' in dataset_upper:
            return DatasetType.MATH
        elif 'AIME' in dataset_upper:
            return DatasetType.AIME
        elif 'HOTPOTQA' in dataset_upper:
            return DatasetType.HotpotQA
        elif 'DROP' in dataset_upper:
            return DatasetType.DROP
        else:
            # 默认
            return DatasetType.HumanEval

    def print_statistics(self):
        """打印统计信息"""
        if not self.enable_meta_learning:
            return

        print("\n" + "="*80)
        print("Meta Learning Statistics")
        print("="*80)

        print(f"Total executions: {self.execution_count}")
        print(f"Success count: {self.success_count}")
        if self.execution_count > 0:
            print(f"Success rate: {self.success_count / self.execution_count:.1%}")

        # 获取元学习器统计
        stats = self.meta_selector.get_statistics()
        print(f"\nMeta selector experiences: {stats['total_experiences']}")

        if stats.get('best_strategy'):
            print(f"Best strategy: {stats['best_strategy']['name']} "
                  f"(avg score: {stats['best_strategy']['avg_score']:.3f})")

        if stats.get('strategy_performance'):
            print("\nStrategy performance:")
            for name, perf in sorted(
                stats['strategy_performance'].items(),
                key=lambda x: x[1]['avg_score'],
                reverse=True
            )[:5]:  # 只显示top 5
                print(f"  {name:20s}: {perf['attempts']:3d} attempts, "
                      f"{perf['success_rate']:5.1%} success, "
                      f"{perf['avg_score']:.3f} avg")

        print("="*80 + "\n")

    def save_checkpoint(self):
        """手动保存检查点"""
        if self.enable_meta_learning:
            self.meta_selector._save_state()
            print("[MetaLearningIntegration] Checkpoint saved")

    def get_best_strategies(self, top_k: int = 3) -> List[Dict]:
        """
        获取表现最好的策略

        Returns:
            List of {name, avg_score, attempts, success_rate}
        """
        if not self.enable_meta_learning:
            return []

        stats = self.meta_selector.get_statistics()
        if not stats.get('strategy_performance'):
            return []

        strategies = [
            {
                'name': name,
                'avg_score': perf['avg_score'],
                'attempts': perf['attempts'],
                'success_rate': perf['success_rate']
            }
            for name, perf in stats['strategy_performance'].items()
        ]

        # 按平均得分排序
        strategies.sort(key=lambda x: x['avg_score'], reverse=True)

        return strategies[:top_k]


def integrate_with_existing_training():
    """
    与现有训练系统集成的示例

    展示如何在现有训练代码中使用元学习
    """
    print("="*80)
    print("Meta Learning Integration Example")
    print("="*80)

    # 1. 创建集成实例
    integration = MetaLearningIntegration(
        enable_meta_learning=True,
        enable_adaptation=True
    )

    # 2. 在训练循环中使用
    print("\n示例：替换现有的 workflow 创建")
    print("-"*80)

    llm_config = {
        'model': 'gpt-4o-mini',
        'api_key': os.getenv('OPENAI_API_KEY', 'your-key'),
        'temperature': 0.7
    }

    # 创建自适应workflow
    workflow = integration.create_workflow(
        name="aime_workflow",
        llm_config=llm_config,
        dataset="AIME"
    )

    print(f"Created workflow: {type(workflow).__name__}")

    # 3. 执行问题（模拟）
    print("\n示例：记录执行结果")
    print("-"*80)

    test_problem = "What is 2 + 2?"
    expected = "4"

    # 模拟执行
    for i in range(5):
        # 这里应该是实际的workflow执行
        actual = "4" if i % 2 == 0 else "3"  # 模拟有时对有时错
        score = 1.0 if actual == expected else 0.0

        integration.record_execution(
            problem=test_problem,
            dataset_type="AIME",
            solution=f"The answer is {actual}",
            expected_answer=expected,
            actual_answer=actual,
            score=score,
            execution_time=1.5
        )

    # 4. 查看统计
    print("\n最终统计：")
    integration.print_statistics()

    # 5. 获取最佳策略
    print("\nTop 3 strategies:")
    best = integration.get_best_strategies(top_k=3)
    for i, strategy in enumerate(best, 1):
        print(f"{i}. {strategy['name']}: score={strategy['avg_score']:.3f}, "
              f"attempts={strategy['attempts']}")


if __name__ == "__main__":
    integrate_with_existing_training()
