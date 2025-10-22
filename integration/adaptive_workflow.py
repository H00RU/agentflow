"""
Adaptive Workflow - 自适应工作流
根据中间结果动态调整operator流程，结合元学习选择器

特点：
1. 动态评估结果质量
2. 根据质量自动切换策略
3. 支持多策略并行尝试
4. 自动选择最佳结果
"""

import asyncio
import sys
import os
from typing import Dict, List, Optional, Any
import time

# Add AFlow to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'AFlow'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'AFlow', 'scripts'))

from scripts import operators as operator
from scripts.async_llm import create_llm_instance
from scripts.evaluator import DatasetType

from meta_operator_selector import MetaOperatorSelector


class ResultQualityAssessor:
    """结果质量评估器"""

    def __init__(self, llm):
        self.llm = llm

    async def assess_quality(self, result: str, problem: str) -> Dict:
        """
        评估结果质量

        Returns:
            {
                'quality_score': float (0-1),
                'has_answer': bool,
                'is_code': bool,
                'is_complete': bool,
                'confidence': float
            }
        """
        assessment = {
            'quality_score': 0.5,
            'has_answer': False,
            'is_code': False,
            'is_complete': False,
            'confidence': 0.5
        }

        result_lower = result.lower()

        # 检查是否有答案
        if any(marker in result_lower for marker in ['answer:', 'result:', 'solution:', '=', 'therefore']):
            assessment['has_answer'] = True
            assessment['quality_score'] += 0.2

        # 检查是否是代码
        if 'def ' in result or 'function' in result_lower or 'return' in result_lower:
            assessment['is_code'] = True
            assessment['quality_score'] += 0.1

        # 检查完整性
        if len(result) > 50:
            assessment['is_complete'] = True
            assessment['quality_score'] += 0.1

        # 检查是否有推理过程
        if any(word in result_lower for word in ['because', 'therefore', 'so', 'thus', 'step']):
            assessment['quality_score'] += 0.1

        # 限制在0-1范围
        assessment['quality_score'] = min(1.0, assessment['quality_score'])
        assessment['confidence'] = assessment['quality_score']

        return assessment


class AdaptiveWorkflow:
    """
    自适应Workflow - 根据中间结果动态调整流程

    工作流程：
    1. 使用元学习选择器选择初始策略
    2. 执行初始策略
    3. 评估结果质量
    4. 根据质量决定：
       - 质量高：直接返回
       - 检测到代码：切换代码策略
       - 质量低：尝试ensemble
       - 中等：尝试review+revise
    5. 记录经验，持续学习
    """

    def __init__(
        self,
        name: str,
        llm_config: Dict,
        dataset: DatasetType,
        meta_selector: Optional[MetaOperatorSelector] = None,
        enable_adaptation: bool = True
    ):
        """
        初始化自适应工作流

        Args:
            name: 工作流名称
            llm_config: LLM配置
            dataset: 数据集类型
            meta_selector: 元学习选择器（可选）
            enable_adaptation: 是否启用自适应（False时使用简单策略）
        """
        self.name = name
        self.dataset = dataset
        self.enable_adaptation = enable_adaptation

        # 创建LLM实例
        self.llm = create_llm_instance(llm_config)

        # 创建所有可能用到的operators
        self.custom = operator.Custom(self.llm)
        self.custom_code_generate = operator.CustomCodeGenerate(self.llm)
        self.programmer = operator.Programmer(self.llm)
        self.sc_ensemble = operator.ScEnsemble(self.llm)
        self.test = operator.Test(self.llm)
        self.review = operator.Review(self.llm)
        self.revise = operator.Revise(self.llm)

        # 质量评估器
        self.quality_assessor = ResultQualityAssessor(self.llm)

        # 元学习选择器
        self.meta_selector = meta_selector or MetaOperatorSelector()

        print(f"[AdaptiveWorkflow] Initialized for {dataset}")
        print(f"[AdaptiveWorkflow] Adaptation: {'Enabled' if enable_adaptation else 'Disabled'}")

    async def __call__(self, problem: str, entry_point: Optional[str] = None) -> str:
        """
        执行自适应工作流

        Args:
            problem: 问题描述
            entry_point: 函数入口点（代码任务需要）

        Returns:
            最终解决方案
        """
        start_time = time.time()

        if not self.enable_adaptation:
            # 简单模式：直接使用元学习选择的策略
            result = await self._simple_execution(problem, entry_point)
        else:
            # 自适应模式：动态调整策略
            result = await self._adaptive_execution(problem, entry_point)

        execution_time = time.time() - start_time

        print(f"[AdaptiveWorkflow] Completed in {execution_time:.2f}s")

        return result

    async def _simple_execution(self, problem: str, entry_point: Optional[str]) -> str:
        """简单执行模式：使用元学习选择的策略"""
        # 使用元学习选择器
        selection = self.meta_selector.select_operators(
            problem=problem,
            dataset_type=str(self.dataset),
            use_exploration=True
        )

        operators_to_use = selection['operators']
        strategy_name = selection['strategy_name']

        print(f"[AdaptiveWorkflow] Using strategy: {strategy_name}")
        print(f"[AdaptiveWorkflow] Operators: {operators_to_use}")

        # 执行选定的策略
        if 'ScEnsemble' in operators_to_use:
            result = await self._ensemble_strategy(problem, entry_point, operators_to_use)
        elif 'Test' in operators_to_use:
            result = await self._code_test_strategy(problem, entry_point)
        else:
            result = await self._direct_strategy(problem, entry_point, operators_to_use)

        return result

    async def _adaptive_execution(self, problem: str, entry_point: Optional[str]) -> str:
        """自适应执行模式：根据中间结果动态调整"""
        # 第一步：初始尝试
        selection = self.meta_selector.select_operators(
            problem=problem,
            dataset_type=str(self.dataset),
            use_exploration=True
        )

        print(f"[AdaptiveWorkflow] Initial strategy: {selection['strategy_name']}")

        # 执行初始策略
        initial_instruction = self._get_instruction_for_dataset(str(self.dataset))
        initial_result = await self.custom(
            input=problem,
            instruction=initial_instruction
        )
        solution = initial_result.get('response', '')

        # 第二步：评估质量
        quality = await self.quality_assessor.assess_quality(solution, problem)

        print(f"[AdaptiveWorkflow] Initial quality: {quality['quality_score']:.3f}")

        # 第三步：根据质量决定下一步
        if quality['quality_score'] > 0.8:
            # 质量高，直接返回
            print("[AdaptiveWorkflow] High quality, returning directly")
            return solution

        elif quality['is_code'] and entry_point:
            # 检测到代码，切换到代码精炼
            print("[AdaptiveWorkflow] Code detected, switching to code refinement")
            return await self._code_refinement(problem, entry_point, solution)

        elif quality['quality_score'] < 0.3:
            # 质量低，尝试ensemble
            print("[AdaptiveWorkflow] Low quality, trying ensemble")
            return await self._rescue_ensemble(problem, entry_point)

        else:
            # 中等质量，尝试iterative refinement
            print("[AdaptiveWorkflow] Medium quality, trying iterative refinement")
            return await self._iterative_refinement(problem, solution)

    async def _direct_strategy(self, problem: str, entry_point: Optional[str], operators: List[str]) -> str:
        """直接策略：单次生成"""
        if 'CustomCodeGenerate' in operators and entry_point:
            result = await self.custom_code_generate(
                problem=problem,
                entry_point=entry_point,
                instruction=""
            )
        else:
            instruction = self._get_instruction_for_dataset(str(self.dataset))
            result = await self.custom(
                input=problem,
                instruction=instruction
            )

        return result.get('response', '')

    async def _ensemble_strategy(self, problem: str, entry_point: Optional[str], operators: List[str]) -> str:
        """Ensemble策略：多次采样+投票"""
        solutions = []
        num_samples = 5 if 'AIME' in str(self.dataset) or 'MATH' in str(self.dataset) else 3

        if 'CustomCodeGenerate' in operators and entry_point:
            # 代码生成
            for i in range(num_samples):
                result = await self.custom_code_generate(
                    problem=problem,
                    entry_point=entry_point,
                    instruction=""
                )
                solutions.append(result.get('response', ''))
        else:
            # 通用生成
            instruction = self._get_instruction_for_dataset(str(self.dataset))
            for i in range(num_samples):
                result = await self.custom(
                    input=problem,
                    instruction=instruction
                )
                solutions.append(result.get('response', ''))

        # Ensemble
        ensemble_result = await self.sc_ensemble(solutions=solutions, problem=problem)
        return ensemble_result.get('response', '')

    async def _code_test_strategy(self, problem: str, entry_point: str) -> str:
        """代码测试策略：生成+测试+修复"""
        result = await self.custom_code_generate(
            problem=problem,
            entry_point=entry_point,
            instruction=""
        )
        solution = result.get('response', '')

        # 测试（如果有测试用例）
        try:
            test_result = await self.test(
                problem=problem,
                solution=solution,
                entry_point=entry_point,
                test_loop=2
            )
            if test_result.get('result', False):
                return test_result.get('solution', solution)
        except Exception as e:
            print(f"[AdaptiveWorkflow] Test failed: {e}")

        return solution

    async def _code_refinement(self, problem: str, entry_point: str, initial_solution: str) -> str:
        """代码精炼：使用Programmer operator"""
        result = await self.programmer(
            problem=problem,
            analysis=f"Initial attempt: {initial_solution[:200]}"
        )
        return result.get('output', initial_solution)

    async def _rescue_ensemble(self, problem: str, entry_point: Optional[str]) -> str:
        """救援策略：质量低时使用ensemble"""
        return await self._ensemble_strategy(problem, entry_point, ['Custom', 'ScEnsemble'])

    async def _iterative_refinement(self, problem: str, initial_solution: str) -> str:
        """迭代精炼：review + revise"""
        # Review
        review_result = await self.review(
            problem=problem,
            solution=initial_solution,
            mode="xml_fill"
        )
        feedback = review_result.get('feedback', 'No specific issues found')

        # Revise
        if 'looks good' not in feedback.lower():
            revise_result = await self.revise(
                problem=problem,
                solution=initial_solution,
                feedback=feedback,
                mode="xml_fill"
            )
            return revise_result.get('solution', initial_solution)

        return initial_solution

    def _get_instruction_for_dataset(self, dataset_type: str) -> str:
        """根据数据集类型获取指令模板"""
        dataset_upper = dataset_type.upper()

        if 'AIME' in dataset_upper or 'MATH' in dataset_upper:
            return "Solve this AIME math problem step by step. Think carefully and provide your final answer as a number between 0 and 999."
        elif 'GSM8K' in dataset_upper:
            return "Solve this math problem step by step. Show your work and provide the final numerical answer."
        elif 'HUMANEVAL' in dataset_upper or 'MBPP' in dataset_upper:
            return "Write clean, efficient code to solve this problem. Include comments if needed."
        else:
            return "Analyze this problem carefully and provide a clear, well-reasoned solution."

    def record_execution_result(self, problem: str, solution: str, success: bool, score: float):
        """记录执行结果到元学习器"""
        # 提取使用的operators（简化版，实际可以更详细）
        operators_used = self._infer_operators_used(solution)

        # 找到对应的策略名称
        strategy_name = 'unknown'
        for name, strategy in self.meta_selector.OPERATOR_STRATEGIES.items():
            if set(strategy['operators']) == set(operators_used):
                strategy_name = name
                break

        self.meta_selector.record_experience(
            problem=problem,
            dataset_type=str(self.dataset),
            strategy_name=strategy_name,
            operators_used=operators_used,
            workflow_structure="adaptive",
            success=success,
            score=score,
            execution_time=0.0
        )

    def _infer_operators_used(self, solution: str) -> List[str]:
        """推断使用了哪些operators（基于解决方案特征）"""
        operators = []

        if 'def ' in solution or 'function' in solution.lower():
            operators.append('CustomCodeGenerate')
        else:
            operators.append('Custom')

        # 可以根据需要添加更多推断逻辑

        return operators


# 用于与现有系统集成的兼容性包装
class Workflow(AdaptiveWorkflow):
    """
    兼容性包装类，可以直接替换现有的Workflow类
    """
    pass


async def test_adaptive_workflow():
    """测试自适应工作流"""
    print("="*80)
    print("Testing Adaptive Workflow")
    print("="*80)

    # LLM配置（使用环境变量）
    llm_config = {
        'model': 'gpt-4o-mini',
        'api_key': os.getenv('OPENAI_API_KEY', 'your-api-key'),
        'temperature': 0.7
    }

    # 创建自适应工作流
    workflow = AdaptiveWorkflow(
        name="test_workflow",
        llm_config=llm_config,
        dataset=DatasetType.AIME,
        enable_adaptation=True
    )

    # 测试问题
    test_problems = [
        ("What is 15 + 27?", None),
        ("Write a function that returns the sum of two numbers", "add_numbers"),
    ]

    for problem, entry_point in test_problems:
        print(f"\n{'='*80}")
        print(f"Problem: {problem}")
        print(f"{'='*80}")

        try:
            result = await workflow(problem, entry_point)
            print(f"\nResult:\n{result}")
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()

    # 显示元学习统计
    print(f"\n{'='*80}")
    print("Meta Learning Statistics")
    print(f"{'='*80}")
    stats = workflow.meta_selector.get_statistics()
    print(f"Total experiences: {stats['total_experiences']}")
    if stats.get('strategy_performance'):
        print("\nStrategy performance:")
        for name, perf in stats['strategy_performance'].items():
            print(f"  {name}: {perf['attempts']} attempts, "
                  f"{perf['success_rate']:.1%} success, "
                  f"{perf['avg_score']:.3f} avg score")


if __name__ == "__main__":
    asyncio.run(test_adaptive_workflow())
