"""
Workflow Evaluator - 真实执行workflow并测试HumanEval
Real workflow execution and HumanEval testing
"""

import sys
import os
import asyncio
import time
from typing import Dict, Any, Optional
import json

# Add AFlow to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'AFlow'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'AFlow', 'scripts'))

from scripts.logs import logger
from scripts.evaluator import DatasetType


class WorkflowEvaluator:
    """
    Workflow评估器 - 执行真实的HumanEval测试

    功能：
    1. 加载workflow实例
    2. 在HumanEval数据集上运行workflow
    3. 计算pass@k分数
    4. 返回真实的性能指标
    """

    def __init__(
        self,
        dataset: str = "HumanEval",
        sample_size: int = 3,
        timeout_per_problem: int = 30,
        train_test_split: float = 0.8
    ):
        """
        初始化evaluator

        Args:
            dataset: 数据集名称
            sample_size: 测试样本数量
            timeout_per_problem: 每个问题的超时时间(秒)
            train_test_split: Ratio of training data (default: 0.8 = 80% train, 20% test)
        """
        self.dataset = dataset
        self.sample_size = sample_size
        self.timeout_per_problem = timeout_per_problem
        self.train_test_split = train_test_split

        # 加载HumanEval数据集
        self.problems = self._load_humaneval_problems()

        logger.info(f"[WorkflowEvaluator] Initialized")
        logger.info(f"[WorkflowEvaluator] Dataset: {dataset}")
        logger.info(f"[WorkflowEvaluator] Sample size: {sample_size}")
        logger.info(f"[WorkflowEvaluator] Loaded {len(self.problems)} problems")

    def _load_humaneval_problems(self) -> Dict:
        """加载HumanEval问题"""
        try:
            # HumanEval数据文件路径
            aflow_path = os.path.join(os.path.dirname(__file__), '..', 'AFlow')
            humaneval_path = os.path.join(aflow_path, 'datasets', 'HumanEval.jsonl')

            if not os.path.exists(humaneval_path):
                # 尝试备用路径
                humaneval_path = os.path.join(aflow_path, 'data', 'HumanEval.jsonl')

            if not os.path.exists(humaneval_path):
                logger.warning(f"[WorkflowEvaluator] HumanEval file not found, using dummy data")
                return self._create_dummy_problems()

            # 加载problems
            problems = {}
            with open(humaneval_path, 'r') as f:
                for line in f:
                    problem = json.loads(line)
                    problems[problem['task_id']] = problem

            return problems

        except Exception as e:
            logger.error(f"[WorkflowEvaluator] Error loading HumanEval: {e}")
            return self._create_dummy_problems()

    def _create_dummy_problems(self) -> Dict:
        """创建一些dummy problems用于测试"""
        return {
            'HumanEval/0': {
                'task_id': 'HumanEval/0',
                'prompt': 'def has_close_elements(numbers: List[float], threshold: float) -> bool:\n    """ Check if in given list of numbers, are any two numbers closer to each other than\n    given threshold.\n    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)\n    False\n    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)\n    True\n    """\n',
                'entry_point': 'has_close_elements',
                'canonical_solution': '    for idx, elem in enumerate(numbers):\n        for idx2, elem2 in enumerate(numbers):\n            if idx != idx2:\n                distance = abs(elem - elem2)\n                if distance < threshold:\n                    return True\n\n    return False\n',
                'test': 'def check(candidate):\n    assert candidate([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.3) == True\n    assert candidate([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.05) == False\n    assert candidate([1.0, 2.0, 5.9, 4.0, 5.0], 0.95) == True\n    assert candidate([1.0, 2.0, 5.9, 4.0, 5.0], 0.8) == False\n    assert candidate([1.0, 2.0, 3.0, 4.0, 5.0, 2.0], 0.1) == True\n\ncheck(has_close_elements)'
            }
        }

    async def evaluate_workflow(
        self,
        workflow,
        num_problems: Optional[int] = None,
        use_test_set: bool = False,
        random_sample: bool = True
    ) -> Dict[str, Any]:
        """
        评估workflow性能

        Args:
            workflow: Workflow实例
            num_problems: 测试的问题数量（None=使用sample_size）
            use_test_set: 是否使用测试集（True=测试集，False=训练集）
            random_sample: 是否随机采样（True=随机，False=固定前N个）

        Returns:
            评估结果字典，包含：
            - pass_at_k: pass@k分数
            - num_passed: 通过的问题数
            - num_total: 总问题数
            - avg_time: 平均时间
            - details: 详细结果
        """
        start_time = time.time()

        # 选择测试问题
        if num_problems is None:
            num_problems = min(self.sample_size, len(self.problems))

        # 划分训练集和测试集 (configurable split)
        all_problem_ids = list(self.problems.keys())
        train_size = int(len(all_problem_ids) * self.train_test_split)
        train_ids = all_problem_ids[:train_size]
        test_ids = all_problem_ids[train_size:]

        # 选择数据集
        if use_test_set:
            available_ids = test_ids
            logger.info(f"[WorkflowEvaluator] 📊 Using TEST set ({len(test_ids)} problems available)")
        else:
            available_ids = train_ids
            logger.info(f"[WorkflowEvaluator] 📚 Using TRAIN set ({len(train_ids)} problems available)")

        # 采样问题
        if random_sample and num_problems < len(available_ids):
            import random
            problem_ids = random.sample(available_ids, min(num_problems, len(available_ids)))
            logger.info(f"[WorkflowEvaluator] 🎲 Randomly sampled {len(problem_ids)} problems")
        else:
            problem_ids = available_ids[:min(num_problems, len(available_ids))]
            logger.info(f"[WorkflowEvaluator] 📋 Using first {len(problem_ids)} problems")

        logger.info(f"[WorkflowEvaluator] Testing workflow on {len(problem_ids)} problems...")

        results = []
        num_passed = 0

        for i, task_id in enumerate(problem_ids):
            problem = self.problems[task_id]

            logger.info(f"[WorkflowEvaluator] [{i+1}/{num_problems}] Testing {task_id}...")

            try:
                # 运行workflow
                problem_start = time.time()

                solution, cost = await asyncio.wait_for(
                    workflow(
                        problem=problem['prompt'],
                        entry_point=problem['entry_point']
                    ),
                    timeout=self.timeout_per_problem
                )

                problem_time = time.time() - problem_start

                # 测试solution
                passed = self._test_solution(
                    solution=solution,
                    test_code=problem.get('test', ''),
                    entry_point=problem['entry_point']
                )

                if passed:
                    num_passed += 1
                    logger.info(f"[WorkflowEvaluator] {task_id}: ✅ PASSED")
                else:
                    logger.info(f"[WorkflowEvaluator] {task_id}: ❌ FAILED")

                results.append({
                    'task_id': task_id,
                    'passed': passed,
                    'time': problem_time,
                    'cost': cost,
                    'solution_length': len(solution) if solution else 0
                })

            except asyncio.TimeoutError:
                logger.error(f"[WorkflowEvaluator] {task_id}: ⏱️ TIMEOUT")
                results.append({
                    'task_id': task_id,
                    'passed': False,
                    'time': self.timeout_per_problem,
                    'error': 'timeout'
                })

            except Exception as e:
                logger.error(f"[WorkflowEvaluator] {task_id}: ❌ ERROR: {e}")
                results.append({
                    'task_id': task_id,
                    'passed': False,
                    'time': 0,
                    'error': str(e)
                })

        total_time = time.time() - start_time

        # 计算pass@k
        pass_at_k = num_passed / num_problems if num_problems > 0 else 0.0
        avg_time = total_time / num_problems if num_problems > 0 else 0.0

        evaluation_result = {
            'pass_at_k': pass_at_k,
            'pass_at_1': pass_at_k,  # 同pass@k
            'num_passed': num_passed,
            'num_total': num_problems,
            'avg_time': avg_time,
            'total_time': total_time,
            'details': results
        }

        logger.info(f"[WorkflowEvaluator] ===== EVALUATION COMPLETE =====")
        logger.info(f"[WorkflowEvaluator] Pass@{num_problems}: {pass_at_k:.4f} ({num_passed}/{num_problems})")
        logger.info(f"[WorkflowEvaluator] Avg time: {avg_time:.2f}s")
        logger.info(f"[WorkflowEvaluator] Total time: {total_time:.2f}s")

        return evaluation_result

    def _test_solution(
        self,
        solution: str,
        test_code: str,
        entry_point: str
    ) -> bool:
        """
        测试solution是否通过

        Args:
            solution: 生成的solution代码
            test_code: 测试代码
            entry_point: 函数入口点

        Returns:
            是否通过测试
        """
        if not solution or not test_code:
            return False

        try:
            # 创建测试环境
            test_env = {}

            # 执行solution代码
            exec(solution, test_env)

            # 执行测试代码
            exec(test_code, test_env)

            # 如果没有异常，说明通过
            return True

        except AssertionError as e:
            # 测试失败
            return False

        except Exception as e:
            # 运行错误
            logger.debug(f"[WorkflowEvaluator] Solution execution error: {e}")
            return False

    def quick_test(self, workflow, num_problems: int = 1) -> float:
        """
        快速测试workflow（用于RL训练）

        Args:
            workflow: Workflow实例
            num_problems: 测试问题数（默认1，更快）

        Returns:
            pass@k分数
        """
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            result = loop.run_until_complete(
                self.evaluate_workflow(workflow, num_problems=num_problems)
            )
            return result['pass_at_k']

        finally:
            loop.close()


# 单例evaluator（避免重复加载）
_global_evaluator = None


def get_evaluator(dataset: str = "HumanEval", sample_size: int = 3):
    """获取全局evaluator单例"""
    global _global_evaluator
    if _global_evaluator is None:
        _global_evaluator = WorkflowEvaluator(
            dataset=dataset,
            sample_size=sample_size
        )
    return _global_evaluator


if __name__ == "__main__":
    # 测试evaluator
    print("Testing WorkflowEvaluator...")

    evaluator = WorkflowEvaluator(sample_size=1)

    # 创建一个简单的测试workflow
    class DummyWorkflow:
        async def __call__(self, problem, entry_point):
            # 返回一个简单的solution
            solution = f"""
def {entry_point}(numbers, threshold):
    for i in range(len(numbers)):
        for j in range(i+1, len(numbers)):
            if abs(numbers[i] - numbers[j]) < threshold:
                return True
    return False
"""
            return solution, 0.01

    workflow = DummyWorkflow()

    # 测试
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    result = loop.run_until_complete(
        evaluator.evaluate_workflow(workflow, num_problems=1)
    )

    print(f"\nTest result:")
    print(f"Pass@K: {result['pass_at_k']:.4f}")
    print(f"Passed: {result['num_passed']}/{result['num_total']}")

    loop.close()
