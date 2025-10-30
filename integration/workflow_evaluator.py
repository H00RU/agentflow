"""
Workflow Evaluator - 执行workflow并评估结果
Real workflow execution and evaluation
支持多种数据集：HumanEval, AIME, 等
"""

import sys
import os
import asyncio
import time
import json
import re
from typing import Dict, Any, Optional, Tuple, List
from abc import ABC, abstractmethod
from pathlib import Path
import yaml

# Add AFlow to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'AFlow'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'AFlow', 'scripts'))

try:
    from scripts.logs import logger
except:
    import logging
    logger = logging.getLogger(__name__)


# ===========================
# 答案验证器基类和具体实现
# ===========================

class AnswerValidator(ABC):
    """答案验证器基类"""

    @abstractmethod
    def validate(self, solution: str, problem: Dict, timeout: int = 10) -> Tuple[bool, str]:
        """
        验证解答是否正确

        Args:
            solution: LLM生成的解答代码/文本
            problem: 包含问题和答案的字典
            timeout: 超时时间(秒)

        Returns:
            (是否通过, 详细说明)
        """
        pass


class CodeExecutionValidator(AnswerValidator):
    """
    代码执行验证器 - 用于HumanEval
    执行Python代码并检查测试用例
    """

    def validate(self, solution: str, problem: Dict, timeout: int = 10) -> Tuple[bool, str]:
        """执行代码并检查测试"""
        if not solution or not problem.get('answer'):
            return False, "Missing solution or test code"

        test_code = problem['answer']
        entry_point = problem.get('entry_point', '')

        try:
            # 创建执行环境
            test_env = {}

            # 执行solution代码
            exec(solution, test_env)

            # 执行测试代码
            exec(test_code, test_env)

            # 如果没有异常，说明通过
            return True, "Code execution passed all tests"

        except AssertionError as e:
            return False, f"Assertion failed: {str(e)}"
        except Exception as e:
            return False, f"Code execution error: {str(e)}"


class NumericComparisonValidator(AnswerValidator):
    """
    数值比较验证器 - 用于AIME等数学问题
    从LLM输出中提取数值答案并与标准答案比较
    """

    def __init__(self, answer_pattern: str = r"<answer>(.*?)</answer>"):
        """
        初始化

        Args:
            answer_pattern: 从输出中提取答案的正则表达式
        """
        self.answer_pattern = answer_pattern

    def validate(self, solution: str, problem: Dict, timeout: int = 10) -> Tuple[bool, str]:
        """提取答案并与标准答案比较"""
        if not solution:
            return False, "No solution provided"

        correct_answer = problem.get('answer')
        if correct_answer is None:
            return False, "No correct answer provided"

        try:
            # 提取答案
            extracted_answer = self._extract_answer(solution)

            if extracted_answer is None:
                return False, f"Could not extract answer from solution. Expected: {correct_answer}"

            # 比较答案
            is_correct, comparison = self._compare_answers(extracted_answer, correct_answer)

            if is_correct:
                return True, f"Correct answer: {extracted_answer} == {correct_answer}"
            else:
                return False, f"Wrong answer: {extracted_answer} != {correct_answer}. {comparison}"

        except Exception as e:
            return False, f"Answer validation error: {str(e)}"

    def _extract_answer(self, solution: str) -> Optional[str]:
        """
        从LLM输出中提取答案

        尝试多种方式：
        1. <answer>...</answer> 标签
        2. 最后一行的数字
        3. 整个输出
        """
        # 方式1：<answer>标签
        matches = re.findall(self.answer_pattern, solution, re.DOTALL)
        if matches:
            return matches[-1].strip()

        # 方式2：最后一行
        lines = solution.strip().split('\n')
        for line in reversed(lines):
            line = line.strip()
            if line and any(c.isdigit() for c in line):
                # 尝试提取数字
                nums = re.findall(r'-?\d+(?:\.\d+)?', line)
                if nums:
                    return nums[-1]

        # 方式3：整个输出
        nums = re.findall(r'-?\d+(?:\.\d+)?', solution)
        if nums:
            return nums[-1]

        return None

    def _compare_answers(self, extracted: str, correct: Any) -> Tuple[bool, str]:
        """
        比较提取的答案和正确答案

        支持：
        - 整数比较
        - 浮点数比较
        - 分数比较（1/2 == 0.5）
        - 字符串比较
        """
        try:
            # 尝试转换为数字
            extracted_num = self._parse_number(extracted)
            correct_num = self._parse_number(str(correct))

            if extracted_num is not None and correct_num is not None:
                # 数值比较
                if isinstance(extracted_num, float) or isinstance(correct_num, float):
                    # 浮点数比较（允许小误差）
                    tolerance = 1e-6
                    is_close = abs(extracted_num - correct_num) < tolerance
                    return is_close, f"Float comparison: {extracted_num} vs {correct_num}"
                else:
                    # 整数比较
                    is_equal = int(extracted_num) == int(correct_num)
                    return is_equal, f"Integer comparison: {extracted_num} vs {correct_num}"

            # 字符串比较
            is_equal = str(extracted).strip().lower() == str(correct).strip().lower()
            return is_equal, f"String comparison: '{extracted}' vs '{correct}'"

        except Exception as e:
            return False, f"Comparison error: {str(e)}"

    def _parse_number(self, value: str) -> Optional[float]:
        """将字符串解析为数字（支持分数、浮点数等）"""
        try:
            # 移除空格
            value = str(value).strip()

            # 分数处理：a/b
            if '/' in value:
                parts = value.split('/')
                if len(parts) == 2:
                    numerator = float(parts[0])
                    denominator = float(parts[1])
                    return numerator / denominator

            # 直接转换为浮点数
            return float(value)
        except:
            return None


class LLMJudgeValidator(AnswerValidator):
    """
    LLM评判验证器 - 用于需要语义理解的任务
    使用LLM来评判答案是否正确
    """

    def __init__(self, llm_engine=None):
        """
        初始化

        Args:
            llm_engine: LLM引擎（可选，默认使用GPT-4o）
        """
        self.llm_engine = llm_engine

    def validate(self, solution: str, problem: Dict, timeout: int = 10) -> Tuple[bool, str]:
        """使用LLM判断答案"""
        if not self.llm_engine:
            # 如果没有LLM引擎，回退到数值比较或字符串匹配
            return self._fallback_validate(solution, problem)

        # TODO: 实现LLM Judge逻辑
        # 这里可以调用GPT-4O进行答案验证
        return False, "LLM Judge not yet implemented"

    def _fallback_validate(self, solution: str, problem: Dict) -> Tuple[bool, str]:
        """回退验证方法"""
        # 简单的字符串比较
        answer = problem.get('answer', '')
        if str(answer).strip().lower() in solution.lower():
            return True, "Answer found in solution"
        return False, "Answer not found in solution"


# ===========================
# 主评估器类
# ===========================

class WorkflowEvaluator:
    """
    Workflow评估器 - 支持多数据集的统一评估框架

    功能：
    1. 加载多种类型的数据集（HumanEval, AIME, 等）
    2. 在数据集上运行workflow
    3. 使用数据集特定的验证器评估结果
    4. 计算pass@k分数
    5. 返回详细的性能指标
    """

    def __init__(
        self,
        dataset: str = "HumanEval",
        sample_size: int = 3,
        timeout_per_problem: int = 30,
        train_test_split: float = 0.8,
        llm_config: dict = None,
        config_file: str = None
    ):
        """
        初始化evaluator

        Args:
            dataset: 数据集名称 (HumanEval, AIME, AIME24等)
            sample_size: 测试样本数量
            timeout_per_problem: 每个问题的超时时间(秒)
            train_test_split: 训练集比例 (默认0.8 = 80% train, 20% test)
            llm_config: LLM配置（用于创建workflow实例）
            config_file: 数据集配置文件路径
        """
        self.dataset = dataset
        self.sample_size = sample_size
        self.timeout_per_problem = timeout_per_problem
        self.train_test_split = train_test_split
        self.llm_config = llm_config or {}

        # 加载数据集配置
        self.config = self._load_dataset_config(config_file, dataset)

        # 根据配置加载数据集
        self.problems = self._load_dataset()

        # 初始化答案验证器
        self.validator = self._create_validator()

        logger.info(f"[WorkflowEvaluator] Initialized")
        logger.info(f"[WorkflowEvaluator] Dataset: {dataset}")
        logger.info(f"[WorkflowEvaluator] Sample size: {sample_size}")
        logger.info(f"[WorkflowEvaluator] Train/Test split: {int(train_test_split*100)}%/{int((1-train_test_split)*100)}%")
        logger.info(f"[WorkflowEvaluator] Evaluation method: {self.config.get('evaluation_method')}")
        logger.info(f"[WorkflowEvaluator] Loaded {len(self.problems)} problems")

    def _load_dataset_config(self, config_file: str = None, dataset: str = None) -> Dict:
        """加载数据集配置"""
        # 确定配置文件路径
        if config_file is None:
            config_file = os.path.join(os.path.dirname(__file__), 'dataset_configs.yaml')

        # 加载配置
        try:
            with open(config_file, 'r') as f:
                all_configs = yaml.safe_load(f)
        except Exception as e:
            logger.warning(f"[WorkflowEvaluator] Could not load config file {config_file}: {e}")
            all_configs = {'datasets': {}}

        # 获取该数据集的配置
        dataset_config = all_configs.get('datasets', {}).get(dataset, {})

        if not dataset_config:
            logger.warning(f"[WorkflowEvaluator] No config found for dataset '{dataset}', using defaults")
            return {
                'evaluation_method': 'code_execution',
                'sample_size': 10,
                'train_test_split': 0.8
            }

        return dataset_config

    def _load_dataset(self) -> Dict:
        """根据配置加载数据集"""
        data_path = self.config.get('data_path', '')

        # 数据路径是相对于agentflow目录的
        if not os.path.isabs(data_path):
            data_path = os.path.join(
                os.path.dirname(__file__),
                '..',
                data_path
            )

        data_path = os.path.abspath(data_path)

        logger.info(f"[WorkflowEvaluator] Loading dataset from: {data_path}")

        if not os.path.exists(data_path):
            logger.warning(f"[WorkflowEvaluator] Data file not found: {data_path}")
            return self._create_dummy_problems()

        try:
            if data_path.endswith('.json'):
                return self._load_json_dataset(data_path)
            elif data_path.endswith('.jsonl'):
                return self._load_jsonl_dataset(data_path)
            else:
                logger.warning(f"[WorkflowEvaluator] Unknown file format: {data_path}")
                return self._create_dummy_problems()

        except Exception as e:
            logger.error(f"[WorkflowEvaluator] Error loading dataset: {e}")
            return self._create_dummy_problems()

    def _load_json_dataset(self, data_path: str) -> Dict:
        """加载JSON格式的数据集"""
        with open(data_path, 'r') as f:
            items = json.load(f)

        problems = {}

        # 处理数组格式（如AIME24）
        if isinstance(items, list):
            for item in items:
                # 使用idx或pid作为key
                problem_id = f"{self.dataset}/{item.get('idx', item.get('pid', len(problems)))}"
                problems[problem_id] = self._standardize_problem(item)

        # 处理字典格式
        elif isinstance(items, dict):
            for key, item in items.items():
                problem_id = f"{self.dataset}/{key}"
                problems[problem_id] = self._standardize_problem(item)

        logger.info(f"[WorkflowEvaluator] ✅ Loaded {len(problems)} problems")
        return problems

    def _load_jsonl_dataset(self, data_path: str) -> Dict:
        """加载JSONL格式的数据集"""
        problems = {}

        with open(data_path, 'r') as f:
            for line in f:
                item = json.loads(line)
                problem_id = item.get('task_id', f"{self.dataset}/{len(problems)}")
                problems[problem_id] = self._standardize_problem(item)

        logger.info(f"[WorkflowEvaluator] ✅ Loaded {len(problems)} problems")
        return problems

    def _standardize_problem(self, item: Dict) -> Dict:
        """
        将原始问题转换为标准格式

        标准格式：
        {
            'question': 问题文本（给LLM的输入）,
            'answer': 标准答案（用于验证）,
            'entry_point': 函数入口点（可选）,
            'original': 原始数据（保留备用）
        }
        """
        fields = self.config.get('problem_fields', {})

        # 获取问题字段名
        question_field = fields.get('question', 'question')
        answer_field = fields.get('answer', 'answer')
        entry_point_field = fields.get('entry_point', 'entry_point')

        standardized = {
            'question': item.get(question_field, item.get('question', '')),
            'answer': item.get(answer_field, item.get('answer')),
            'entry_point': item.get(entry_point_field, item.get('entry_point', 'solve')),
            'original': item  # 保留原始数据
        }

        return standardized

    def _create_dummy_problems(self) -> Dict:
        """创建dummy问题用于测试"""
        evaluation_method = self.config.get('evaluation_method', 'code_execution')

        if evaluation_method == 'numeric_comparison':
            # AIME风格的dummy问题
            return {
                f'{self.dataset}/0': {
                    'question': 'Find the sum of all positive integers less than 1000 that are divisible by 3 or 5.',
                    'answer': 233168,
                    'entry_point': 'solve',
                    'original': {}
                }
            }
        else:
            # HumanEval风格的dummy问题
            return {
                f'{self.dataset}/0': {
                    'question': 'def has_close_elements(numbers: List[float], threshold: float) -> bool:\n    """ Check if in given list of numbers..."""',
                    'answer': 'def check(candidate):\n    assert candidate([1.0, 2.0, 3.9], 0.3) == True',
                    'entry_point': 'has_close_elements',
                    'original': {}
                }
            }

    def _create_validator(self) -> AnswerValidator:
        """根据评估方法创建答案验证器"""
        evaluation_method = self.config.get('evaluation_method', 'code_execution')

        if evaluation_method == 'code_execution':
            return CodeExecutionValidator()

        elif evaluation_method == 'numeric_comparison':
            pattern = self.config.get('answer_extraction_pattern', r'<answer>(.*?)</answer>')
            return NumericComparisonValidator(answer_pattern=pattern)

        elif evaluation_method == 'llm_judge':
            return LLMJudgeValidator(llm_engine=None)

        else:
            logger.warning(f"[WorkflowEvaluator] Unknown evaluation method: {evaluation_method}")
            return CodeExecutionValidator()  # 默认使用代码执行

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

        # 划分训练集和测试集
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

            logger.info(f"[WorkflowEvaluator] [{i+1}/{len(problem_ids)}] Testing {task_id}...")

            try:
                # 运行workflow
                problem_start = time.time()

                solution, cost = await asyncio.wait_for(
                    workflow(
                        problem=problem['question'],
                        entry_point=problem['entry_point']
                    ),
                    timeout=self.timeout_per_problem
                )

                problem_time = time.time() - problem_start

                # 验证solution
                passed, validation_msg = self.validator.validate(solution, problem)

                if passed:
                    num_passed += 1
                    logger.info(f"[WorkflowEvaluator] {task_id}: ✅ PASSED - {validation_msg}")
                else:
                    logger.info(f"[WorkflowEvaluator] {task_id}: ❌ FAILED - {validation_msg}")

                results.append({
                    'task_id': task_id,
                    'passed': passed,
                    'time': problem_time,
                    'cost': cost,
                    'solution_length': len(solution) if solution else 0,
                    'validation_message': validation_msg
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
        pass_at_k = num_passed / len(problem_ids) if problem_ids else 0.0
        avg_time = total_time / len(problem_ids) if problem_ids else 0.0

        evaluation_result = {
            'pass_at_k': pass_at_k,
            'pass_at_1': pass_at_k,
            'num_passed': num_passed,
            'num_total': len(problem_ids),
            'avg_time': avg_time,
            'total_time': total_time,
            'details': results,
            'dataset': self.dataset,
            'evaluation_method': self.config.get('evaluation_method')
        }

        logger.info(f"[WorkflowEvaluator] ===== EVALUATION COMPLETE =====")
        logger.info(f"[WorkflowEvaluator] Pass@{len(problem_ids)}: {pass_at_k:.4f} ({num_passed}/{len(problem_ids)})")
        logger.info(f"[WorkflowEvaluator] Avg time: {avg_time:.2f}s")
        logger.info(f"[WorkflowEvaluator] Total time: {total_time:.2f}s")

        return evaluation_result

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


# ===========================
# 单例和工具函数
# ===========================

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

    evaluator = WorkflowEvaluator(dataset="AIME", sample_size=1)

    # 创建一个简单的测试workflow
    class DummyWorkflow:
        async def __call__(self, problem, entry_point):
            # 返回一个简单的solution
            solution = f"""
<answer>233168</answer>
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
