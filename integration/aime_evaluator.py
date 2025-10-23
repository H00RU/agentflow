"""
AIME Evaluator for Mathematical Problem Solving
Evaluates workflow performance on AIME mathematics problems
"""

import json
import re
import asyncio
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path
from scripts.logs import logger


class AIMEEvaluator:
    """
    Evaluator for AIME mathematical problems

    Unlike code generation, math problems require:
    - Answer extraction from natural language
    - Numerical comparison (exact match)
    - Multi-step reasoning evaluation
    """

    def __init__(
        self,
        llm_config: Dict[str, Any],
        dataset_path: str = "/content/agentflow/AFlow/data/AIME_2024.jsonl",
        sample_size: int = 30
    ):
        """
        Args:
            llm_config: LLM configuration for workflow execution
            dataset_path: Path to AIME dataset JSONL file
            sample_size: Default number of problems to test (default: 30)
        """
        self.llm_config = llm_config
        self.dataset_path = dataset_path
        self.sample_size = sample_size
        self.problems = {}
        self.dataset_type = "AIME"

    async def initialize(self):
        """Load AIME dataset"""
        logger.info(f"[AIMEEvaluator] Loading AIME dataset from {self.dataset_path}")

        try:
            with open(self.dataset_path, 'r', encoding='utf-8') as f:
                for line in f:
                    problem = json.loads(line)
                    task_id = problem['task_id']
                    self.problems[task_id] = problem

            logger.info(f"[AIMEEvaluator] Loaded {len(self.problems)} AIME problems")

        except FileNotFoundError:
            logger.warning(f"[AIMEEvaluator] Dataset not found, creating dummy data")
            self.problems = {
                "AIME-DUMMY-1": {
                    "task_id": "AIME-DUMMY-1",
                    "problem": "What is 2 + 2?",
                    "answer": "4"
                }
            }

    async def evaluate_workflow(
        self,
        workflow: Any,
        num_problems: Optional[int] = None,
        use_test_set: bool = False,
        random_sample: bool = False
    ) -> Dict[str, Any]:
        """
        Evaluate workflow on AIME problems

        Args:
            workflow: Workflow instance to evaluate
            num_problems: Number of problems to test (None = use sample_size from init)
            use_test_set: Whether to use test set (last 20%) or train set (first 80%)
            random_sample: Whether to randomly sample problems

        Returns:
            Evaluation results
        """
        # Use sample_size from init if num_problems not specified
        if num_problems is None:
            num_problems = self.sample_size
        logger.info(f"[AIMEEvaluator] Starting evaluation")
        logger.info(f"[AIMEEvaluator] Dataset: AIME")
        logger.info(f"[AIMEEvaluator] Total problems: {len(self.problems)}")

        # Split dataset
        all_task_ids = sorted(list(self.problems.keys()))
        total = len(all_task_ids)
        train_size = int(total * 0.8)

        if use_test_set:
            available_ids = all_task_ids[train_size:]
            logger.info(f"[AIMEEvaluator] 📚 Using TEST set ({len(available_ids)} problems)")
        else:
            available_ids = all_task_ids[:train_size]
            logger.info(f"[AIMEEvaluator] 📚 Using TRAIN set ({len(available_ids)} problems)")

        # Sample problems
        if random_sample and num_problems < len(available_ids):
            import random
            test_ids = random.sample(available_ids, num_problems)
        else:
            test_ids = available_ids[:num_problems]

        logger.info(f"[AIMEEvaluator] 📋 Testing on {len(test_ids)} problems")

        # Evaluate each problem
        passed = 0
        failed = 0
        results = []

        for i, task_id in enumerate(test_ids, 1):
            problem = self.problems[task_id]
            logger.info(f"[AIMEEvaluator] [{i}/{len(test_ids)}] Testing {task_id}...")

            try:
                # Execute workflow
                answer, cost = await workflow(problem['problem'])

                # Check answer
                is_correct = self._check_answer(answer, problem['answer'])

                if is_correct:
                    passed += 1
                    logger.info(f"[AIMEEvaluator] {task_id}: ✅ PASSED")
                else:
                    failed += 1
                    logger.info(f"[AIMEEvaluator] {task_id}: ❌ FAILED (expected: {problem['answer']}, got: {self._extract_answer(answer)})")

                results.append({
                    'task_id': task_id,
                    'passed': is_correct,
                    'expected': problem['answer'],
                    'actual': self._extract_answer(answer)
                })

            except Exception as e:
                failed += 1
                logger.error(f"[AIMEEvaluator] {task_id}: ❌ ERROR: {e}")
                results.append({
                    'task_id': task_id,
                    'passed': False,
                    'error': str(e)
                })

        # Calculate metrics
        pass_rate = passed / len(test_ids) if test_ids else 0.0

        logger.info(f"[AIMEEvaluator] ===== EVALUATION COMPLETE =====")
        logger.info(f"[AIMEEvaluator] Pass@{len(test_ids)}: {pass_rate:.4f} ({passed}/{len(test_ids)})")

        return {
            'pass_at_k': pass_rate,
            'passed': passed,
            'failed': failed,
            'total': len(test_ids),
            'results': results
        }

    def _extract_answer(self, response: str) -> str:
        """
        Extract numerical answer from response text

        AIME answers are integers from 0 to 999.
        Common formats:
        - "The answer is 42"
        - "Therefore, the answer is \\boxed{42}"
        - "So m + n = 42"
        """
        if not response:
            return ""

        # Try to find boxed answer first
        boxed_match = re.search(r'\\boxed\{(\d+)\}', response)
        if boxed_match:
            return boxed_match.group(1)

        # Try to find "answer is X"
        answer_match = re.search(r'answer is[:\s]+(\d+)', response, re.IGNORECASE)
        if answer_match:
            return answer_match.group(1)

        # Try to find final number in the response
        numbers = re.findall(r'\b\d{1,3}\b', response)
        if numbers:
            return numbers[-1]  # Take last number as answer

        return ""

    def _check_answer(self, response: str, correct_answer: str) -> bool:
        """
        Check if response contains correct answer

        Args:
            response: Workflow output
            correct_answer: Ground truth answer

        Returns:
            True if answer is correct
        """
        extracted = self._extract_answer(response)

        if not extracted:
            return False

        # Normalize answers
        try:
            extracted_num = int(extracted)
            correct_num = int(correct_answer)
            return extracted_num == correct_num
        except ValueError:
            # String comparison as fallback
            return extracted.strip() == str(correct_answer).strip()


# For compatibility with existing code
WorkflowEvaluator = AIMEEvaluator
