"""
AIME Benchmark - American Invitational Mathematics Examination
数学竞赛题评估
"""

import re
from typing import Callable, List, Tuple
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_fixed

from benchmarks.benchmark import BaseBenchmark
from scripts.logs import logger


class AIMEBenchmark(BaseBenchmark):
    """
    AIME Benchmark for mathematical problem solving

    AIME problems have integer answers from 0 to 999.
    """

    def __init__(self, name: str, file_path: str, log_path: str):
        super().__init__(name, file_path, log_path)

    def extract_model_answer(self, text: str) -> str:
        """
        Extract numerical answer from model output

        AIME answers are integers from 0 to 999.
        Common formats:
        - "The answer is 42"
        - "Therefore, the answer is \\boxed{42}"
        - "So m + n = 42"
        """
        if not text:
            return ""

        # Try to find boxed answer first (LaTeX format)
        boxed_match = re.search(r'\\boxed\{(\d+)\}', text)
        if boxed_match:
            return boxed_match.group(1)

        # Try to find "answer is X" pattern
        answer_match = re.search(r'answer is[:\s]+(\d+)', text, re.IGNORECASE)
        if answer_match:
            return answer_match.group(1)

        # Try to find "= X" at end of sentence
        equals_match = re.search(r'=\s*(\d+)\s*[.!?]?\s*$', text)
        if equals_match:
            return equals_match.group(1)

        # Try to find final number in the response (0-999 range for AIME)
        numbers = re.findall(r'\b\d{1,3}\b', text)
        if numbers:
            return numbers[-1]  # Take last number as answer

        return ""

    def calculate_score(self, expected_output: str, prediction: str) -> Tuple[int, str]:
        """
        Calculate score by comparing predicted answer with expected answer

        Args:
            expected_output: Ground truth answer (string or full solution)
            prediction: Model's output text

        Returns:
            (score, extracted_answer) where score is 1 for correct, 0 for incorrect
        """
        # Extract answer from model prediction
        predicted_answer = self.extract_model_answer(prediction)

        # Extract ground truth answer (may be just the number or full solution)
        if expected_output.isdigit():
            expected_answer = expected_output
        else:
            expected_answer = self.extract_model_answer(expected_output)

        # Compare answers
        if not predicted_answer:
            return 0, predicted_answer

        try:
            predicted_num = int(predicted_answer)
            expected_num = int(expected_answer)

            # AIME answers are 0-999
            if 0 <= predicted_num <= 999 and predicted_num == expected_num:
                return 1, predicted_answer
            else:
                return 0, predicted_answer
        except ValueError:
            # Fallback to string comparison
            if predicted_answer.strip() == expected_answer.strip():
                return 1, predicted_answer
            return 0, predicted_answer

    @retry(stop=stop_after_attempt(5), wait=wait_fixed(1), retry=retry_if_exception_type(Exception), reraise=True)
    async def _generate_output(self, graph, input_text):
        """Generate output from workflow with retry logic"""
        return await graph(input_text)

    async def evaluate_problem(self, problem: dict, graph: Callable) -> Tuple[str, str, str, int, float]:
        """
        Evaluate a single AIME problem

        Args:
            problem: Problem dict with 'problem' and 'answer' fields
            graph: Workflow to evaluate

        Returns:
            (question, prediction, expected_output, score, cost)
        """
        input_text = problem["problem"]
        expected_output = problem.get("answer", problem.get("solution", ""))

        try:
            # Execute workflow
            output, cost = await self._generate_output(graph, input_text)

            # Calculate score
            uni_score, extracted_output = self.calculate_score(expected_output, output)

            # Log failures
            if uni_score == 0:
                self.log_mismatch(
                    input_text,
                    expected_output,
                    output,
                    extracted_output,
                )

            return input_text, output, expected_output, uni_score, cost

        except Exception as e:
            logger.error(f"Error evaluating problem: {e}")
            return input_text, str(e), expected_output, 0, 0.0

    def get_result_columns(self) -> List[str]:
        """Return column names for results CSV"""
        return ["question", "prediction", "expected_output", "score", "cost"]
