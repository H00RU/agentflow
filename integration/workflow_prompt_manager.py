"""
Workflow Prompt Manager - 管理Qwen的prompt和输出格式
Manages prompts and output formats for Qwen to generate workflows
"""

from typing import Dict, Optional


class WorkflowPromptManager:
    """
    管理Qwen生成workflow的prompt

    支持多数据集：HumanEval (代码生成), AIME (数学题), 等

    功能：
    1. 提供system prompt告诉Qwen如何输出workflow
    2. 提供example展示正确的输出格式
    3. 根据观测构造完整的prompt
    """

    def __init__(self, dataset: str = "HumanEval"):
        """
        初始化prompt manager

        Args:
            dataset: 数据集名称 (HumanEval, AIME, etc.)
        """
        self.dataset = dataset.upper()
        self.system_prompt = self._create_system_prompt()
        self.examples = self._create_examples()

    def _create_system_prompt(self) -> str:
        """创建system prompt，根据dataset生成对应的prompt"""
        if self.dataset == "AIME":
            return self._create_aime_system_prompt()
        elif self.dataset == "HUMANEVAL":
            return self._create_humaneval_system_prompt()
        else:
            # 默认使用通用prompt
            return self._create_generic_system_prompt()

    def _create_aime_system_prompt(self) -> str:
        """创建AIME数学题的system prompt"""
        return """You are an AI workflow optimizer for mathematical problem-solving tasks.

Your task is to design and improve agent workflows that solve AIME (American Invitational Mathematics Examination) problems.

IMPORTANT OUTPUT FORMAT:
You must output your workflow design in the following XML format:

<workflow_modification>
[Brief description of the modification or improvement you're making]
</workflow_modification>

<operators>
[Comma-separated list of operators to use from available operators]
</operators>

<workflow_steps>
1. [First step description]
2. [Second step description]
3. [Third step description]
...
</workflow_steps>

Available Operators for AIME:
- Custom: Custom operator that can perform any reasoning task with specific instructions
- ScEnsemble: Self-consistency ensemble that generates multiple solutions and selects the best through voting
- Test: Tests and validates mathematical solutions
- Review: Reviews and analyzes solution quality
- Revise: Revises and improves solutions based on feedback

AIME Problem Characteristics:
- High-difficulty mathematics competition problems
- Answers are integers from 0 to 999
- Require deep mathematical knowledge (algebra, geometry, number theory, combinatorics)
- Often need multiple solution steps
- Benefit from multiple solution attempts and ensemble methods

Your workflow will be converted to executable Python code and tested on real AIME problems.
The performance (accuracy/pass@k) will be used as a reward signal to train you.

Focus on:
1. Selecting appropriate operators for mathematical reasoning
2. Using ensemble methods (ScEnsemble) for robust solutions
3. Incorporating review and revision steps for accuracy
4. Balancing exploration (trying new approaches) vs exploitation (improving known good methods)
5. Learning from previous workflow scores

Remember: Your output will be directly parsed and executed, so ALWAYS use the XML format above."""

    def _create_humaneval_system_prompt(self) -> str:
        """创建HumanEval代码生成的system prompt"""
        return """You are an AI workflow optimizer for code generation tasks.

Your task is to design and improve agent workflows that solve coding problems (HumanEval dataset).

IMPORTANT OUTPUT FORMAT:
You must output your workflow design in the following XML format:

<workflow_modification>
[Brief description of the modification or improvement you're making]
</workflow_modification>

<operators>
[Comma-separated list of operators to use, chosen from: Custom, CustomCodeGenerate, ScEnsemble, Test]
</operators>

<workflow_steps>
1. [First step description]
2. [Second step description]
3. [Third step description]
...
</workflow_steps>

Available Operators:
- Custom: Custom operator that can generate any content with specific instructions
- CustomCodeGenerate: Specialized operator for generating standard Python code
- ScEnsemble: Self-consistency ensemble that generates multiple solutions and selects the best one
- Test: Tests generated code against test cases

Your workflow will be converted to executable Python code and tested on real HumanEval problems.
The performance (pass@k) will be used as a reward signal to train you.

Focus on:
1. Selecting appropriate operators
2. Ordering operators effectively
3. Balancing exploration (trying new ideas) vs exploitation (improving known good approaches)
4. Learning from previous workflow scores

Remember: Your output will be directly parsed and executed, so ALWAYS use the XML format above."""

    def _create_generic_system_prompt(self) -> str:
        """创建通用的system prompt"""
        return f"""You are an AI workflow optimizer for problem-solving tasks.

Your task is to design and improve agent workflows that solve problems in the {self.dataset} dataset.

IMPORTANT OUTPUT FORMAT:
You must output your workflow design in the following XML format:

<workflow_modification>
[Brief description of the modification or improvement you're making]
</workflow_modification>

<operators>
[Comma-separated list of operators to use from available operators]
</operators>

<workflow_steps>
1. [First step description]
2. [Second step description]
3. [Third step description]
...
</workflow_steps>

Your workflow will be converted to executable Python code and tested on real problems.
The performance will be used as a reward signal to train you.

Focus on:
1. Selecting appropriate operators
2. Ordering operators effectively
3. Balancing exploration (trying new ideas) vs exploitation (improving known good approaches)
4. Learning from previous workflow scores

Remember: Your output will be directly parsed and executed, so ALWAYS use the XML format above."""

    def _create_examples(self) -> str:
        """创建示例，根据dataset生成对应的examples"""
        if self.dataset == "AIME":
            return self._create_aime_examples()
        elif self.dataset == "HUMANEVAL":
            return self._create_humaneval_examples()
        else:
            return self._create_generic_examples()

    def _create_aime_examples(self) -> str:
        """创建AIME数学题的示例"""
        return """
Example 1 - Simple Math Workflow:
<workflow_modification>
Use Custom operator to solve mathematical problems with step-by-step reasoning
</workflow_modification>

<operators>
Custom
</operators>

<workflow_steps>
1. Use Custom operator to analyze the problem and identify required mathematical concepts
2. Apply systematic reasoning to solve the problem step by step
3. Extract and validate the numerical answer (0-999)
</workflow_steps>


Example 2 - Ensemble Math Workflow:
<workflow_modification>
Add self-consistency ensemble to improve solution accuracy by generating and comparing multiple solution attempts
</workflow_modification>

<operators>
Custom, ScEnsemble
</operators>

<workflow_steps>
1. Use Custom operator to generate 3-5 independent solution attempts
2. Use ScEnsemble to analyze all solutions and select the most consistent answer
3. Validate that the final answer is an integer between 0 and 999
</workflow_steps>


Example 3 - Review and Revise Workflow:
<workflow_modification>
Add review and revision steps to catch and correct mathematical errors
</workflow_modification>

<operators>
Custom, Review, Revise
</operators>

<workflow_steps>
1. Use Custom operator to generate initial solution
2. Use Review operator to check solution for mathematical errors and logical gaps
3. Use Revise operator to fix identified issues and improve the solution
4. Extract and return the final answer
</workflow_steps>
"""

    def _create_humaneval_examples(self) -> str:
        """创建HumanEval代码生成的示例"""
        return """
Example 1 - Simple Workflow:
<workflow_modification>
Use CustomCodeGenerate to directly generate code solutions
</workflow_modification>

<operators>
CustomCodeGenerate
</operators>

<workflow_steps>
1. Use CustomCodeGenerate to generate a Python function that solves the problem
2. Return the generated code
</workflow_steps>


Example 2 - Ensemble Workflow:
<workflow_modification>
Add self-consistency ensemble to improve solution quality by generating and comparing multiple candidates
</workflow_modification>

<operators>
CustomCodeGenerate, ScEnsemble
</operators>

<workflow_steps>
1. Use CustomCodeGenerate to generate 3 candidate solutions
2. Use ScEnsemble to analyze all candidates and select the most consistent solution
3. Return the selected solution
</workflow_steps>


Example 3 - Test-driven Workflow:
<workflow_modification>
Add testing step to validate generated code before returning
</workflow_modification>

<operators>
CustomCodeGenerate, Test
</operators>

<workflow_steps>
1. Use CustomCodeGenerate to generate initial code solution
2. Use Test operator to run test cases on the solution
3. If tests fail, generate improved solution based on test feedback
4. Return the final solution
</workflow_steps>
"""

    def _create_generic_examples(self) -> str:
        """创建通用的示例"""
        return """
Example 1 - Simple Workflow:
<workflow_modification>
Use Custom operator to solve the problem directly
</workflow_modification>

<operators>
Custom
</operators>

<workflow_steps>
1. Use Custom operator to analyze and solve the problem
2. Return the solution
</workflow_steps>


Example 2 - Ensemble Workflow:
<workflow_modification>
Add self-consistency ensemble to improve solution quality
</workflow_modification>

<operators>
Custom, ScEnsemble
</operators>

<workflow_steps>
1. Use Custom operator to generate multiple solution attempts
2. Use ScEnsemble to select the best solution
3. Return the final solution
</workflow_steps>
"""

    def get_system_prompt(self) -> str:
        """获取system prompt"""
        return self.system_prompt

    def get_examples(self) -> str:
        """获取示例"""
        return self.examples

    def construct_full_prompt(
        self,
        observation: str,
        include_examples: bool = True,
        include_system: bool = False
    ) -> str:
        """
        构造完整的prompt

        Args:
            observation: 环境的观测
            include_examples: 是否包含示例
            include_system: 是否包含system prompt（如果False，system prompt应该单独设置）

        Returns:
            完整的prompt文本
        """
        parts = []

        if include_system:
            parts.append(self.system_prompt)
            parts.append("\n" + "="*70 + "\n")

        parts.append("CURRENT SITUATION:")
        parts.append(observation)

        if include_examples:
            parts.append("\n" + "="*70)
            parts.append("\nEXAMPLES:")
            parts.append(self.examples)
            parts.append("\n" + "="*70 + "\n")

        parts.append("\nNow, design your workflow using the XML format described above:")

        return "\n".join(parts)

    def validate_output(self, output: str) -> Dict[str, bool]:
        """
        验证Qwen输出是否符合格式

        Args:
            output: Qwen的输出

        Returns:
            验证结果字典
        """
        validation = {
            'has_modification': '<workflow_modification>' in output,
            'has_operators': '<operators>' in output,
            'has_steps': '<workflow_steps>' in output,
            'all_required_fields': False
        }

        validation['all_required_fields'] = all([
            validation['has_modification'],
            validation['has_operators'],
            validation['has_steps']
        ])

        return validation

    def create_feedback_prompt(
        self,
        invalid_output: str,
        validation: Dict[str, bool]
    ) -> str:
        """
        创建反馈prompt（当输出格式错误时）

        Args:
            invalid_output: 错误的输出
            validation: 验证结果

        Returns:
            反馈prompt
        """
        feedback = "Your previous output did not follow the required format.\n\n"

        if not validation['has_modification']:
            feedback += "❌ Missing <workflow_modification> tag\n"
        if not validation['has_operators']:
            feedback += "❌ Missing <operators> tag\n"
        if not validation['has_steps']:
            feedback += "❌ Missing <workflow_steps> tag\n"

        feedback += "\nPlease provide your workflow design using the correct XML format:\n\n"
        feedback += self.system_prompt.split("IMPORTANT OUTPUT FORMAT:")[1].split("Available Operators:")[0]

        return feedback


# 全局prompt manager缓存（按dataset缓存）
_prompt_manager_cache: Dict[str, WorkflowPromptManager] = {}


def get_prompt_manager(dataset: str = "HumanEval") -> WorkflowPromptManager:
    """
    获取prompt manager（按dataset缓存）

    Args:
        dataset: 数据集名称 (HumanEval, AIME, etc.)

    Returns:
        对应dataset的prompt manager
    """
    dataset_upper = dataset.upper()
    if dataset_upper not in _prompt_manager_cache:
        _prompt_manager_cache[dataset_upper] = WorkflowPromptManager(dataset=dataset)
    return _prompt_manager_cache[dataset_upper]


if __name__ == "__main__":
    # 测试
    manager = WorkflowPromptManager()

    print("System Prompt:")
    print(manager.get_system_prompt())

    print("\n" + "="*70)
    print("\nExamples:")
    print(manager.get_examples())

    # 测试验证
    test_output = """
<workflow_modification>
Test modification
</workflow_modification>

<operators>
CustomCodeGenerate
</operators>

<workflow_steps>
1. Step 1
2. Step 2
</workflow_steps>
"""

    validation = manager.validate_output(test_output)
    print("\n" + "="*70)
    print("\nValidation result:")
    print(validation)

    # 测试incomplete output
    incomplete_output = "Just some text without proper format"
    validation = manager.validate_output(incomplete_output)
    print("\nIncomplete output validation:")
    print(validation)

    if not validation['all_required_fields']:
        feedback = manager.create_feedback_prompt(incomplete_output, validation)
        print("\nFeedback for incomplete output:")
        print(feedback)
