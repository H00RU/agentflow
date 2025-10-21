"""
Workflow Prompt Manager - 管理Qwen的prompt和输出格式
Manages prompts and output formats for Qwen to generate workflows
"""

from typing import Dict, Optional


class WorkflowPromptManager:
    """
    管理Qwen生成workflow的prompt

    功能：
    1. 提供system prompt告诉Qwen如何输出workflow
    2. 提供example展示正确的输出格式
    3. 根据观测构造完整的prompt
    """

    def __init__(self):
        """初始化prompt manager"""
        self.system_prompt = self._create_system_prompt()
        self.examples = self._create_examples()

    def _create_system_prompt(self) -> str:
        """创建system prompt"""
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

    def _create_examples(self) -> str:
        """创建示例"""
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


# 全局prompt manager单例
_global_prompt_manager = None


def get_prompt_manager() -> WorkflowPromptManager:
    """获取全局prompt manager"""
    global _global_prompt_manager
    if _global_prompt_manager is None:
        _global_prompt_manager = WorkflowPromptManager()
    return _global_prompt_manager


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
