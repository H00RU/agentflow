"""
Workflow Code Prompt Manager - 让Qwen直接生成完整Python代码
完全对齐原版AFlow设计理念：LLM → Python代码 → 执行

与 workflow_prompt_manager.py 的区别：
- 旧版：要求Qwen输出XML描述 → Parser转换为代码
- 新版：要求Qwen直接输出完整Python代码（无Parser）

这与原版AFlow的设计完全一致。
"""

from typing import List, Dict, Optional


class WorkflowCodePromptManager:
    """
    管理Qwen生成完整Python workflow代码的prompt

    完全对齐原版AFlow设计：
    - LLM直接生成可执行的Python代码
    - 代码空间搜索（而非描述空间）
    - 经验池驱动的迭代优化
    """

    def __init__(self, dataset: str = "HumanEval"):
        self.dataset = dataset.upper()

    def get_system_prompt(self) -> str:
        """
        生成system prompt - 要求Qwen输出完整Python代码

        与原版AFlow prompt对齐：
        - 要求LLM生成class Workflow
        - 指定可用的operators
        - 提供代码示例
        - 强调代码必须可执行
        """

        operators_desc = self._get_available_operators()
        dataset_rules = self._get_dataset_specific_rules()
        examples = self._get_code_examples()

        prompt = f"""You are an expert workflow designer for {self.dataset} problems.

Your task is to generate COMPLETE, EXECUTABLE PYTHON CODE for a workflow that solves {self._get_problem_description()} problems.

## CRITICAL OUTPUT FORMAT (STRICTLY REQUIRED):

You MUST output in this exact XML structure:

<modification>
Brief description of your workflow design or changes from previous attempts.
Example: "Use ensemble with 15 samples to improve accuracy on hard AIME problems"
</modification>

<graph>
class Workflow:
    def __init__(self, name: str, llm_config, dataset: str) -> None:
        self.name = name
        self.dataset = dataset
        self.llm = create_llm_instance(llm_config)

        # Initialize the operators you need
        # Available operators: {', '.join([op.split(':')[0] for op in operators_desc.split('\n') if op.strip()])}
        self.custom = operator.Custom(self.llm)
        self.sc_ensemble = operator.ScEnsemble(self.llm)
        # Add more as needed: Review, Revise, etc.

    async def __call__(self, problem: str, entry_point: Optional[str] = None):
        # YOUR COMPLETE WORKFLOW LOGIC HERE
        # This code will be executed directly!

        # You have FULL CONTROL over:
        # 1. Which operators to use
        # 2. How many samples to generate
        # 3. Control flow (if/else, loops)
        # 4. Operator calling order

        # MUST return (solution_string, cost) tuple
        return solution, 0.0
</graph>

<prompt>
# Custom prompts for prompt_custom (if needed)
# Most workflows don't need this - leave empty if not using prompt_custom
</prompt>

## AVAILABLE OPERATORS:
{operators_desc}

## DATASET-SPECIFIC RULES FOR {self.dataset}:
{dataset_rules}

## CODE EXAMPLES:
{examples}

## CRITICAL REQUIREMENTS:
1. Your code MUST be syntactically correct Python - it will be executed directly
2. MUST return (solution, cost) tuple from __call__
3. MUST use async/await for operator calls
4. You can use Python control flow: if/else, for/while loops, list comprehensions
5. You can choose sampling counts, operator combinations freely
6. The code runs in environment with: operator, create_llm_instance, Optional already imported
7. For {self.dataset}: {dataset_rules}

## LEARNING FROM EXPERIENCE:
Previous workflows and their scores will be provided in the observation.
- High score → make small refinements
- Low score → try different operators or logic
- Balance exploration (new ideas) vs computation cost

Remember: This is NOT pseudocode. This is REAL Python code that will execute.
Any syntax errors or runtime errors will result in failure and negative reward.
"""
        return prompt

    def _get_problem_description(self) -> str:
        """获取问题类型描述"""
        if self.dataset == "AIME":
            return "advanced mathematical reasoning (AIME competition)"
        elif self.dataset == "HUMANEVAL":
            return "Python code generation"
        elif self.dataset == "MBPP":
            return "Python code generation"
        elif self.dataset == "GSM8K":
            return "grade school math problems"
        elif self.dataset == "MATH":
            return "mathematical reasoning"
        else:
            return "problem-solving"

    def _get_available_operators(self) -> str:
        """获取可用operators的描述"""
        if self.dataset in ["AIME", "MATH", "GSM8K"]:
            return """
- Custom: General-purpose LLM call for mathematical reasoning
  Usage: await self.custom(input=problem, instruction="your instruction")
  Returns: {'response': str, ...}

- ScEnsemble: Self-consistency ensemble - generate multiple solutions and vote
  Usage: await self.sc_ensemble(solutions=[sol1, sol2, ...], problem=problem)
  Returns: {'response': str, ...}

- Review: Review and critique a solution for errors
  Usage: await self.review(problem=problem, solution=solution)
  Returns: {'feedback': str, ...}

- Revise: Improve a solution based on feedback
  Usage: await self.revise(problem=problem, solution=solution, feedback=feedback)
  Returns: {'response': str, ...}
"""
        elif self.dataset in ["HUMANEVAL", "MBPP"]:
            return """
- Custom: General-purpose LLM call (can be used for code generation)
  Usage: await self.custom(input=problem, instruction="your instruction")
  Returns: {'response': str, ...}

- CustomCodeGenerate: Specialized for generating Python code
  Usage: await self.custom_code_generate(problem=problem, entry_point=entry_point, instruction="")
  Returns: {'response': str, ...}

- ScEnsemble: Self-consistency ensemble
  Usage: await self.sc_ensemble(solutions=[sol1, sol2, ...], problem=problem)
  Returns: {'response': str, ...}

- Test: Test generated code (though external evaluator handles this)
  Usage: await self.test(code=code, problem=problem)
  Returns: test results

- Review: Review code quality
  Usage: await self.review(problem=problem, solution=code)
  Returns: {'feedback': str, ...}
"""
        else:
            return """
- Custom: General-purpose LLM call
- ScEnsemble: Self-consistency ensemble
- Review: Review solutions
- Revise: Improve solutions
"""

    def _get_dataset_specific_rules(self) -> str:
        """获取数据集特定规则"""
        if self.dataset == "AIME":
            return """
- Solution MUST be an integer between 0 and 999
- Use Custom operator (NOT CustomCodeGenerate) for mathematical reasoning
- Focus on step-by-step reasoning and answer extraction
- Common strategy: generate multiple solutions, use ensemble to select best
"""
        elif self.dataset in ["HUMANEVAL", "MBPP"]:
            return """
- Solution must be valid Python function code
- Use CustomCodeGenerate or Custom operator
- Code must match the function signature specified in entry_point
- Ensure proper indentation and syntax
"""
        elif self.dataset == "GSM8K":
            return """
- Solution should be a numerical answer
- Show step-by-step reasoning
- Extract final numerical answer clearly
"""
        else:
            return "Follow standard problem-solving practices."

    def _get_code_examples(self) -> str:
        """提供代码示例"""
        if self.dataset in ["AIME", "MATH", "GSM8K"]:
            return """
Example 1 - Simple single-shot approach:
<graph>
class Workflow:
    def __init__(self, name: str, llm_config, dataset: str) -> None:
        self.name = name
        self.dataset = dataset
        self.llm = create_llm_instance(llm_config)
        self.custom = operator.Custom(self.llm)

    async def __call__(self, problem: str, entry_point: Optional[str] = None):
        result = await self.custom(
            input=problem,
            instruction="Solve this math problem step by step. Provide final answer as integer 0-999."
        )
        solution = result['response']
        return solution, 0.0
</graph>

Example 2 - Self-consistency ensemble (recommended for AIME):
<graph>
class Workflow:
    def __init__(self, name: str, llm_config, dataset: str) -> None:
        self.name = name
        self.dataset = dataset
        self.llm = create_llm_instance(llm_config)
        self.custom = operator.Custom(self.llm)
        self.sc_ensemble = operator.ScEnsemble(self.llm)

    async def __call__(self, problem: str, entry_point: Optional[str] = None):
        # Generate multiple solution attempts
        solutions = []
        for i in range(15):  # You can adjust this number
            sol = await self.custom(
                input=problem,
                instruction="Solve this AIME problem step by step. Think carefully and provide your final answer as an integer between 0 and 999."
            )
            solutions.append(sol['response'])

        # Use ensemble to select most consistent answer
        result = await self.sc_ensemble(solutions=solutions, problem=problem)
        solution = result['response']

        return solution, 0.0
</graph>

Example 3 - Review and Revise approach:
<graph>
class Workflow:
    def __init__(self, name: str, llm_config, dataset: str) -> None:
        self.name = name
        self.dataset = dataset
        self.llm = create_llm_instance(llm_config)
        self.custom = operator.Custom(self.llm)
        self.review = operator.Review(self.llm)
        self.revise = operator.Revise(self.llm)

    async def __call__(self, problem: str, entry_point: Optional[str] = None):
        # Initial solution
        initial = await self.custom(
            input=problem,
            instruction="Solve this math problem step by step."
        )
        solution = initial['response']

        # Review for errors
        review_result = await self.review(problem=problem, solution=solution)
        feedback = review_result.get('feedback', '')

        # Revise if needed
        if feedback and len(feedback) > 10:
            revised = await self.revise(
                problem=problem,
                solution=solution,
                feedback=feedback
            )
            solution = revised['response']

        return solution, 0.0
</graph>
"""
        elif self.dataset in ["HUMANEVAL", "MBPP"]:
            return """
Example 1 - Simple code generation:
<graph>
class Workflow:
    def __init__(self, name: str, llm_config, dataset: str) -> None:
        self.name = name
        self.dataset = dataset
        self.llm = create_llm_instance(llm_config)
        self.custom_code_generate = operator.CustomCodeGenerate(self.llm)

    async def __call__(self, problem: str, entry_point: Optional[str] = None):
        result = await self.custom_code_generate(
            problem=problem,
            entry_point=entry_point,
            instruction=""
        )
        solution = result['response']
        return solution, 0.0
</graph>

Example 2 - Ensemble code generation:
<graph>
class Workflow:
    def __init__(self, name: str, llm_config, dataset: str) -> None:
        self.name = name
        self.dataset = dataset
        self.llm = create_llm_instance(llm_config)
        self.custom_code_generate = operator.CustomCodeGenerate(self.llm)
        self.sc_ensemble = operator.ScEnsemble(self.llm)

    async def __call__(self, problem: str, entry_point: Optional[str] = None):
        # Generate multiple code candidates
        solutions = []
        for i in range(3):
            sol = await self.custom_code_generate(
                problem=problem,
                entry_point=entry_point,
                instruction=""
            )
            solutions.append(sol['response'])

        # Select best via ensemble
        result = await self.sc_ensemble(solutions=solutions, problem=problem)
        solution = result['response']

        return solution, 0.0
</graph>
"""
        else:
            return "See AIME examples above and adapt to your dataset."

    def format_observation(self,
                          round_num: int,
                          history: List[Dict],
                          dataset: str,
                          current_best_score: float = 0.0) -> str:
        """
        格式化observation - 包含历史workflow和分数

        与原版AFlow的observation格式对齐
        """

        obs = f"""## Workflow Design Task - Round {round_num}
Dataset: {dataset}

## Your Task:
Design a complete Python workflow to solve {dataset} problems.
Your workflow will be executed on real test cases and scored.

## Previous Workflows and Performance:
"""

        if history:
            # 显示最近5个workflow
            recent_history = history[-5:] if len(history) > 5 else history
            for i, entry in enumerate(recent_history):
                score = entry.get('score', 0.0)
                modification = entry.get('modification', 'N/A')
                round_id = entry.get('round', i+1)

                obs += f"\n### Workflow {round_id} - Score: {score:.4f}\n"
                obs += f"Modification: {modification}\n"

                # 如果有operators信息
                if 'operators' in entry:
                    obs += f"Operators used: {', '.join(entry['operators'])}\n"

                # 如果有代码片段（显示关键部分）
                if 'code_snippet' in entry:
                    obs += f"Key code:\n```python\n{entry['code_snippet'][:200]}...\n```\n"

                obs += "\n"

            obs += f"\n**Current Best Score: {current_best_score:.4f}**\n"
        else:
            obs += "No previous workflows yet. This is your first design.\n"
            obs += "Start with a simple approach and iterate based on results.\n"

        obs += f"""
## Design Guidelines:
1. Output COMPLETE Python code in <graph> tags
2. Code will be executed directly - ensure it's syntactically correct
3. For {dataset}: {self._get_dataset_specific_rules()}
4. Balance effectiveness vs computational cost
5. Learn from previous attempts - refine what works, try new ideas for what doesn't

## Your Output:
Generate your workflow following the exact XML format specified in your system prompt.
Start with <modification>, then <graph>, then <prompt> (if needed).

GO:
"""
        return obs

    def get_available_operators(self) -> List[str]:
        """返回可用的operator列表（用于配置）"""
        if self.dataset in ["AIME", "MATH", "GSM8K"]:
            return ["Custom", "ScEnsemble", "Review", "Revise"]
        elif self.dataset in ["HUMANEVAL", "MBPP"]:
            return ["Custom", "CustomCodeGenerate", "ScEnsemble", "Test", "Review"]
        else:
            return ["Custom", "ScEnsemble", "Review", "Revise"]


def get_code_prompt_manager(dataset: str = "HumanEval") -> WorkflowCodePromptManager:
    """
    工厂函数：创建WorkflowCodePromptManager实例

    Args:
        dataset: 数据集名称

    Returns:
        WorkflowCodePromptManager实例
    """
    return WorkflowCodePromptManager(dataset=dataset)
