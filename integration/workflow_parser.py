"""
Workflow Parser - 将Qwen输出转换为AFlow可执行的Workflow代码
Converts Qwen output to AFlow executable workflow code
"""

import re
import os
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class WorkflowSpec:
    """Workflow规格说明"""
    modification: str  # 修改说明
    operators: List[str]  # 使用的operators
    steps: List[str]  # 执行步骤
    workflow_code: str  # 生成的workflow代码


class WorkflowParser:
    """
    解析Qwen生成的workflow描述，转换成AFlow可执行代码
    """

    # 支持的operators及其说明
    SUPPORTED_OPERATORS = {
        "Custom": "自定义operator，可以生成任何内容（推荐用于数学/推理任务）",
        "CustomCodeGenerate": "生成标准代码（仅用于代码生成任务）",
        "ScEnsemble": "自一致性集成，从多个方案中选择最佳",
        "Test": "测试代码，返回测试结果",
        "Review": "检查和改进解决方案",
        "Revise": "修订和改进解决方案"
    }

    def __init__(self):
        self.template_path = None

    def parse_qwen_output(self, qwen_output: str, dataset_type: str = "HumanEval") -> Optional[WorkflowSpec]:
        """
        解析Qwen的输出文本

        Args:
            qwen_output: Qwen生成的workflow描述
            dataset_type: 数据集类型 (HumanEval, AIME, 等)

        Returns:
            WorkflowSpec或None（如果解析失败）
        """
        try:
            # 提取modification
            modification = self._extract_field(qwen_output, "workflow_modification")
            if not modification:
                modification = self._extract_field(qwen_output, "modification")
            if not modification:
                modification = "Optimize workflow structure"

            # 提取operators
            operators = self._extract_operators(qwen_output)
            if not operators:
                # 根据数据集类型选择默认operator
                if dataset_type.upper() in ["AIME", "MATH", "GSM8K"]:
                    operators = ["Custom", "ScEnsemble"]  # 数学任务：用Custom让LLM推理
                else:
                    operators = ["CustomCodeGenerate"]  # 代码任务：用代码生成器

            # 提取步骤
            steps = self._extract_steps(qwen_output)
            if not steps:
                # 根据数据集类型设置默认步骤
                if dataset_type.upper() in ["HUMANEVAL", "MBPP", "CODEEVAL"]:
                    steps = ["Generate code solution"]
                else:
                    steps = ["Generate solution for the problem"]

            # 生成workflow代码
            workflow_code = self._generate_workflow_code(operators, steps, dataset_type)

            return WorkflowSpec(
                modification=modification,
                operators=operators,
                steps=steps,
                workflow_code=workflow_code
            )

        except Exception as e:
            print(f"[WorkflowParser] Error parsing Qwen output: {e}")
            return None

    def _extract_field(self, text: str, field_name: str) -> str:
        """提取XML标签内容"""
        pattern = rf"<{field_name}>(.*?)</{field_name}>"
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return ""

    def _extract_operators(self, text: str) -> List[str]:
        """提取operators列表"""
        # 方法1: 从<operators>标签提取
        operators_text = self._extract_field(text, "operators")
        if operators_text:
            # 分割operator名称
            operators = [op.strip() for op in re.split(r'[,，、\n]', operators_text) if op.strip()]
            # 只保留支持的operators
            valid_operators = [op for op in operators if op in self.SUPPORTED_OPERATORS]
            if valid_operators:
                return valid_operators

        # 方法2: 从文本中搜索operator名称
        found_operators = []
        for op_name in self.SUPPORTED_OPERATORS.keys():
            if op_name.lower() in text.lower():
                found_operators.append(op_name)

        if found_operators:
            return list(set(found_operators))  # 去重

        # 方法3: 使用智能推断
        return self._infer_operators(text)

    def _infer_operators(self, text: str) -> List[str]:
        """根据文本内容智能推断需要的operators"""
        operators = []
        text_lower = text.lower()

        # 关键词映射
        if any(word in text_lower for word in ["ensemble", "vote", "select", "best", "multiple"]):
            operators.append("ScEnsemble")

        if any(word in text_lower for word in ["test", "validate", "check", "verify"]):
            operators.append("Test")

        if any(word in text_lower for word in ["review", "improve", "refine"]):
            operators.append("Review")

        if any(word in text_lower for word in ["revise", "modify", "update"]):
            operators.append("Revise")

        # 默认使用Custom（更通用）
        if not operators:
            operators.append("Custom")

        return operators

    def _extract_steps(self, text: str) -> List[str]:
        """提取workflow执行步骤"""
        # 方法1: 从<workflow_steps>标签提取
        steps_text = self._extract_field(text, "workflow_steps")
        if not steps_text:
            steps_text = self._extract_field(text, "steps")

        if steps_text:
            # 提取编号的步骤
            step_pattern = r'(\d+[\.\):])\s*(.+?)(?=\n\d+[\.\):]|\n*$)'
            matches = re.findall(step_pattern, steps_text, re.DOTALL)
            if matches:
                return [match[1].strip() for match in matches]

            # 如果没有编号，按行分割
            lines = [line.strip() for line in steps_text.split('\n') if line.strip()]
            if lines:
                return lines

        # 方法2: 从整个文本推断
        return self._infer_steps(text)

    def _infer_steps(self, text: str) -> List[str]:
        """从文本推断执行步骤"""
        steps = []
        text_lower = text.lower()

        # 根据关键词推断步骤
        if any(word in text_lower for word in ["math", "calculate", "solve"]):
            steps.append("Solve the math problem")
        elif "code" in text_lower or "generate" in text_lower:
            steps.append("Generate code solution")
        else:
            steps.append("Generate solution for the problem")

        if "ensemble" in text_lower or "multiple" in text_lower:
            steps.append("Use ScEnsemble to select best solution")

        if "test" in text_lower or "validate" in text_lower:
            steps.append("Test the solution")

        return steps

    def _generate_workflow_code(self, operators: List[str], steps: List[str], dataset_type: str = "HumanEval") -> str:
        """
        生成AFlow workflow Python代码

        Args:
            operators: operator列表
            steps: 执行步骤
            dataset_type: 数据集类型 (HumanEval, AIME, 等)

        Returns:
            可执行的workflow代码
        """
        # 导入语句 - 泛用版本，不依赖DatasetType枚举
        # 这允许支持任意新数据集（如AIME）而无需修改evaluator.py
        imports = """from typing import Literal, Optional, Union
from scripts import operators as operator
from scripts.async_llm import create_llm_instance
"""

        # 根据数据集类型决定主要的operator
        is_math_dataset = dataset_type.upper() in ["AIME", "MATH", "GSM8K"]

        if is_math_dataset:
            # 数学任务：确保有Custom，但不要CustomCodeGenerate
            if "CustomCodeGenerate" in operators:
                operators.remove("CustomCodeGenerate")
            if "Custom" not in operators:
                operators.insert(0, "Custom")
        else:
            # 代码任务：确保有CustomCodeGenerate
            if "CustomCodeGenerate" not in operators:
                operators.insert(0, "CustomCodeGenerate")

        # 生成operator初始化代码
        operator_init = []
        operator_mapping = {
            "Custom": "self.custom = operator.Custom(self.llm)",
            "CustomCodeGenerate": "self.custom_code_generate = operator.CustomCodeGenerate(self.llm)",
            "ScEnsemble": "self.sc_ensemble = operator.ScEnsemble(self.llm)",
            "Test": "self.test = operator.Test(self.llm)",
            "Review": "self.review = operator.Review(self.llm)",
            "Revise": "self.revise = operator.Revise(self.llm)"
        }

        for op in operators:
            if op in operator_mapping:
                operator_init.append(operator_mapping[op])

        operator_init_code = "\n        ".join(operator_init)

        # 生成workflow执行代码
        workflow_logic = self._generate_workflow_logic(operators, steps, dataset_type)

        # 根据数据集类型生成不同的__call__签名
        is_code_dataset = dataset_type.upper() in ["HUMANEVAL", "MBPP", "CODEEVAL"]

        if is_code_dataset:
            # 代码生成任务：需要problem和entry_point
            call_signature = "async def __call__(self, problem: str, entry_point: str):"
            call_docstring = '        """\n        RL-generated workflow execution logic for code generation\n        """'
        else:
            # 数学/推理任务：只需要problem
            call_signature = "async def __call__(self, problem: str, entry_point: Optional[str] = None):"
            call_docstring = '        """\n        RL-generated workflow execution logic\n        Note: entry_point is optional for non-code datasets\n        """'

        # 完整的workflow类
        workflow_code = f'''{imports}

class Workflow:
    """
    RL-generated workflow for {dataset_type}

    Steps:
    {self._format_steps_as_comments(steps)}
    """

    def __init__(
        self,
        name: str,
        llm_config,
        dataset: str,  # 使用str类型支持任意数据集
    ) -> None:
        self.name = name
        self.dataset = dataset
        self.llm = create_llm_instance(llm_config)
        {operator_init_code}

    {call_signature}
{call_docstring}
{workflow_logic}
'''

        return workflow_code

    def _format_steps_as_comments(self, steps: List[str]) -> str:
        """将步骤格式化为注释"""
        return "\n    ".join([f"{i+1}. {step}" for i, step in enumerate(steps)])

    def _generate_workflow_logic(self, operators: List[str], steps: List[str], dataset_type: str = "HumanEval") -> str:
        """生成workflow执行逻辑"""
        # 根据数据集类型和operators生成执行流程
        is_code_dataset = dataset_type.upper() in ["HUMANEVAL", "MBPP", "CODEEVAL"]
        is_math_dataset = dataset_type.upper() in ["AIME", "MATH", "GSM8K"]

        if "ScEnsemble" in operators:
            # 有ensemble，需要生成多个候选
            if is_code_dataset:
                # 代码生成任务 - 使用CustomCodeGenerate
                logic = """        # Generate multiple candidate code solutions
        solutions = []
        for i in range(3):
            sol = await self.custom_code_generate(
                problem=problem,
                entry_point=entry_point,
                instruction=""
            )
            solutions.append(sol['response'])

        # Use ensemble to select best solution
        result = await self.sc_ensemble(solutions=solutions, problem=problem)
        solution = result['response']"""
            elif is_math_dataset:
                # 数学任务 - 使用Custom让LLM直接推理数学问题
                logic = """        # Generate multiple candidate math solutions using Custom
        solutions = []
        for i in range(5):
            sol = await self.custom(
                input=problem,
                instruction="Solve this AIME math problem step by step. Think carefully and provide your final answer as a number between 0 and 999."
            )
            solutions.append(sol['response'])

        # Use ensemble to select best solution
        result = await self.sc_ensemble(solutions=solutions, problem=problem)
        solution = result['response']"""
            else:
                # 其他推理任务 - 使用Custom
                logic = """        # Generate multiple candidate solutions
        solutions = []
        for i in range(3):
            sol = await self.custom(
                input=problem,
                instruction=""
            )
            solutions.append(sol['response'])

        # Use ensemble to select best solution
        result = await self.sc_ensemble(solutions=solutions, problem=problem)
        solution = result['response']"""
        else:
            # 简单流程：直接生成解决方案
            if is_code_dataset:
                # 代码生成任务 - 使用CustomCodeGenerate
                logic = """        # Generate code solution
        sol = await self.custom_code_generate(
            problem=problem,
            entry_point=entry_point,
            instruction=""
        )
        solution = sol['response']"""
            elif is_math_dataset:
                # 数学任务 - 使用Custom让LLM推理
                logic = """        # Solve math problem using Custom
        sol = await self.custom(
            input=problem,
            instruction="Solve this AIME math problem step by step. Think carefully and provide your final answer as a number between 0 and 999."
        )
        solution = sol['response']"""
            else:
                # 其他推理任务 - 使用Custom
                logic = """        # Generate solution
        sol = await self.custom(
            input=problem,
            instruction=""
        )
        solution = sol['response']"""

        # 如果有Test operator（但不执行，因为我们有独立的evaluator）
        if "Test" in operators and is_code_dataset:
            logic += """

        # Test operator available but not used (we use external evaluator)
        # test_result = self.test.exec_code(solution, entry_point)"""

        # 返回solution
        logic += """

        return solution, self.llm.get_usage_summary()["total_cost"]"""

        return logic

    def save_workflow_to_file(self, workflow_spec: WorkflowSpec, round_num: int, output_dir: str) -> str:
        """
        保存workflow代码到文件

        Args:
            workflow_spec: workflow规格
            round_num: round编号
            output_dir: 输出目录

        Returns:
            保存的文件路径
        """
        # 创建目录
        round_dir = os.path.join(output_dir, f"round_{round_num}")
        os.makedirs(round_dir, exist_ok=True)

        # 保存graph.py
        graph_path = os.path.join(round_dir, "graph.py")
        with open(graph_path, 'w') as f:
            f.write(workflow_spec.workflow_code)

        # 创建__init__.py
        init_path = os.path.join(round_dir, "__init__.py")
        with open(init_path, 'w') as f:
            f.write("")

        # 创建prompt.py（如果需要自定义prompt）
        prompt_path = os.path.join(round_dir, "prompt.py")
        if not os.path.exists(prompt_path):
            with open(prompt_path, 'w') as f:
                f.write("# Auto-generated prompts\n")

        # 保存modification记录
        modification_path = os.path.join(round_dir, "modification.txt")
        with open(modification_path, 'w') as f:
            f.write(f"Round {round_num} Modification:\n")
            f.write(f"{workflow_spec.modification}\n\n")
            f.write("Operators:\n")
            f.write(", ".join(workflow_spec.operators) + "\n\n")
            f.write("Steps:\n")
            for i, step in enumerate(workflow_spec.steps):
                f.write(f"{i+1}. {step}\n")

        print(f"[WorkflowParser] Workflow saved to {graph_path}")
        return graph_path


# 示例用法
if __name__ == "__main__":
    parser = WorkflowParser()

    # 测试Qwen输出解析
    test_output = """
<workflow_modification>
Add ensemble operator to improve solution quality by generating multiple candidates
</workflow_modification>

<operators>
CustomCodeGenerate, ScEnsemble
</operators>

<workflow_steps>
1. Use CustomCodeGenerate to generate 3 candidate solutions
2. Use ScEnsemble to select the best solution based on consistency
3. Return the selected solution
</workflow_steps>
"""

    spec = parser.parse_qwen_output(test_output)
    if spec:
        print("Parsed workflow spec:")
        print(f"Modification: {spec.modification}")
        print(f"Operators: {spec.operators}")
        print(f"Steps: {spec.steps}")
        print("\nGenerated code:")
        print(spec.workflow_code)

        # 保存到文件
        parser.save_workflow_to_file(spec, 2, "./test_workflows")
