"""
AIME Workflow Prompt Manager
Generates prompts for RL-based workflow optimization on AIME problems
"""

from typing import List, Dict, Any


class AIMEWorkflowPromptManager:
    """
    Manages prompts for AIME mathematical problem-solving workflows
    """

    def __init__(self, dataset_name: str = "AIME"):
        self.dataset_name = dataset_name

    def get_optimization_prompt(
        self,
        current_workflow_desc: str,
        performance_history: List[Dict[str, float]]
    ) -> str:
        """
        Generate prompt for workflow optimization

        Args:
            current_workflow_desc: Current workflow description
            performance_history: History of Pass@K scores

        Returns:
            Optimization prompt
        """
        # Calculate performance stats
        if performance_history:
            latest_score = performance_history[-1]['pass_at_k']
            avg_score = sum(h['pass_at_k'] for h in performance_history) / len(performance_history)
            best_score = max(h['pass_at_k'] for h in performance_history)
        else:
            latest_score = avg_score = best_score = 0.0

        prompt = f"""You are optimizing a workflow for solving AIME (American Invitational Mathematics Examination) problems.

## Current Workflow
{current_workflow_desc}

## Performance
- Latest Pass@K: {latest_score:.2%}
- Average Pass@K: {avg_score:.2%}
- Best Pass@K: {best_score:.2%}

## Available Operators for AIME
1. **MathSolver**: Solves mathematical problems using step-by-step reasoning
2. **WebSearch**: Searches for relevant mathematical formulas, theorems, and methods
3. **FormulaExtract**: Extracts and applies relevant formulas from problem text
4. **StepByStepSolver**: Breaks down complex problems into manageable steps
5. **AnswerExtract**: Extracts numerical answer from solution text
6. **ScEnsemble**: Generates multiple solutions and selects the best through voting

## AIME Problem Characteristics
- High-difficulty mathematics competition problems
- Answers are integers from 0 to 999
- Require deep mathematical knowledge (algebra, geometry, number theory, combinatorics)
- Often need multiple solution steps
- May benefit from web search for obscure theorems or formulas

## Optimization Guidelines
1. **Multi-step approach**: AIME problems typically require 3-5 solution steps
2. **Knowledge augmentation**: Use WebSearch when problem involves specialized knowledge
3. **Ensemble methods**: Generate multiple solution attempts (n=3-5) and use ScEnsemble
4. **Error recovery**: Include fallback strategies if initial approach fails
5. **Answer validation**: Always extract and validate the numerical answer

## Your Task
Design an improved workflow that:
- Increases Pass@K accuracy
- Uses appropriate operators for mathematical reasoning
- Incorporates web search strategically
- Ensures robust answer extraction

Provide the new workflow as a clear, step-by-step description."""

        return prompt

    def get_execution_prompt(self, problem: str) -> str:
        """
        Generate prompt for workflow execution

        Args:
            problem: AIME problem statement

        Returns:
            Execution prompt
        """
        prompt = f"""Solve this AIME mathematics problem.

## Problem
{problem}

## Instructions
1. Read the problem carefully
2. Identify what mathematical concepts are involved
3. If you need information about specific theorems or formulas, describe what you need to search for
4. Solve the problem step by step
5. Your final answer should be an integer from 0 to 999
6. Present your answer in the format: "The answer is [number]" or "\\\\boxed{{number}}"

Remember: AIME problems require careful reasoning. Take your time to work through each step."""

        return prompt

    def get_search_query_prompt(self, problem: str, context: str) -> str:
        """
        Generate prompt for formulating search queries

        Args:
            problem: AIME problem statement
            context: Current solution context

        Returns:
            Search query generation prompt
        """
        prompt = f"""Based on this AIME problem and current solution context, formulate a concise web search query.

## Problem
{problem}

## Current Context
{context}

## Task
Generate a search query (max 10 words) to find:
- Relevant mathematical theorems
- Formula derivations
- Solution techniques

Output only the search query, nothing else."""

        return prompt


# For compatibility
WorkflowPromptManager = AIMEWorkflowPromptManager
