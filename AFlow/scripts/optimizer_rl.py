"""
RL-Enhanced Optimizer for Deep Integration
RL 增强的优化器，用于深度集成

This module extends AFlow's Optimizer to incorporate RL policy guidance,
enabling bidirectional learning between MCTS and RL.

Key features:
1. RL policy participates in MCTS node selection
2. RL Q-values fused with UCB scores
3. RL suggestions guide LLM code generation
4. Shared experience pool for cross-system learning
5. WorkflowState tracking for unified representation
"""

import asyncio
import time
import sys
import os
from typing import List, Optional, Dict, Any, Tuple
import numpy as np

# Add integration directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'integration'))

from scripts.optimizer import Optimizer, GraphOptimize
from scripts.formatter import XmlFormatter, FormatError
from scripts.logs import logger

# Import unified state and shared experience from integration
try:
    from unified_state import WorkflowState, StateManager
    from AFlow.scripts.shared_experience import SharedExperiencePool, Experience
except ImportError:
    logger.warning("Could not import unified_state or shared_experience. Using fallback.")
    WorkflowState = None
    StateManager = None
    SharedExperiencePool = None
    Experience = None


class RLEnhancedOptimizer(Optimizer):
    """
    RL 增强的优化器
    Optimizer enhanced with RL policy guidance

    Extends AFlow's Optimizer to:
    1. Accept RL policy for node selection and action generation
    2. Fuse MCTS UCB scores with RL Q-values
    3. Maintain shared experience pool
    4. Track WorkflowState for unified representation
    5. Enable bidirectional learning
    """

    def __init__(
        self,
        rl_policy=None,
        use_rl_guidance: bool = True,
        rl_weight: float = 0.5,
        shared_experience_pool: Optional[Any] = None,
        state_manager: Optional[Any] = None,
        enable_state_tracking: bool = True,
        use_qwen_code_generation: bool = False,
        qwen_code_generator=None,
        qwen_max_retries: int = 2,
        **kwargs
    ):
        """
        Initialize RL-enhanced optimizer

        Args:
            rl_policy: RL policy object (has methods: get_q_value, suggest_action)
            use_rl_guidance: Whether to use RL guidance in selection
            rl_weight: Weight for RL Q-value in combined score (0.0-1.0)
                      combined_score = (1-w)*ucb + w*q_value
            shared_experience_pool: Shared experience pool instance
            state_manager: State manager instance
            enable_state_tracking: Whether to track WorkflowState objects
            use_qwen_code_generation: Whether to use Qwen to generate code directly (MCTS + Qwen)
            qwen_code_generator: Qwen policy instance for code generation
            qwen_max_retries: Maximum retries for Qwen code generation when syntax errors occur
            **kwargs: Arguments passed to base Optimizer
        """
        super().__init__(**kwargs)

        # RL components
        self.rl_policy = rl_policy
        self.use_rl_guidance = use_rl_guidance
        self.rl_weight = rl_weight

        # Qwen direct code generation (MCTS + Qwen生成代码)
        self.use_qwen_code_generation = use_qwen_code_generation
        self.qwen_code_generator = qwen_code_generator or rl_policy
        self.qwen_max_retries = qwen_max_retries

        # Shared components
        if shared_experience_pool is None and SharedExperiencePool is not None:
            self.shared_experience_pool = SharedExperiencePool(max_size=10000)
        else:
            self.shared_experience_pool = shared_experience_pool

        if state_manager is None and StateManager is not None:
            self.state_manager = StateManager()
        else:
            self.state_manager = state_manager

        self.enable_state_tracking = enable_state_tracking

        # Mapping between nodes and states
        self.node_to_state_mapping: Dict[int, str] = {}  # round -> state_id
        self.round_to_parent: Dict[int, int] = {}  # round -> parent_round

        # RL trajectory tracking
        self.current_trajectory_id = None
        self.trajectory_step_index = 0
        self.rl_trajectory: List[Dict[str, Any]] = []

        # Statistics
        self.rl_stats = {
            "total_rl_selections": 0,
            "total_ucb_selections": 0,
            "avg_q_value": 0.0,
            "avg_ucb_score": 0.0,
            "avg_combined_score": 0.0
        }

    def _generate_baseline_workflow(self, directory: str, round_number: int):
        """
        Generate baseline workflow for MCTS initialization
        自动生成MCTS的baseline workflow（第一轮）

        Args:
            directory: Round directory path
            round_number: Round number (should be 1)
        """
        import os

        logger.info(f"[MCTS] Creating baseline workflow for round {round_number}...")

        # Create baseline graph using Custom operator only
        baseline_response = {
            "graph": """class Workflow:
    def __init__(self, name: str, llm_config, dataset: str) -> None:
        '''Baseline workflow - MCTS starting point'''
        self.name = name
        self.dataset = dataset
        from scripts.async_llm import create_llm_instance
        from scripts.operators import Custom
        self.llm = create_llm_instance(llm_config)
        self.custom = Custom(self.llm)

    async def __call__(self, problem: str):
        '''Simple baseline: direct LLM solution with Custom operator'''
        solution = await self.custom(
            input=problem,
            instruction="Solve this problem step by step. Provide your final answer clearly."
        )
        # Extract text from dict response and return with cost 0
        response_text = solution.get("response", str(solution)) if isinstance(solution, dict) else str(solution)
        return response_text, 0.0""",

            "prompt": """# Baseline prompts for initial workflow

SOLVE_PROMPT = \"\"\"Solve this problem step by step. Provide your final answer clearly.\"\"\"
""",

            "modification": "Initial baseline workflow using Custom operator"
        }

        # Write baseline graph and prompt files directly (bypass WORKFLOW_TEMPLATE)
        graph_code = baseline_response["graph"]
        prompt_code = baseline_response["prompt"]

        # Write graph.py
        graph_file = os.path.join(directory, "graph.py")
        with open(graph_file, "w", encoding="utf-8") as f:
            f.write("from typing import Literal\n")
            f.write("from scripts.async_llm import create_llm_instance\n")
            f.write("from scripts.evaluator import DatasetType\n\n")
            f.write(graph_code)

        # Write prompt.py
        prompt_file = os.path.join(directory, "prompt.py")
        with open(prompt_file, "w", encoding="utf-8") as f:
            f.write(prompt_code)

        # Write __init__.py
        init_file = os.path.join(directory, "__init__.py")
        with open(init_file, "w", encoding="utf-8") as f:
            f.write("# Auto-generated baseline workflow\n")

        logger.info(f"[MCTS] ✅ Baseline workflow created at {directory}")
        logger.info(f"[MCTS] MCTS tree search will optimize from this baseline")

    async def _optimize_graph(self):
        """
        Override base method to incorporate RL guidance
        重写基类方法以整合 RL 指导
        """
        validation_n = self.validation_rounds
        graph_path = f"{self.root_path}/workflows"
        data = self.data_utils.load_results(graph_path)

        if self.round == 1:
            # Initial round - create baseline workflow if not exists
            directory = self.graph_utils.create_round_directory(graph_path, self.round)

            # Check if baseline workflow exists, if not, generate it
            graph_file = os.path.join(graph_path, f"round_{self.round}", "graph.py")
            if not os.path.exists(graph_file):
                logger.info(f"[MCTS Init] Generating baseline workflow for round {self.round}...")
                self._generate_baseline_workflow(directory, self.round)

            self.graph = self.graph_utils.load_graph(self.round, graph_path)
            avg_score = await self.evaluation_utils.evaluate_graph(
                self, directory, validation_n, data, initial=True
            )

            # Create initial state
            if self.enable_state_tracking and WorkflowState is not None:
                initial_state = await self._create_workflow_state(
                    round_number=self.round,
                    score=avg_score,
                    parent_round=None,
                    graph_path=graph_path
                )
                self.state_manager.add_state(initial_state)
                self.node_to_state_mapping[self.round] = initial_state.mcts_node_id

            return avg_score

        # RL-enhanced optimization loop
        while True:
            directory = self.graph_utils.create_round_directory(graph_path, self.round + 1)

            # Top rounds selection with RL guidance
            top_rounds = self.data_utils.get_top_rounds(self.sample)

            if self.use_rl_guidance and self.rl_policy is not None:
                # RL-guided selection: fuse UCB with Q-value
                sample = await self._rl_guided_selection(top_rounds)
                self.rl_stats["total_rl_selections"] += 1
            else:
                # Standard selection
                sample = self.data_utils.select_round(top_rounds)
                self.rl_stats["total_ucb_selections"] += 1

            # Load parent workflow
            prompt, graph_load = self.graph_utils.read_graph_files(sample["round"], graph_path)
            graph = self.graph_utils.extract_solve_graph(graph_load)

            processed_experience = self.experience_utils.load_experience()
            experience = self.experience_utils.format_experience(processed_experience, sample["round"])

            operator_description = self.graph_utils.load_operators_description(self.operators)
            log_data = self.data_utils.load_log(sample["round"])

            # Generate new workflow with RL guidance
            if self.use_rl_guidance and self.rl_policy is not None:
                response = await self._generate_with_rl_guidance(
                    experience, sample, graph[0], prompt, operator_description, log_data
                )
            else:
                # Standard generation
                graph_optimize_prompt = self.graph_utils.create_graph_optimize_prompt(
                    experience, sample["score"], graph[0], prompt, operator_description,
                    self.type, log_data
                )
                response = await self._generate_graph(graph_optimize_prompt)

            # Check modification validity
            check = self.experience_utils.check_modification(
                processed_experience, response["modification"], sample["round"]
            )

            if check:
                # Record parent relationship
                self.round_to_parent[self.round + 1] = sample["round"]
                break

        # Save and evaluate
        self.graph_utils.write_graph_files(directory, response, self.round + 1, self.dataset)

        experience_data = self.experience_utils.create_experience_data(
            sample, response["modification"]
        )

        self.graph = self.graph_utils.load_graph(self.round + 1, graph_path)

        logger.info(directory)

        avg_score = await self.evaluation_utils.evaluate_graph(
            self, directory, validation_n, data, initial=False
        )

        self.experience_utils.update_experience(directory, experience_data, avg_score)

        # Update shared experience pool
        if self.shared_experience_pool is not None and Experience is not None:
            await self._update_shared_experience(
                sample, response, avg_score, graph_path
            )

        # Create and track workflow state
        if self.enable_state_tracking and WorkflowState is not None:
            parent_round = sample["round"]
            parent_state_id = self.node_to_state_mapping.get(parent_round)

            new_state = await self._create_workflow_state(
                round_number=self.round + 1,
                score=avg_score,
                parent_round=parent_round,
                graph_path=graph_path,
                parent_state_id=parent_state_id,
                parent_score=sample["score"]
            )

            self.state_manager.add_state(new_state)
            self.node_to_state_mapping[self.round + 1] = new_state.mcts_node_id

            # Update RL estimates if policy available
            if self.rl_policy is not None:
                await self._update_rl_estimates(new_state)

        # Record trajectory step
        self._record_trajectory_step(sample, response, avg_score)

        return avg_score

    async def _rl_guided_selection(self, top_rounds: List[Dict]) -> Dict:
        """
        Select parent node by fusing UCB score with RL Q-value
        通过融合 UCB 分数和 RL Q 值来选择父节点

        Args:
            top_rounds: List of top-performing rounds with their scores

        Returns:
            Dict: Selected round with score and round number
        """
        if len(top_rounds) == 0:
            raise ValueError("No top rounds available for selection")

        if len(top_rounds) == 1:
            return top_rounds[0]

        # Compute UCB scores for each round
        scores_with_q = []

        for round_data in top_rounds:
            round_num = round_data["round"]
            score = round_data["score"]

            # Compute UCB score (simplified version)
            # In full MCTS, this would use visit counts and exploration constant
            ucb_score = score  # Simplified: just use score as UCB

            # Get RL Q-value estimate
            q_value = 0.0
            if self.rl_policy is not None:
                try:
                    # Get state representation
                    state_id = self.node_to_state_mapping.get(round_num)
                    if state_id and self.state_manager:
                        state = self.state_manager.get_state(state_id)
                        if state:
                            # Ask RL policy for Q-value
                            q_value = await self._get_q_value_from_policy(state)
                except Exception as e:
                    logger.warning(f"Error getting Q-value from RL policy: {e}")
                    q_value = 0.0

            # Combine UCB and Q-value
            combined_score = (1 - self.rl_weight) * ucb_score + self.rl_weight * q_value

            scores_with_q.append({
                "round_data": round_data,
                "ucb_score": ucb_score,
                "q_value": q_value,
                "combined_score": combined_score
            })

            # Update statistics
            self.rl_stats["avg_ucb_score"] = (
                (self.rl_stats["avg_ucb_score"] * self.rl_stats["total_rl_selections"] + ucb_score)
                / (self.rl_stats["total_rl_selections"] + 1)
            )
            self.rl_stats["avg_q_value"] = (
                (self.rl_stats["avg_q_value"] * self.rl_stats["total_rl_selections"] + q_value)
                / (self.rl_stats["total_rl_selections"] + 1)
            )
            self.rl_stats["avg_combined_score"] = (
                (self.rl_stats["avg_combined_score"] * self.rl_stats["total_rl_selections"] + combined_score)
                / (self.rl_stats["total_rl_selections"] + 1)
            )

        # Select round with highest combined score
        best = max(scores_with_q, key=lambda x: x["combined_score"])

        logger.info(
            f"RL-guided selection: round {best['round_data']['round']}, "
            f"UCB={best['ucb_score']:.4f}, Q={best['q_value']:.4f}, "
            f"Combined={best['combined_score']:.4f}"
        )

        return best["round_data"]

    async def _generate_with_rl_guidance(
        self,
        experience: str,
        sample: Dict,
        graph: str,
        prompt: str,
        operator_description: str,
        log_data: str
    ) -> Dict[str, str]:
        """
        Generate new workflow with RL policy suggestions
        使用 RL 策略建议生成新的工作流

        MCTS+Qwen实现：如果启用use_qwen_code_generation，则让Qwen直接生成完整代码

        Args:
            experience: Formatted experience string
            sample: Selected parent round data
            graph: Parent graph code
            prompt: Parent prompt code
            operator_description: Available operators description
            log_data: Execution logs

        Returns:
            Dict: Response with modification, graph, and prompt
        """
        # MCTS + Qwen: 使用Qwen直接生成代码（无GPT-4 fallback）
        if self.use_qwen_code_generation and self.qwen_code_generator is not None:
            logger.info("[RLEnhancedOptimizer] 🎯 MCTS + Qwen: Using Qwen to generate code directly (no GPT-4 fallback)")

            # 尝试使用Qwen生成代码
            try:
                qwen_response = await self._generate_code_with_qwen(
                    experience, sample, graph, prompt, operator_description, log_data,
                    max_retries=self.qwen_max_retries
                )

                if qwen_response is not None:
                    logger.info("[RLEnhancedOptimizer] ✅ Qwen code generation successful")
                    return qwen_response
                else:
                    # ✅ 不fallback到GPT-4 - 返回失败workflow让Qwen通过负奖励学习
                    logger.warning("[RLEnhancedOptimizer] ⚠️ Qwen failed to generate valid code after retries")
                    logger.info("[RLEnhancedOptimizer] 📚 Returning empty workflow for negative reward signal")
                    logger.info("[RLEnhancedOptimizer] 🎓 Qwen will learn from this failure through PPO")

                    return {
                        'modification': 'Failed to generate valid code - syntax errors after retries',
                        'graph': '''class Workflow:
    def __init__(self, name: str, llm_config, dataset: str) -> None:
        self.name = name
        self.dataset = dataset
        from scripts.async_llm import create_llm_instance
        self.llm = create_llm_instance(llm_config)

    async def __call__(self, problem: str, entry_point=None):
        # Empty workflow - failed code generation
        # Will result in zero score and negative reward
        return "", 0.0''',
                        'prompt': '# No custom prompts - failed generation'
                    }

            except Exception as e:
                logger.error(f"[RLEnhancedOptimizer] ❌ Exception in Qwen code generation: {e}")
                logger.info("[RLEnhancedOptimizer] 📚 Returning empty workflow for negative reward signal")
                logger.info("[RLEnhancedOptimizer] 🎓 Qwen will learn to avoid this error")

                return {
                    'modification': f'Code generation error: {str(e)[:200]}',
                    'graph': '''class Workflow:
    def __init__(self, name: str, llm_config, dataset: str) -> None:
        self.name = name
        self.dataset = dataset
        from scripts.async_llm import create_llm_instance
        self.llm = create_llm_instance(llm_config)

    async def __call__(self, problem: str, entry_point=None):
        # Empty workflow - exception during generation
        return "", 0.0''',
                    'prompt': '# No custom prompts - exception occurred'
                }

        # 原版流程：使用GPT-4生成代码（带或不带RL suggestion）
        # Get RL suggestion if available
        rl_suggestion = ""
        if self.rl_policy is not None:
            try:
                state_id = self.node_to_state_mapping.get(sample["round"])
                if state_id and self.state_manager:
                    state = self.state_manager.get_state(state_id)
                    if state:
                        rl_suggestion = await self._get_action_suggestion_from_policy(state)
            except Exception as e:
                logger.warning(f"Error getting RL suggestion: {e}")
                rl_suggestion = ""

        # Create enhanced prompt with RL suggestion
        base_prompt = self.graph_utils.create_graph_optimize_prompt(
            experience, sample["score"], graph, prompt, operator_description,
            self.type, log_data
        )

        if rl_suggestion:
            enhanced_prompt = f"{base_prompt}\n\n## RL Policy Suggestion\n{rl_suggestion}"
            logger.info(f"Using RL suggestion: {rl_suggestion}")
        else:
            enhanced_prompt = base_prompt

        # Generate graph with GPT-4
        return await self._generate_graph(enhanced_prompt)

    async def _generate_graph(self, graph_optimize_prompt: str) -> Dict[str, str]:
        """
        Generate graph using LLM with formatter
        使用 LLM 和格式化器生成图

        Args:
            graph_optimize_prompt: Optimization prompt

        Returns:
            Dict: Response with modification, graph, and prompt
        """
        try:
            graph_formatter = XmlFormatter.from_model(GraphOptimize)

            response = await self.optimize_llm.call_with_format(
                graph_optimize_prompt,
                graph_formatter
            )

            logger.info("Graph optimization response received successfully")
            return response

        except FormatError as e:
            logger.error(f"Format error in graph optimization: {str(e)}")

            # Fallback: direct call with post-processing
            raw_response = await self.optimize_llm(graph_optimize_prompt)
            response = self._extract_fields_from_response(raw_response)

            if not response:
                logger.error("Failed to extract fields from raw response")
                # Return empty response as last resort
                return {
                    "modification": "Failed to generate modification",
                    "graph": graph_optimize_prompt,  # Keep original
                    "prompt": ""
                }

            return response

    async def _get_q_value_from_policy(self, state: 'WorkflowState') -> float:
        """
        Get Q-value estimate from RL policy
        从 RL 策略获取 Q 值估计

        Args:
            state: WorkflowState object

        Returns:
            float: Q-value estimate
        """
        if not hasattr(self.rl_policy, 'get_q_value'):
            return 0.0

        try:
            # Convert state to policy input format
            state_repr = state.to_text_representation()

            # Get Q-value from policy
            q_value = self.rl_policy.get_q_value(state_repr)

            return float(q_value)

        except Exception as e:
            logger.warning(f"Error getting Q-value: {e}")
            return 0.0

    async def _get_action_suggestion_from_policy(self, state: 'WorkflowState') -> str:
        """
        Get action suggestion from RL policy
        从 RL 策略获取动作建议

        Args:
            state: WorkflowState object

        Returns:
            str: Action suggestion text
        """
        if not hasattr(self.rl_policy, 'suggest_action'):
            return ""

        try:
            # Convert state to policy input format
            state_repr = state.to_text_representation()

            # Get suggestion from policy
            suggestion = self.rl_policy.suggest_action(state_repr)

            return str(suggestion)

        except Exception as e:
            logger.warning(f"Error getting action suggestion: {e}")
            return ""

    async def _create_workflow_state(
        self,
        round_number: int,
        score: float,
        parent_round: Optional[int],
        graph_path: str,
        parent_state_id: Optional[str] = None,
        parent_score: float = 0.0
    ) -> 'WorkflowState':
        """
        Create WorkflowState from current round
        从当前轮次创建 WorkflowState

        Args:
            round_number: Current round number
            score: Score achieved
            parent_round: Parent round number (None for initial round)
            graph_path: Path to workflows directory
            parent_state_id: Parent state ID (if available)
            parent_score: Parent score

        Returns:
            WorkflowState: Created state object
        """
        # Load graph and prompt code
        try:
            prompt_code, graph_code = self.graph_utils.read_graph_files(round_number, graph_path)
        except:
            prompt_code = ""
            graph_code = ""

        # Extract operators from graph code (simplified)
        operators = self.operators.copy() if self.operators else []

        # Create state
        state = WorkflowState(
            mcts_node_id=None,  # Will be auto-generated
            parent_node_id=parent_state_id,
            round_number=round_number,
            graph_code=graph_code,
            prompt_code=prompt_code,
            operators=operators,
            score=score,
            dataset=str(self.dataset),
            parent_score=parent_score,
            timestamp=time.time()
        )

        return state

    async def _update_rl_estimates(self, state: 'WorkflowState'):
        """
        Update RL estimates for a state
        更新状态的 RL 估计值

        Args:
            state: WorkflowState to update
        """
        if self.rl_policy is None:
            return

        try:
            # Get Q-value
            q_value = await self._get_q_value_from_policy(state)

            # Get value estimate if available
            value_estimate = 0.0
            if hasattr(self.rl_policy, 'get_value'):
                state_repr = state.to_text_representation()
                value_estimate = self.rl_policy.get_value(state_repr)

            # Update state
            state.update_rl_estimates(q_value=q_value, value_estimate=value_estimate)

        except Exception as e:
            logger.warning(f"Error updating RL estimates: {e}")

    async def _update_shared_experience(
        self,
        sample: Dict,
        response: Dict,
        score: float,
        graph_path: str
    ):
        """
        Update shared experience pool with new workflow
        用新工作流更新共享经验池

        Args:
            sample: Parent round data
            response: Generated response
            score: Achieved score
            graph_path: Path to workflows directory
        """
        try:
            # Create experience entry
            experience = Experience(
                graph_code=response["graph"],
                prompt_code=response["prompt"],
                operators=self.operators.copy() if self.operators else [],
                operator_sequence=[],  # Could be extracted from execution
                score=score,
                parent_score=sample["score"],
                improvement=score - sample["score"],
                round_number=self.round + 1,
                dataset=str(self.dataset),
                parent_node_id=str(sample["round"]),
                node_id=str(self.round + 1),
                timestamp=time.time()
            )

            # Add to pool
            self.shared_experience_pool.add(experience)

            logger.info(
                f"Added experience to shared pool: round={self.round + 1}, "
                f"score={score:.4f}, improvement={experience.improvement:.4f}"
            )

        except Exception as e:
            logger.error(f"Error updating shared experience: {e}")

    def _record_trajectory_step(self, sample: Dict, response: Dict, score: float):
        """
        Record a step in the RL trajectory
        记录 RL 轨迹中的一步

        Args:
            sample: Parent round data
            response: Generated response
            score: Achieved score
        """
        step = {
            "step_index": self.trajectory_step_index,
            "parent_round": sample["round"],
            "current_round": self.round + 1,
            "parent_score": sample["score"],
            "current_score": score,
            "improvement": score - sample["score"],
            "modification": response.get("modification", ""),
            "timestamp": time.time()
        }

        self.rl_trajectory.append(step)
        self.trajectory_step_index += 1

    def get_rl_statistics(self) -> Dict[str, Any]:
        """
        Get RL-related statistics
        获取 RL 相关统计信息

        Returns:
            Dict: Statistics dictionary
        """
        stats = self.rl_stats.copy()
        stats["trajectory_length"] = len(self.rl_trajectory)
        stats["total_states"] = len(self.state_manager) if self.state_manager else 0

        if self.shared_experience_pool:
            stats["shared_pool_size"] = len(self.shared_experience_pool)
            stats["shared_pool_stats"] = self.shared_experience_pool.get_statistics()

        return stats

    def set_rl_policy(self, rl_policy):
        """
        Set or update RL policy
        设置或更新 RL 策略

        Args:
            rl_policy: New RL policy object
        """
        self.rl_policy = rl_policy
        logger.info("RL policy updated")

    def set_rl_weight(self, weight: float):
        """
        Set weight for RL Q-value in combined score
        设置 RL Q 值在组合分数中的权重

        Args:
            weight: Weight value (0.0-1.0)
        """
        if not 0.0 <= weight <= 1.0:
            raise ValueError("RL weight must be between 0.0 and 1.0")

        self.rl_weight = weight
        logger.info(f"RL weight updated to {weight}")

    def enable_rl_guidance(self, enabled: bool = True):
        """
        Enable or disable RL guidance
        启用或禁用 RL 指导

        Args:
            enabled: Whether to enable RL guidance
        """
        self.use_rl_guidance = enabled
        logger.info(f"RL guidance {'enabled' if enabled else 'disabled'}")

    def _extract_code_from_qwen(self, qwen_output: str) -> Optional[Dict[str, str]]:
        """
        从Qwen输出提取代码 - 完全对齐原版AFlow

        原版AFlow期望LLM返回：
        <modification>...</modification>
        <graph>...</graph>
        <prompt>...</prompt>

        Args:
            qwen_output: Qwen生成的输出

        Returns:
            {'graph': str, 'modification': str, 'prompt': str} 或 None
        """
        import re

        result = {}

        # 提取 modification
        modification_pattern = r"<modification>(.*?)</modification>"
        modification_match = re.search(modification_pattern, qwen_output, re.DOTALL)
        if modification_match:
            result['modification'] = modification_match.group(1).strip()
        else:
            result['modification'] = "No modification description provided"

        # 提取 graph (必需)
        graph_pattern = r"<graph>(.*?)</graph>"
        graph_match = re.search(graph_pattern, qwen_output, re.DOTALL)
        if not graph_match:
            logger.error("[RLEnhancedOptimizer] No <graph> tag found in Qwen output")
            return None

        result['graph'] = graph_match.group(1).strip()

        # 提取 prompt (可选)
        prompt_pattern = r"<prompt>(.*?)</prompt>"
        prompt_match = re.search(prompt_pattern, qwen_output, re.DOTALL)
        if prompt_match:
            result['prompt'] = prompt_match.group(1).strip()
        else:
            result['prompt'] = "# Auto-generated - no custom prompts needed\n"

        return result

    def _validate_python_syntax(self, code: str) -> bool:
        """
        验证Python代码语法

        Args:
            code: Python代码字符串

        Returns:
            bool: 语法是否正确
        """
        try:
            compile(code, '<string>', 'exec')
            return True
        except SyntaxError as e:
            logger.error(f"[RLEnhancedOptimizer] Syntax error in code: {e}")
            logger.error(f"[RLEnhancedOptimizer] Error line: {e.lineno}, offset: {e.offset}")
            logger.error(f"[RLEnhancedOptimizer] Error text: {e.text}")
            return False
        except Exception as e:
            logger.error(f"[RLEnhancedOptimizer] Unexpected error during syntax check: {e}")
            return False

    async def _generate_code_with_qwen(
        self,
        experience: str,
        sample: Dict,
        graph: str,
        prompt: str,
        operator_description: str,
        log_data: str,
        max_retries: int = 2
    ) -> Optional[Dict[str, str]]:
        """
        使用Qwen直接生成完整workflow代码 (MCTS + Qwen的核心方法)

        完全对齐原版AFlow设计：
        1. Qwen生成完整Python代码（不是建议）
        2. 代码包含在<graph>标签中
        3. 验证语法
        4. 返回与GPT-4相同格式的response

        Args:
            experience: 经验池字符串
            sample: 选中的父节点数据
            graph: 父workflow的graph代码
            prompt: 父workflow的prompt代码
            operator_description: 可用operators描述
            log_data: 执行日志
            max_retries: 最大重试次数（语法错误时）

        Returns:
            Dict[str, str]: {'modification': str, 'graph': str, 'prompt': str} 或 None
        """
        # 构建observation（类似deep_workflow_env的format_observation）
        observation = self._build_observation_for_qwen(
            experience, sample, graph, prompt, operator_description, log_data
        )

        logger.info(f"[RLEnhancedOptimizer] Generating code with Qwen (max_retries={max_retries})")
        logger.info(f"[RLEnhancedOptimizer] Parent round: {sample['round']}, Parent score: {sample['score']:.4f}")

        # 尝试生成代码（带重试机制）
        for attempt in range(max_retries):
            try:
                # 调用Qwen生成代码
                qwen_output = await self._call_qwen_generator(observation)

                if not qwen_output:
                    logger.warning(f"[RLEnhancedOptimizer] Attempt {attempt+1}/{max_retries}: Empty output from Qwen")
                    continue

                logger.info(f"[RLEnhancedOptimizer] Attempt {attempt+1}/{max_retries}: Received {len(qwen_output)} chars from Qwen")

                # 提取代码
                extraction_result = self._extract_code_from_qwen(qwen_output)

                if extraction_result is None:
                    logger.warning(f"[RLEnhancedOptimizer] Attempt {attempt+1}/{max_retries}: Failed to extract <graph> tag")
                    continue

                graph_code = extraction_result['graph']
                modification = extraction_result['modification']
                prompt_code = extraction_result['prompt']

                logger.info(f"[RLEnhancedOptimizer] Extracted code: {len(graph_code)} chars")
                logger.info(f"[RLEnhancedOptimizer] Modification: {modification[:100]}...")

                # 验证语法
                if not self._validate_python_syntax(graph_code):
                    logger.warning(f"[RLEnhancedOptimizer] Attempt {attempt+1}/{max_retries}: Syntax validation failed")
                    continue

                # 成功！返回response
                logger.info(f"[RLEnhancedOptimizer] ✅ Qwen generated valid code on attempt {attempt+1}")

                return {
                    'modification': modification,
                    'graph': graph_code,
                    'prompt': prompt_code
                }

            except Exception as e:
                logger.error(f"[RLEnhancedOptimizer] Attempt {attempt+1}/{max_retries}: Error: {e}")
                if attempt < max_retries - 1:
                    logger.info(f"[RLEnhancedOptimizer] Retrying...")
                continue

        # 所有尝试都失败
        logger.error(f"[RLEnhancedOptimizer] ❌ Failed to generate valid code after {max_retries} attempts")
        return None

    def _build_observation_for_qwen(
        self,
        experience: str,
        sample: Dict,
        graph: str,
        prompt: str,
        operator_description: str,
        log_data: str
    ) -> str:
        """
        为Qwen构建observation

        类似于workflow_code_prompt_manager的format_observation，
        但基于MCTS的上下文（父节点、经验池等）

        Args:
            experience: 经验池字符串
            sample: 父节点数据
            graph: 父graph代码
            prompt: 父prompt代码
            operator_description: 可用operators
            log_data: 执行日志

        Returns:
            str: 格式化的observation
        """
        parent_round = sample['round']
        parent_score = sample['score']

        obs = f"""## Workflow Optimization Task - Dynamic Mode with MCTS

Dataset: {self.dataset}
Current Round: {self.round + 1}
Parent Round: {parent_round}
Parent Score: {parent_score:.4f}

## Your Task:
Design a NEW workflow that improves upon the parent workflow.
Your workflow will be executed on real test cases and scored.

## Parent Workflow (Round {parent_round}):

### Parent Graph Code:
```python
{graph[:500]}...
```

### Parent Prompt Code:
```python
{prompt[:200]}...
```

## Available Operators:
{operator_description}

## Experience from Previous Attempts:
{experience[:1000]}...

## Execution Logs (if available):
{log_data[:500] if log_data else 'No logs available'}

## Instructions:
1. Analyze the parent workflow and identify areas for improvement
2. Generate a COMPLETE Python workflow using available operators
3. Output your code in the required XML format:

<modification>
Brief description of your changes and why they should improve performance.
Example: "Increase ensemble size from 5 to 15 samples to improve accuracy on hard problems"
</modification>

<graph>
class Workflow:
    def __init__(self, name: str, llm_config, dataset: str) -> None:
        self.name = name
        self.dataset = dataset
        self.llm = create_llm_instance(llm_config)

        # Initialize operators you need
        self.custom = operator.Custom(self.llm)
        self.sc_ensemble = operator.ScEnsemble(self.llm)
        # Add more as needed

    async def __call__(self, problem: str, entry_point: Optional[str] = None):
        # YOUR COMPLETE WORKFLOW LOGIC HERE
        # This code will be executed directly!

        # MUST return (solution, cost) tuple
        return solution, 0.0
</graph>

<prompt>
# Custom prompts if needed (optional)
</prompt>

## Critical Requirements:
- Code MUST be syntactically correct Python
- MUST return (solution, cost) tuple from __call__
- MUST use async/await for operator calls
- Focus on improving the parent workflow's weaknesses
- Balance effectiveness vs computational cost

GO:
"""
        return obs

    async def _call_qwen_generator(self, observation: str) -> str:
        """
        调用Qwen generator生成代码

        支持多种接口：
        1. get_action_and_value(obs, max_new_tokens, temperature) - VERL style
        2. generate(prompt) - simple style
        3. __call__(prompt) - callable style

        Args:
            observation: 输入observation

        Returns:
            str: Qwen生成的输出
        """
        # 尝试VERL-style接口 (TrainableQwenPolicy)
        if hasattr(self.qwen_code_generator, 'get_action_and_value'):
            try:
                action, _, _, _ = self.qwen_code_generator.get_action_and_value(
                    obs=observation,
                    max_new_tokens=800,  # 足够生成完整代码
                    temperature=0.7
                )
                return action
            except Exception as e:
                logger.warning(f"[RLEnhancedOptimizer] Error calling get_action_and_value: {e}")

        # 尝试generate方法
        if hasattr(self.qwen_code_generator, 'generate'):
            try:
                result = self.qwen_code_generator.generate(observation)
                if asyncio.iscoroutine(result):
                    result = await result
                return str(result)
            except Exception as e:
                logger.warning(f"[RLEnhancedOptimizer] Error calling generate: {e}")

        # 尝试callable接口
        if callable(self.qwen_code_generator):
            try:
                result = self.qwen_code_generator(observation)
                if asyncio.iscoroutine(result):
                    result = await result
                return str(result)
            except Exception as e:
                logger.warning(f"[RLEnhancedOptimizer] Error calling generator: {e}")

        # 无法调用
        raise ValueError(
            "qwen_code_generator does not have a supported interface. "
            "Expected: get_action_and_value(), generate(), or __call__()"
        )
