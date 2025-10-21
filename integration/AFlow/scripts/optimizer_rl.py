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
            **kwargs: Arguments passed to base Optimizer
        """
        super().__init__(**kwargs)

        # RL components
        self.rl_policy = rl_policy
        self.use_rl_guidance = use_rl_guidance
        self.rl_weight = rl_weight

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

    async def _optimize_graph(self):
        """
        Override base method to incorporate RL guidance
        重写基类方法以整合 RL 指导
        """
        validation_n = self.validation_rounds
        graph_path = f"{self.root_path}/workflows"
        data = self.data_utils.load_results(graph_path)

        if self.round == 1:
            # Initial round - same as base class
            directory = self.graph_utils.create_round_directory(graph_path, self.round)
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

        # Generate graph
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
