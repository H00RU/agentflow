"""
Unified State Representation for Deep Integration
统一状态表示，用于 AFlow 和 verl-agent 的深度集成

This module provides a unified state representation that bridges AFlow's MCTS-based
workflow optimization with verl-agent's RL training framework.
"""

import hashlib
import json
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any, Tuple
import numpy as np


@dataclass
class WorkflowState:
    """
    统一的 workflow 状态表示
    Unified workflow state representation shared between AFlow and verl-agent

    This class contains:
    1. AFlow MCTS attributes (node_id, visit_count, ucb_score)
    2. RL policy attributes (q_value, policy_logits, value_estimate)
    3. Shared workflow attributes (graph_code, operators, score)
    """

    # ===== AFlow MCTS Attributes =====
    mcts_node_id: Optional[str] = None  # Unique identifier for MCTS node
    parent_node_id: Optional[str] = None  # Parent node in MCTS tree
    visit_count: int = 0  # Number of times this node has been visited
    ucb_score: float = 0.0  # Upper Confidence Bound score for MCTS selection
    round_number: int = 0  # AFlow optimization round number

    # ===== RL Policy Attributes =====
    q_value: float = 0.0  # Q-value estimate from RL policy
    policy_logits: Optional[List[float]] = None  # Policy network logits
    value_estimate: float = 0.0  # Value function estimate
    advantage: float = 0.0  # Advantage estimate for policy gradient

    # ===== Shared Workflow Attributes =====
    graph_code: str = ""  # Python code representing the workflow graph
    prompt_code: str = ""  # Prompts used in the workflow operators
    operators: List[str] = field(default_factory=list)  # List of operator names
    operator_sequence: List[str] = field(default_factory=list)  # Execution sequence
    score: float = 0.0  # Performance score (e.g., accuracy on test set)
    dataset: str = ""  # Dataset name (e.g., "HumanEval", "MATH")

    # ===== Trajectory Information =====
    trajectory_id: Optional[str] = None  # Unique identifier for RL trajectory
    step_index: int = 0  # Step index within trajectory
    is_terminal: bool = False  # Whether this is a terminal state

    # ===== Experience Information =====
    timestamp: float = 0.0  # When this state was created
    parent_score: float = 0.0  # Score of parent node (for computing reward)
    children_states: List[str] = field(default_factory=list)  # Child node IDs

    # ===== Metadata =====
    metadata: Dict[str, Any] = field(default_factory=dict)  # Additional metadata

    def __post_init__(self):
        """Generate node_id if not provided"""
        if self.mcts_node_id is None:
            self.mcts_node_id = self._generate_node_id()

    def _generate_node_id(self) -> str:
        """
        Generate unique node ID based on graph code and operators
        基于图代码和操作符生成唯一节点 ID
        """
        content = f"{self.graph_code}_{self.operators}_{self.round_number}"
        return hashlib.md5(content.encode()).hexdigest()[:16]

    def to_text_representation(self) -> str:
        """
        Convert state to text representation for LLM input
        将状态转换为文本表示，用于 LLM 输入

        Returns:
            str: Text representation suitable for LLM context
        """
        text_parts = []

        # Workflow information
        text_parts.append(f"Dataset: {self.dataset}")
        text_parts.append(f"Round: {self.round_number}")
        text_parts.append(f"Current Score: {self.score:.4f}")

        # Operators
        if self.operators:
            text_parts.append(f"Operators: {', '.join(self.operators)}")

        if self.operator_sequence:
            text_parts.append(f"Execution Sequence: {' -> '.join(self.operator_sequence)}")

        # MCTS information
        text_parts.append(f"Visit Count: {self.visit_count}")
        text_parts.append(f"UCB Score: {self.ucb_score:.4f}")

        # RL information
        text_parts.append(f"Q-Value: {self.q_value:.4f}")
        text_parts.append(f"Value Estimate: {self.value_estimate:.4f}")

        # Graph code (truncated if too long)
        graph_preview = self.graph_code[:200] + "..." if len(self.graph_code) > 200 else self.graph_code
        text_parts.append(f"Graph Code:\n{graph_preview}")

        return "\n".join(text_parts)

    def to_anchor_representation(self) -> str:
        """
        Convert state to anchor representation for GiGPO grouping
        将状态转换为锚点表示，用于 GiGPO 分组

        This representation is used to group similar states together
        for computing advantages in GiGPO.

        Returns:
            str: Anchor representation (hash of key features)
        """
        # Key features for grouping: dataset, operators, round_number, parent_node
        anchor_content = {
            "dataset": self.dataset,
            "operators": sorted(self.operators),  # Sort for consistency
            "round_number": self.round_number,
            "parent_node_id": self.parent_node_id,
            "operator_count": len(self.operators)
        }

        anchor_str = json.dumps(anchor_content, sort_keys=True)
        return hashlib.md5(anchor_str.encode()).hexdigest()[:12]

    def to_vector_representation(self, operator_vocab: Dict[str, int], max_operators: int = 10) -> np.ndarray:
        """
        Convert state to vector representation for neural networks
        将状态转换为向量表示，用于神经网络输入

        Args:
            operator_vocab: Mapping from operator names to indices
            max_operators: Maximum number of operators to encode

        Returns:
            np.ndarray: Vector representation of the state
        """
        vector_parts = []

        # Scalar features
        scalar_features = [
            self.score,
            self.parent_score,
            self.visit_count / 100.0,  # Normalize
            self.ucb_score,
            self.q_value,
            self.value_estimate,
            self.round_number / 20.0,  # Normalize (assuming max 20 rounds)
            float(self.is_terminal),
            len(self.operators) / max_operators
        ]
        vector_parts.extend(scalar_features)

        # One-hot encoding of operators
        operator_encoding = [0.0] * len(operator_vocab)
        for op in self.operators[:max_operators]:
            if op in operator_vocab:
                operator_encoding[operator_vocab[op]] = 1.0
        vector_parts.extend(operator_encoding)

        return np.array(vector_parts, dtype=np.float32)

    def compute_reward(self) -> float:
        """
        Compute reward signal for RL training
        计算 RL 训练的奖励信号

        Returns:
            float: Reward value
        """
        # Reward is improvement over parent
        improvement = self.score - self.parent_score

        # Bonus for terminal states with high scores
        terminal_bonus = 0.0
        if self.is_terminal and self.score > 0.8:
            terminal_bonus = 0.1

        # Penalty for too many operators (encourage efficiency)
        complexity_penalty = -0.01 * len(self.operators) if len(self.operators) > 5 else 0.0

        return improvement + terminal_bonus + complexity_penalty

    def clone(self) -> 'WorkflowState':
        """
        Create a deep copy of this state
        创建此状态的深拷贝

        Returns:
            WorkflowState: Cloned state
        """
        return WorkflowState(
            mcts_node_id=self.mcts_node_id,
            parent_node_id=self.parent_node_id,
            visit_count=self.visit_count,
            ucb_score=self.ucb_score,
            round_number=self.round_number,
            q_value=self.q_value,
            policy_logits=self.policy_logits.copy() if self.policy_logits else None,
            value_estimate=self.value_estimate,
            advantage=self.advantage,
            graph_code=self.graph_code,
            prompt_code=self.prompt_code,
            operators=self.operators.copy(),
            operator_sequence=self.operator_sequence.copy(),
            score=self.score,
            dataset=self.dataset,
            trajectory_id=self.trajectory_id,
            step_index=self.step_index,
            is_terminal=self.is_terminal,
            timestamp=self.timestamp,
            parent_score=self.parent_score,
            children_states=self.children_states.copy(),
            metadata=self.metadata.copy()
        )

    def update_rl_estimates(self, q_value: float, value_estimate: float,
                           policy_logits: Optional[List[float]] = None):
        """
        Update RL-related estimates
        更新 RL 相关的估计值

        Args:
            q_value: New Q-value estimate
            value_estimate: New value function estimate
            policy_logits: New policy logits (optional)
        """
        self.q_value = q_value
        self.value_estimate = value_estimate
        if policy_logits is not None:
            self.policy_logits = policy_logits

    def update_mcts_stats(self, ucb_score: float, visit_count: Optional[int] = None):
        """
        Update MCTS-related statistics
        更新 MCTS 相关的统计信息

        Args:
            ucb_score: New UCB score
            visit_count: New visit count (optional, increments by 1 if not provided)
        """
        self.ucb_score = ucb_score
        if visit_count is not None:
            self.visit_count = visit_count
        else:
            self.visit_count += 1

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert state to dictionary
        将状态转换为字典

        Returns:
            Dict[str, Any]: Dictionary representation
        """
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'WorkflowState':
        """
        Create state from dictionary
        从字典创建状态

        Args:
            data: Dictionary containing state data

        Returns:
            WorkflowState: Reconstructed state
        """
        return cls(**data)

    def __hash__(self) -> int:
        """Hash based on node_id for use in sets/dicts"""
        return hash(self.mcts_node_id)

    def __eq__(self, other) -> bool:
        """Equality based on node_id"""
        if not isinstance(other, WorkflowState):
            return False
        return self.mcts_node_id == other.mcts_node_id


class StateManager:
    """
    Manages workflow states and their relationships
    管理工作流状态及其关系

    This class provides utilities for:
    1. Storing and retrieving states
    2. Building MCTS tree structure
    3. Computing state similarities
    4. Querying states by various criteria
    """

    def __init__(self):
        self.states: Dict[str, WorkflowState] = {}  # node_id -> state
        self.anchor_groups: Dict[str, List[str]] = {}  # anchor -> [node_ids]
        self.dataset_index: Dict[str, List[str]] = {}  # dataset -> [node_ids]
        self.trajectory_index: Dict[str, List[str]] = {}  # trajectory_id -> [node_ids]

    def add_state(self, state: WorkflowState):
        """
        Add a state to the manager
        添加状态到管理器

        Args:
            state: WorkflowState to add
        """
        node_id = state.mcts_node_id
        self.states[node_id] = state

        # Update anchor groups
        anchor = state.to_anchor_representation()
        if anchor not in self.anchor_groups:
            self.anchor_groups[anchor] = []
        self.anchor_groups[anchor].append(node_id)

        # Update dataset index
        if state.dataset not in self.dataset_index:
            self.dataset_index[state.dataset] = []
        self.dataset_index[state.dataset].append(node_id)

        # Update trajectory index
        if state.trajectory_id:
            if state.trajectory_id not in self.trajectory_index:
                self.trajectory_index[state.trajectory_id] = []
            self.trajectory_index[state.trajectory_id].append(node_id)

    def get_state(self, node_id: str) -> Optional[WorkflowState]:
        """
        Retrieve a state by node_id
        通过 node_id 检索状态

        Args:
            node_id: Node identifier

        Returns:
            WorkflowState or None if not found
        """
        return self.states.get(node_id)

    def get_states_by_anchor(self, anchor: str) -> List[WorkflowState]:
        """
        Get all states with the same anchor (for GiGPO grouping)
        获取具有相同锚点的所有状态（用于 GiGPO 分组）

        Args:
            anchor: Anchor representation

        Returns:
            List[WorkflowState]: States in the same anchor group
        """
        node_ids = self.anchor_groups.get(anchor, [])
        return [self.states[nid] for nid in node_ids if nid in self.states]

    def get_states_by_dataset(self, dataset: str) -> List[WorkflowState]:
        """
        Get all states for a specific dataset
        获取特定数据集的所有状态

        Args:
            dataset: Dataset name

        Returns:
            List[WorkflowState]: States for the dataset
        """
        node_ids = self.dataset_index.get(dataset, [])
        return [self.states[nid] for nid in node_ids if nid in self.states]

    def get_trajectory(self, trajectory_id: str) -> List[WorkflowState]:
        """
        Get all states in a trajectory, sorted by step_index
        获取轨迹中的所有状态，按 step_index 排序

        Args:
            trajectory_id: Trajectory identifier

        Returns:
            List[WorkflowState]: States in the trajectory, sorted by step
        """
        node_ids = self.trajectory_index.get(trajectory_id, [])
        states = [self.states[nid] for nid in node_ids if nid in self.states]
        return sorted(states, key=lambda s: s.step_index)

    def get_children(self, node_id: str) -> List[WorkflowState]:
        """
        Get all children of a node
        获取节点的所有子节点

        Args:
            node_id: Parent node identifier

        Returns:
            List[WorkflowState]: Child states
        """
        parent_state = self.get_state(node_id)
        if parent_state is None:
            return []

        child_ids = parent_state.children_states
        return [self.states[cid] for cid in child_ids if cid in self.states]

    def get_parent(self, node_id: str) -> Optional[WorkflowState]:
        """
        Get parent of a node
        获取节点的父节点

        Args:
            node_id: Child node identifier

        Returns:
            WorkflowState or None: Parent state
        """
        child_state = self.get_state(node_id)
        if child_state is None or child_state.parent_node_id is None:
            return None

        return self.get_state(child_state.parent_node_id)

    def get_path_to_root(self, node_id: str) -> List[WorkflowState]:
        """
        Get path from node to root in MCTS tree
        获取从节点到根节点的路径

        Args:
            node_id: Starting node identifier

        Returns:
            List[WorkflowState]: Path from node to root (including both)
        """
        path = []
        current_id = node_id

        while current_id is not None:
            state = self.get_state(current_id)
            if state is None:
                break
            path.append(state)
            current_id = state.parent_node_id

        return path

    def get_best_states(self, n: int = 10, metric: str = "score") -> List[WorkflowState]:
        """
        Get top N states by a metric
        按指标获取前 N 个状态

        Args:
            n: Number of states to return
            metric: Metric to sort by ("score", "q_value", "ucb_score")

        Returns:
            List[WorkflowState]: Top N states
        """
        all_states = list(self.states.values())

        if metric == "score":
            sorted_states = sorted(all_states, key=lambda s: s.score, reverse=True)
        elif metric == "q_value":
            sorted_states = sorted(all_states, key=lambda s: s.q_value, reverse=True)
        elif metric == "ucb_score":
            sorted_states = sorted(all_states, key=lambda s: s.ucb_score, reverse=True)
        else:
            raise ValueError(f"Unknown metric: {metric}")

        return sorted_states[:n]

    def clear(self):
        """Clear all stored states"""
        self.states.clear()
        self.anchor_groups.clear()
        self.dataset_index.clear()
        self.trajectory_index.clear()

    def __len__(self) -> int:
        """Number of states in the manager"""
        return len(self.states)
