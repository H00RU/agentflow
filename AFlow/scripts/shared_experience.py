"""
Shared Experience Pool for Deep Integration
共享经验池，用于 AFlow 和 verl-agent 的深度集成

This module provides a thread-safe shared experience pool that enables
bidirectional learning between AFlow's MCTS optimization and verl-agent's RL training.

Key features:
1. Thread-safe operations for concurrent access
2. Fast indexing by score, operator, round, and dataset
3. Experience sampling strategies for RL training
4. Statistics and analytics for monitoring
"""

import threading
import time
import pickle
import json
from collections import defaultdict
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import heapq
import numpy as np


@dataclass
class Experience:
    """
    Single experience entry
    单个经验条目

    Contains complete information about a workflow optimization step:
    - State information (graph, operators, prompts)
    - Performance metrics (score, improvement)
    - Context (round, dataset, parent)
    - Metadata (timestamp, tags)
    """
    # State information
    graph_code: str
    prompt_code: str
    operators: List[str]
    operator_sequence: List[str]

    # Performance metrics
    score: float
    parent_score: float
    improvement: float

    # Context
    round_number: int
    dataset: str
    parent_node_id: Optional[str] = None
    node_id: Optional[str] = None

    # MCTS information
    visit_count: int = 0
    ucb_score: float = 0.0

    # RL information
    q_value: float = 0.0
    value_estimate: float = 0.0
    advantage: float = 0.0

    # Trajectory information
    trajectory_id: Optional[str] = None
    step_index: int = 0
    is_terminal: bool = False

    # Metadata
    timestamp: float = 0.0
    tags: List[str] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.tags is None:
            self.tags = []
        if self.metadata is None:
            self.metadata = {}
        if self.timestamp == 0.0:
            self.timestamp = time.time()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Experience':
        """Create from dictionary"""
        return cls(**data)


class SharedExperiencePool:
    """
    Thread-safe shared experience pool
    线程安全的共享经验池

    Maintained jointly by AFlow and verl-agent for bidirectional learning.

    Design principles:
    1. Thread-safe: Uses locks for all operations
    2. Fast queries: Multiple indices for efficient retrieval
    3. Memory efficient: LRU eviction when max_size is reached
    4. Flexible sampling: Multiple sampling strategies
    """

    def __init__(self, max_size: int = 10000, eviction_strategy: str = "fifo"):
        """
        Initialize shared experience pool

        Args:
            max_size: Maximum number of experiences to store
            eviction_strategy: Strategy for removing old experiences
                             ("fifo", "lru", "lowest_score")
        """
        self.max_size = max_size
        self.eviction_strategy = eviction_strategy

        # Main storage
        self.experiences: List[Experience] = []
        self.experience_dict: Dict[str, Experience] = {}  # node_id -> experience

        # Thread safety
        self.lock = threading.RLock()

        # Indices for fast queries
        self.score_index = defaultdict(list)  # score_bucket -> [indices]
        self.operator_index = defaultdict(list)  # operator -> [indices]
        self.round_index = defaultdict(list)  # round_number -> [indices]
        self.dataset_index = defaultdict(list)  # dataset -> [indices]
        self.trajectory_index = defaultdict(list)  # trajectory_id -> [indices]

        # Priority queue for top-k experiences
        self.top_k_heap = []  # Min heap for maintaining top K experiences

        # Statistics
        self.stats = {
            "total_added": 0,
            "total_evicted": 0,
            "total_queries": 0,
            "avg_score": 0.0,
            "max_score": 0.0,
            "min_score": float('inf')
        }

        # Access tracking for LRU
        self.access_count: Dict[int, int] = {}  # index -> access_count
        self.last_access_time: Dict[int, float] = {}  # index -> timestamp

    def add(self, experience: Experience) -> bool:
        """
        Add experience to the pool
        添加经验到池中

        Args:
            experience: Experience to add

        Returns:
            bool: True if added successfully, False if rejected
        """
        with self.lock:
            # Check if we need to evict
            if len(self.experiences) >= self.max_size:
                self._evict_one()

            # Add to main storage
            index = len(self.experiences)
            self.experiences.append(experience)

            if experience.node_id:
                self.experience_dict[experience.node_id] = experience

            # Update indices
            self._update_indices_on_add(index, experience)

            # Update statistics
            self._update_stats_on_add(experience)

            return True

    def add_batch(self, experiences: List[Experience]):
        """
        Add multiple experiences efficiently
        批量添加经验

        Args:
            experiences: List of experiences to add
        """
        with self.lock:
            for exp in experiences:
                self.add(exp)

    def get_by_score(self, min_score: float, max_score: float = float('inf'),
                     limit: Optional[int] = None) -> List[Experience]:
        """
        Get experiences within a score range
        获取分数范围内的经验

        Args:
            min_score: Minimum score (inclusive)
            max_score: Maximum score (inclusive)
            limit: Maximum number of results to return

        Returns:
            List[Experience]: Matching experiences
        """
        with self.lock:
            self.stats["total_queries"] += 1

            results = []
            for bucket, indices in self.score_index.items():
                if min_score <= bucket <= max_score:
                    for idx in indices:
                        if idx < len(self.experiences):
                            exp = self.experiences[idx]
                            if min_score <= exp.score <= max_score:
                                results.append(exp)
                                self._record_access(idx)

                            if limit and len(results) >= limit:
                                return results

            return results

    def get_by_operator(self, operator: str, limit: Optional[int] = None) -> List[Experience]:
        """
        Get experiences that use a specific operator
        获取使用特定操作符的经验

        Args:
            operator: Operator name
            limit: Maximum number of results

        Returns:
            List[Experience]: Matching experiences
        """
        with self.lock:
            self.stats["total_queries"] += 1

            indices = self.operator_index.get(operator, [])
            results = []

            for idx in indices:
                if idx < len(self.experiences):
                    results.append(self.experiences[idx])
                    self._record_access(idx)

                    if limit and len(results) >= limit:
                        break

            return results

    def get_by_round(self, round_number: int) -> List[Experience]:
        """
        Get all experiences from a specific round
        获取特定轮次的所有经验

        Args:
            round_number: Round number

        Returns:
            List[Experience]: Experiences from that round
        """
        with self.lock:
            self.stats["total_queries"] += 1

            indices = self.round_index.get(round_number, [])
            results = []

            for idx in indices:
                if idx < len(self.experiences):
                    results.append(self.experiences[idx])
                    self._record_access(idx)

            return results

    def get_by_dataset(self, dataset: str, limit: Optional[int] = None) -> List[Experience]:
        """
        Get experiences for a specific dataset
        获取特定数据集的经验

        Args:
            dataset: Dataset name
            limit: Maximum number of results

        Returns:
            List[Experience]: Matching experiences
        """
        with self.lock:
            self.stats["total_queries"] += 1

            indices = self.dataset_index.get(dataset, [])
            results = []

            for idx in indices:
                if idx < len(self.experiences):
                    results.append(self.experiences[idx])
                    self._record_access(idx)

                    if limit and len(results) >= limit:
                        break

            return results

    def get_trajectory(self, trajectory_id: str) -> List[Experience]:
        """
        Get all experiences in a trajectory, sorted by step_index
        获取轨迹中的所有经验，按 step_index 排序

        Args:
            trajectory_id: Trajectory identifier

        Returns:
            List[Experience]: Experiences in the trajectory
        """
        with self.lock:
            self.stats["total_queries"] += 1

            indices = self.trajectory_index.get(trajectory_id, [])
            results = []

            for idx in indices:
                if idx < len(self.experiences):
                    results.append(self.experiences[idx])
                    self._record_access(idx)

            # Sort by step_index
            return sorted(results, key=lambda e: e.step_index)

    def get_best(self, n: int = 10, dataset: Optional[str] = None) -> List[Experience]:
        """
        Get top N experiences by score
        获取分数最高的 N 个经验

        Args:
            n: Number of experiences to return
            dataset: Optional dataset filter

        Returns:
            List[Experience]: Top N experiences
        """
        with self.lock:
            self.stats["total_queries"] += 1

            if dataset:
                candidates = [exp for exp in self.experiences if exp.dataset == dataset]
            else:
                candidates = self.experiences

            # Sort by score descending
            sorted_exps = sorted(candidates, key=lambda e: e.score, reverse=True)

            for i in range(min(n, len(sorted_exps))):
                # Find original index for access tracking
                for idx, exp in enumerate(self.experiences):
                    if exp == sorted_exps[i]:
                        self._record_access(idx)
                        break

            return sorted_exps[:n]

    def get_worst(self, n: int = 10, dataset: Optional[str] = None) -> List[Experience]:
        """
        Get worst N experiences by score (useful for negative examples)
        获取分数最低的 N 个经验（用于负样本）

        Args:
            n: Number of experiences to return
            dataset: Optional dataset filter

        Returns:
            List[Experience]: Worst N experiences
        """
        with self.lock:
            self.stats["total_queries"] += 1

            if dataset:
                candidates = [exp for exp in self.experiences if exp.dataset == dataset]
            else:
                candidates = self.experiences

            # Sort by score ascending
            sorted_exps = sorted(candidates, key=lambda e: e.score)

            return sorted_exps[:n]

    def sample_random(self, n: int, dataset: Optional[str] = None,
                     min_score: Optional[float] = None) -> List[Experience]:
        """
        Sample N random experiences
        随机采样 N 个经验

        Args:
            n: Number of samples
            dataset: Optional dataset filter
            min_score: Optional minimum score filter

        Returns:
            List[Experience]: Random sample
        """
        with self.lock:
            self.stats["total_queries"] += 1

            # Filter candidates
            candidates = self.experiences
            if dataset:
                candidates = [e for e in candidates if e.dataset == dataset]
            if min_score is not None:
                candidates = [e for e in candidates if e.score >= min_score]

            # Sample
            n = min(n, len(candidates))
            if n == 0:
                return []

            indices = np.random.choice(len(candidates), size=n, replace=False)
            return [candidates[i] for i in indices]

    def sample_weighted(self, n: int, temperature: float = 1.0,
                       dataset: Optional[str] = None) -> List[Experience]:
        """
        Sample experiences with probability proportional to score
        按分数加权采样经验

        Args:
            n: Number of samples
            temperature: Temperature for softmax (higher = more uniform)
            dataset: Optional dataset filter

        Returns:
            List[Experience]: Weighted sample
        """
        with self.lock:
            self.stats["total_queries"] += 1

            # Filter candidates
            candidates = self.experiences
            if dataset:
                candidates = [e for e in candidates if e.dataset == dataset]

            if len(candidates) == 0:
                return []

            # Compute weights
            scores = np.array([e.score for e in candidates])
            scores = scores / temperature
            probs = np.exp(scores - np.max(scores))  # Numerical stability
            probs = probs / np.sum(probs)

            # Sample
            n = min(n, len(candidates))
            indices = np.random.choice(len(candidates), size=n, replace=False, p=probs)

            return [candidates[i] for i in indices]

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get pool statistics
        获取池统计信息

        Returns:
            Dict[str, Any]: Statistics dictionary
        """
        with self.lock:
            stats = self.stats.copy()
            stats["current_size"] = len(self.experiences)
            stats["max_size"] = self.max_size
            stats["fill_ratio"] = len(self.experiences) / self.max_size

            # Distribution by dataset
            dataset_dist = defaultdict(int)
            for exp in self.experiences:
                dataset_dist[exp.dataset] += 1
            stats["dataset_distribution"] = dict(dataset_dist)

            # Distribution by round
            round_dist = defaultdict(int)
            for exp in self.experiences:
                round_dist[exp.round_number] += 1
            stats["round_distribution"] = dict(round_dist)

            return stats

    def clear(self):
        """Clear all experiences"""
        with self.lock:
            self.experiences.clear()
            self.experience_dict.clear()
            self.score_index.clear()
            self.operator_index.clear()
            self.round_index.clear()
            self.dataset_index.clear()
            self.trajectory_index.clear()
            self.top_k_heap.clear()
            self.access_count.clear()
            self.last_access_time.clear()

            # Reset stats
            self.stats = {
                "total_added": 0,
                "total_evicted": 0,
                "total_queries": 0,
                "avg_score": 0.0,
                "max_score": 0.0,
                "min_score": float('inf')
            }

    def save(self, filepath: str):
        """
        Save pool to disk
        保存池到磁盘

        Args:
            filepath: Path to save file
        """
        with self.lock:
            data = {
                "experiences": [e.to_dict() for e in self.experiences],
                "stats": self.stats,
                "max_size": self.max_size,
                "eviction_strategy": self.eviction_strategy
            }

            with open(filepath, 'wb') as f:
                pickle.dump(data, f)

    def load(self, filepath: str):
        """
        Load pool from disk
        从磁盘加载池

        Args:
            filepath: Path to load file
        """
        with self.lock:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)

            self.clear()

            # Restore settings
            self.max_size = data.get("max_size", 10000)
            self.eviction_strategy = data.get("eviction_strategy", "fifo")
            self.stats = data.get("stats", {})

            # Restore experiences
            for exp_dict in data.get("experiences", []):
                exp = Experience.from_dict(exp_dict)
                self.add(exp)

    # Private methods

    def _update_indices_on_add(self, index: int, experience: Experience):
        """Update all indices when adding an experience"""
        # Score index (bucketed by 0.1)
        score_bucket = round(experience.score, 1)
        self.score_index[score_bucket].append(index)

        # Operator index
        for op in experience.operators:
            self.operator_index[op].append(index)

        # Round index
        self.round_index[experience.round_number].append(index)

        # Dataset index
        self.dataset_index[experience.dataset].append(index)

        # Trajectory index
        if experience.trajectory_id:
            self.trajectory_index[experience.trajectory_id].append(index)

        # Initialize access tracking
        self.access_count[index] = 0
        self.last_access_time[index] = time.time()

    def _update_stats_on_add(self, experience: Experience):
        """Update statistics when adding an experience"""
        self.stats["total_added"] += 1

        # Update average score
        n = len(self.experiences)
        old_avg = self.stats["avg_score"]
        self.stats["avg_score"] = (old_avg * (n - 1) + experience.score) / n

        # Update max/min score
        self.stats["max_score"] = max(self.stats["max_score"], experience.score)
        self.stats["min_score"] = min(self.stats["min_score"], experience.score)

    def _record_access(self, index: int):
        """Record access for LRU tracking"""
        self.access_count[index] = self.access_count.get(index, 0) + 1
        self.last_access_time[index] = time.time()

    def _evict_one(self):
        """Evict one experience based on eviction strategy"""
        if len(self.experiences) == 0:
            return

        if self.eviction_strategy == "fifo":
            # Remove oldest (first added)
            evict_idx = 0

        elif self.eviction_strategy == "lru":
            # Remove least recently accessed
            evict_idx = min(self.last_access_time.keys(),
                          key=lambda k: self.last_access_time[k])

        elif self.eviction_strategy == "lowest_score":
            # Remove lowest scoring
            evict_idx = min(range(len(self.experiences)),
                          key=lambda i: self.experiences[i].score)

        else:
            evict_idx = 0  # Default to FIFO

        # Remove from main storage
        evicted_exp = self.experiences.pop(evict_idx)

        if evicted_exp.node_id and evicted_exp.node_id in self.experience_dict:
            del self.experience_dict[evicted_exp.node_id]

        # Rebuild indices (expensive, but ensures consistency)
        self._rebuild_indices()

        self.stats["total_evicted"] += 1

    def _rebuild_indices(self):
        """Rebuild all indices from scratch"""
        self.score_index.clear()
        self.operator_index.clear()
        self.round_index.clear()
        self.dataset_index.clear()
        self.trajectory_index.clear()
        self.access_count.clear()
        self.last_access_time.clear()

        for idx, exp in enumerate(self.experiences):
            self._update_indices_on_add(idx, exp)

    def __len__(self) -> int:
        """Number of experiences in pool"""
        return len(self.experiences)

    def __repr__(self) -> str:
        """String representation"""
        return (f"SharedExperiencePool(size={len(self)}/{self.max_size}, "
                f"avg_score={self.stats['avg_score']:.4f})")

    def __getstate__(self):
        """Get state for pickling (exclude lock)"""
        state = self.__dict__.copy()
        # Remove the unpicklable lock
        state.pop('lock', None)
        return state

    def __setstate__(self, state):
        """Restore state after unpickling (recreate lock)"""
        self.__dict__.update(state)
        # Recreate the lock
        self.lock = threading.RLock()
