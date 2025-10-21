"""
AFlow Integrated Environment Implementation
AFlow 深度集成环境实现

This environment deeply integrates with AFlow's internal components:
1. Uses RLEnhancedOptimizer to participate in MCTS selection
2. Shares experience pool bidirectionally
3. Tracks WorkflowState for unified representation
4. Maps MCTS nodes to RL episodes

The environment treats workflow design as an RL problem:
- State: Current workflow graph + operators + performance
- Action: Modifications to the workflow (text-based)
- Reward: Improvement in task performance
"""

import gymnasium as gym
import ray
import numpy as np
import asyncio
import sys
import os
from typing import List, Dict, Any, Optional, Tuple
import json
import time

# Add paths for importing AFlow and integration components
# Use absolute paths if possible, fallback to relative
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Try to find aflow_integration directory
if 'aflow_integration' in SCRIPT_DIR:
    # Extract base directory
    base_idx = SCRIPT_DIR.index('aflow_integration')
    BASE_DIR = SCRIPT_DIR[:base_idx + len('aflow_integration')]
    AFLOW_PATH = os.path.join(BASE_DIR, 'AFlow')
    INTEGRATION_PATH = os.path.join(BASE_DIR, 'integration')
    VERL_PATH = os.path.join(BASE_DIR, 'verl-agent')
else:
    # Fallback to relative paths
    AFLOW_PATH = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..', '..', '..', '..', 'AFlow'))
    INTEGRATION_PATH = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..', '..', '..', '..', 'integration'))
    VERL_PATH = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..', '..', '..', '..'))

# Add to sys.path at module level (before any imports)
for path in [AFLOW_PATH, INTEGRATION_PATH, VERL_PATH]:
    if path not in sys.path:
        sys.path.insert(0, path)

# Import AFlow components
try:
    from scripts.optimizer_rl import RLEnhancedOptimizer
    from scripts.shared_experience import SharedExperiencePool, Experience
    from scripts.evaluator import DatasetType
    from scripts.logs import logger
    from unified_state import WorkflowState, StateManager
    AFLOW_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import AFlow components: {e}")
    AFLOW_AVAILABLE = False
    RLEnhancedOptimizer = None
    SharedExperiencePool = None
    Experience = None
    WorkflowState = None
    StateManager = None


def _setup_worker_paths():
    """
    Setup Python paths in Ray worker
    在 Ray worker 中设置 Python 路径

    This function is called at worker initialization to ensure
    all necessary modules are importable.
    """
    import sys
    import os

    # Get the directory of this file
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Navigate up to find aflow_integration directory
    # From: .../verl-agent/agent_system/environments/env_package/aflow_integrated
    # To: .../aflow_integration
    base_dir = current_dir
    for _ in range(6):  # Go up 6 levels
        base_dir = os.path.dirname(base_dir)

    # Add paths
    aflow_path = os.path.join(base_dir, 'AFlow')
    integration_path = os.path.join(base_dir, 'integration')
    verl_path = os.path.join(base_dir, 'verl-agent')

    paths_to_add = [aflow_path, integration_path, verl_path, base_dir]

    for path in paths_to_add:
        if os.path.exists(path) and path not in sys.path:
            sys.path.insert(0, path)
            print(f"[Worker] Added to sys.path: {path}")


class AFlowWorker:
    """
    Worker that manages a single AFlow optimization process
    管理单个 AFlow 优化过程的工作器

    This worker:
    1. Creates RLEnhancedOptimizer with RL policy
    2. Runs workflow optimization with RL guidance
    3. Returns observations, rewards, and state information
    """

    def __init__(
        self,
        dataset: str,
        question_type: str,
        opt_llm_config: Dict,
        exec_llm_config: Dict,
        operators: List[str],
        sample: int,
        optimized_path: str,
        max_rounds: int = 20,
        validation_rounds: int = 5,
        shared_experience_pool = None,
        state_manager = None,
        worker_id: int = 0
    ):
        """
        Initialize AFlow worker

        Args:
            dataset: Dataset name (e.g., "HumanEval", "MATH", "GSM8K")
            question_type: Question type ("math", "code", "qa")
            opt_llm_config: LLM config for optimization
            exec_llm_config: LLM config for execution
            operators: List of available operators
            sample: Number of top rounds to sample from
            optimized_path: Path to store optimized workflows
            max_rounds: Maximum optimization rounds
            validation_rounds: Number of validation rounds
            shared_experience_pool: Shared experience pool (optional)
            state_manager: State manager (optional)
            worker_id: Worker identifier
        """
        # Setup paths in worker
        _setup_worker_paths()

        # Now import required modules
        from scripts.optimizer_rl import RLEnhancedOptimizer
        from scripts.shared_experience import SharedExperiencePool
        from unified_state import StateManager

        print(f"[Worker {worker_id}] Initializing AFlowWorker for {dataset}")

        self.dataset = dataset
        self.question_type = question_type
        self.worker_id = worker_id

        # Create local shared components if not provided
        if shared_experience_pool is None:
            shared_experience_pool = SharedExperiencePool(max_size=1000)
            print(f"[Worker {worker_id}] Created local experience pool")

        if state_manager is None:
            state_manager = StateManager()
            print(f"[Worker {worker_id}] Created local state manager")

        # Create optimizer with RL enhancement
        # Use AFlow workspace path - must be /path/to/aflow_integration/AFlow/workspace
        import os

        # Find aflow_integration base directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        base_dir = current_dir
        for _ in range(6):
            base_dir = os.path.dirname(base_dir)
            if os.path.exists(os.path.join(base_dir, 'AFlow', 'workspace')):
                break

        # Always use AFlow workspace directory
        aflow_workspace = os.path.join(base_dir, 'AFlow', 'workspace')

        if not os.path.exists(aflow_workspace):
            raise RuntimeError(f"AFlow workspace not found at: {aflow_workspace}")

        optimized_path = aflow_workspace
        print(f"[Worker {worker_id}] Using AFlow workspace: {optimized_path}")

        self.optimizer = RLEnhancedOptimizer(
            dataset=dataset,  # Fixed: pass string directly
            question_type=question_type,
            opt_llm_config=opt_llm_config,
            exec_llm_config=exec_llm_config,
            operators=operators,
            sample=sample,
            optimized_path=optimized_path,
            initial_round=1,
            max_rounds=max_rounds,
            validation_rounds=validation_rounds,
            rl_policy=None,  # Will be set later
            use_rl_guidance=True,
            rl_weight=0.5,
            shared_experience_pool=shared_experience_pool,
            state_manager=state_manager,
            enable_state_tracking=True
        )

        print(f"✓ [Worker {worker_id}] AFlowWorker initialized successfully")

        # Episode state
        self.current_round = 0
        self.current_score = 0.0
        self.episode_history = []
        self.is_done = False

        # RL policy (will be set externally)
        self.rl_policy = None

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict]:
        """
        Reset the environment for a new episode
        重置环境以开始新的 episode

        Returns:
            Tuple[str, Dict]: (observation, info)
        """
        if seed is not None:
            np.random.seed(seed)

        # Reset episode state
        self.current_round = 1
        self.current_score = 0.0
        self.episode_history = []
        self.is_done = False

        # Reset optimizer
        self.optimizer.round = 1
        self.optimizer.rl_trajectory = []
        self.optimizer.trajectory_step_index = 0

        # Set RL policy
        if self.rl_policy is not None:
            self.optimizer.set_rl_policy(self.rl_policy)

        # Initial observation
        obs = self._get_observation()

        info = {
            "round": self.current_round,
            "dataset": self.dataset,
            "worker_id": self.worker_id
        }

        return obs, info

    def step(self, action: str) -> Tuple[str, float, bool, Dict]:
        """
        Execute one optimization step
        执行一次优化步骤

        Args:
            action: Action text (modification suggestion for workflow)

        Returns:
            Tuple: (observation, reward, done, info)
        """
        # Update RL policy's current action suggestion
        if self.rl_policy is not None and hasattr(self.rl_policy, 'set_action'):
            self.rl_policy.set_action(action)

        # Run one round of optimization
        try:
            # Create event loop and run optimization
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            new_score = loop.run_until_complete(self.optimizer._optimize_graph())
            loop.close()

            # Compute reward (improvement over previous score)
            reward = new_score - self.current_score if self.current_round > 1 else new_score

            # Update state
            previous_score = self.current_score
            self.current_score = new_score
            self.current_round = self.optimizer.round

            # Check if done
            self.is_done = (
                self.current_round >= self.optimizer.max_rounds or
                self.current_score >= 0.95  # High performance threshold
            )

            # Record history
            self.episode_history.append({
                "round": self.current_round,
                "action": action,
                "score": new_score,
                "reward": reward,
                "previous_score": previous_score
            })

            # Get next observation
            obs = self._get_observation()

            # Info dict
            info = {
                "round": self.current_round,
                "score": self.current_score,
                "reward": reward,
                "dataset": self.dataset,
                "worker_id": self.worker_id,
                "episode_length": len(self.episode_history)
            }

            # Add state information if available
            if self.optimizer.state_manager:
                state_id = self.optimizer.node_to_state_mapping.get(self.current_round)
                if state_id:
                    state = self.optimizer.state_manager.get_state(state_id)
                    if state:
                        info["state_id"] = state_id
                        info["mcts_node_id"] = state.mcts_node_id
                        info["parent_node_id"] = state.parent_node_id
                        info["q_value"] = state.q_value
                        info["ucb_score"] = state.ucb_score

        except Exception as e:
            logger.error(f"Error in step: {e}")
            import traceback
            traceback.print_exc()

            # Return error state
            obs = self._get_observation()
            reward = -0.1  # Penalty for error
            self.is_done = True

            info = {
                "error": str(e),
                "round": self.current_round,
                "dataset": self.dataset,
                "worker_id": self.worker_id
            }

        return obs, reward, self.is_done, info

    def _get_observation(self) -> str:
        """
        Get current observation (state representation)
        获取当前观测（状态表示）

        Returns:
            str: Text observation for LLM
        """
        # Get current workflow state
        state_id = self.optimizer.node_to_state_mapping.get(self.current_round)

        if state_id and self.optimizer.state_manager:
            state = self.optimizer.state_manager.get_state(state_id)
            if state:
                return state.to_text_representation()

        # Fallback observation
        obs_parts = [
            f"Dataset: {self.dataset}",
            f"Question Type: {self.question_type}",
            f"Current Round: {self.current_round}",
            f"Current Score: {self.current_score:.4f}",
            f"Available Operators: {', '.join(self.optimizer.operators)}"
        ]

        if len(self.episode_history) > 0:
            obs_parts.append(f"Previous Reward: {self.episode_history[-1]['reward']:.4f}")

        return "\n".join(obs_parts)

    def set_rl_policy(self, rl_policy):
        """Set RL policy for this worker"""
        self.rl_policy = rl_policy
        if self.optimizer:
            self.optimizer.set_rl_policy(rl_policy)

    def get_statistics(self) -> Dict[str, Any]:
        """Get worker statistics"""
        stats = {
            "worker_id": self.worker_id,
            "dataset": self.dataset,
            "current_round": self.current_round,
            "current_score": self.current_score,
            "episode_length": len(self.episode_history),
            "is_done": self.is_done
        }

        if self.optimizer:
            rl_stats = self.optimizer.get_rl_statistics()
            stats.update(rl_stats)

        return stats


class AFlowMultiProcessEnv(gym.Env):
    """
    Ray-based parallel environment for AFlow workflow optimization
    基于 Ray 的并行 AFlow 工作流优化环境

    This environment:
    1. Manages multiple AFlow workers in parallel
    2. Coordinates shared experience pool
    3. Provides gym-compatible interface
    4. Supports GiGPO's group-based training
    """

    def __init__(
        self,
        dataset: str,
        question_type: str,
        opt_llm_config: Dict,
        exec_llm_config: Dict,
        operators: List[str],
        sample: int = 3,
        optimized_path: str = "./optimized_workflows",
        max_rounds: int = 20,
        validation_rounds: int = 5,
        seed: int = 0,
        env_num: int = 1,
        group_n: int = 1,
        resources_per_worker: Dict = {"num_cpus": 1.0},
        is_train: bool = True
    ):
        """
        Initialize parallel AFlow environment

        Args:
            dataset: Dataset name
            question_type: Question type
            opt_llm_config: LLM config for optimization
            exec_llm_config: LLM config for execution
            operators: Available operators
            sample: Number of top rounds to sample
            optimized_path: Path for optimized workflows
            max_rounds: Maximum rounds per episode
            validation_rounds: Validation rounds
            seed: Random seed
            env_num: Number of distinct environments
            group_n: Number of replicas per group (for GiGPO)
            resources_per_worker: Ray resources per worker
            is_train: Training mode flag
        """
        super().__init__()

        if not AFLOW_AVAILABLE:
            raise RuntimeError("AFlow components not available. Please check imports.")

        # Initialize Ray (paths are set up at module level)
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)

        self.dataset = dataset
        self.question_type = question_type
        self.is_train = is_train
        self.group_n = group_n
        self.env_num = env_num
        self.num_processes = env_num * group_n

        np.random.seed(seed)

        # Note: For now, workers will create their own experience pools and state managers
        # In future, we can use Ray actors for true sharing
        self.shared_pool_ref = None
        self.state_manager_ref = None

        # Create Ray workers (paths are set up at module level)
        env_worker = ray.remote(**resources_per_worker)(AFlowWorker)
        self.workers = []

        for i in range(self.num_processes):
            worker = env_worker.remote(
                dataset=dataset,
                question_type=question_type,
                opt_llm_config=opt_llm_config,
                exec_llm_config=exec_llm_config,
                operators=operators,
                sample=sample,
                optimized_path=f"{optimized_path}/worker_{i}",
                max_rounds=max_rounds,
                validation_rounds=validation_rounds,
                shared_experience_pool=self.shared_pool_ref,
                state_manager=self.state_manager_ref,
                worker_id=i
            )
            self.workers.append(worker)

        # RL policy (will be set externally)
        self.rl_policy = None

    def reset(self) -> Tuple[List[str], List[Dict]]:
        """
        Reset all environments
        重置所有环境

        Returns:
            Tuple: (obs_list, info_list)
        """
        # Generate seeds
        if self.is_train:
            seeds = np.random.randint(0, 2**16 - 1, size=self.env_num)
        else:
            seeds = np.random.randint(2**16, 2**32 - 1, size=self.env_num)

        # Repeat seeds for environments in the same group
        seeds = np.repeat(seeds, self.group_n).tolist()

        # Reset all workers in parallel
        futures = []
        for i, worker in enumerate(self.workers):
            # Set RL policy if available
            if self.rl_policy is not None:
                worker.set_rl_policy.remote(self.rl_policy)

            future = worker.reset.remote(seeds[i])
            futures.append(future)

        # Collect results
        results = ray.get(futures)
        obs_list = [r[0] for r in results]
        info_list = [r[1] for r in results]

        return obs_list, info_list

    def step(self, actions: List[str]) -> Tuple[List[str], List[float], List[bool], List[Dict]]:
        """
        Execute actions in all environments
        在所有环境中执行动作

        Args:
            actions: List of action strings, one per worker

        Returns:
            Tuple: (obs_list, reward_list, done_list, info_list)
        """
        assert len(actions) == self.num_processes, \
            f"Expected {self.num_processes} actions, got {len(actions)}"

        # Execute steps in parallel
        futures = []
        for worker, action in zip(self.workers, actions):
            future = worker.step.remote(action)
            futures.append(future)

        # Collect results
        results = ray.get(futures)

        obs_list = [r[0] for r in results]
        reward_list = [r[1] for r in results]
        done_list = [r[2] for r in results]
        info_list = [r[3] for r in results]

        return obs_list, reward_list, done_list, info_list

    def set_rl_policy(self, rl_policy):
        """
        Set RL policy for all workers
        为所有工作器设置 RL 策略

        Args:
            rl_policy: RL policy object
        """
        self.rl_policy = rl_policy

        # Set for all workers
        futures = []
        for worker in self.workers:
            future = worker.set_rl_policy.remote(rl_policy)
            futures.append(future)

        ray.get(futures)

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics from all workers
        从所有工作器获取统计信息

        Returns:
            Dict: Aggregated statistics
        """
        futures = [worker.get_statistics.remote() for worker in self.workers]
        worker_stats = ray.get(futures)

        # Aggregate statistics
        stats = {
            "num_workers": len(self.workers),
            "dataset": self.dataset,
            "worker_stats": worker_stats
        }

        # Compute aggregates
        total_rounds = sum(s["current_round"] for s in worker_stats)
        total_score = sum(s["current_score"] for s in worker_stats)
        num_done = sum(s["is_done"] for s in worker_stats)

        stats["total_rounds"] = total_rounds
        stats["avg_score"] = total_score / len(worker_stats) if worker_stats else 0.0
        stats["num_done"] = num_done

        return stats

    def close(self):
        """Close all workers"""
        for worker in self.workers:
            ray.kill(worker)

    def __del__(self):
        try:
            self.close()
        except:
            pass


def build_aflow_envs(
    dataset: str,
    question_type: str,
    opt_llm_config: Dict,
    exec_llm_config: Dict,
    operators: List[str],
    sample: int = 3,
    optimized_path: str = "./optimized_workflows",
    max_rounds: int = 20,
    validation_rounds: int = 5,
    seed: int = 0,
    env_num: int = 1,
    group_n: int = 1,
    resources_per_worker: Dict = {"num_cpus": 1.0},
    is_train: bool = True
) -> AFlowMultiProcessEnv:
    """
    Builder function for AFlow integrated environment
    构建 AFlow 集成环境

    Args:
        dataset: Dataset name (e.g., "HumanEval", "MATH", "GSM8K")
        question_type: Question type ("math", "code", "qa")
        opt_llm_config: LLM configuration for optimization
        exec_llm_config: LLM configuration for execution
        operators: List of available operators
        sample: Number of top rounds to sample from
        optimized_path: Path to store optimized workflows
        max_rounds: Maximum optimization rounds per episode
        validation_rounds: Number of validation rounds
        seed: Random seed
        env_num: Number of distinct environments
        group_n: Number of replicas per group (for GiGPO)
        resources_per_worker: Ray resources per worker
        is_train: Training mode flag

    Returns:
        AFlowMultiProcessEnv: Parallel environment instance
    """
    return AFlowMultiProcessEnv(
        dataset=dataset,
        question_type=question_type,
        opt_llm_config=opt_llm_config,
        exec_llm_config=exec_llm_config,
        operators=operators,
        sample=sample,
        optimized_path=optimized_path,
        max_rounds=max_rounds,
        validation_rounds=validation_rounds,
        seed=seed,
        env_num=env_num,
        group_n=group_n,
        resources_per_worker=resources_per_worker,
        is_train=is_train
    )
