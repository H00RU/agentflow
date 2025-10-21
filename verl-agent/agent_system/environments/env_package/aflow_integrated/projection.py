"""
Projection utilities for AFlow environment
AFlow 环境的投影工具

Handles projection of observations and actions between
AFlow's workflow representation and RL agent's text format.
"""

from typing import List, Dict, Any, Optional


class AFlowProjection:
    """
    Handles projection between AFlow workflow states and RL observations
    处理 AFlow 工作流状态和 RL 观测之间的投影
    """

    def __init__(self, max_obs_length: int = 2000):
        """
        Initialize projection

        Args:
            max_obs_length: Maximum observation text length
        """
        self.max_obs_length = max_obs_length

    def project_observation(self, obs: str) -> str:
        """
        Project observation to standardized format
        将观测投影到标准格式

        Args:
            obs: Raw observation text

        Returns:
            str: Projected observation
        """
        # Truncate if too long
        if len(obs) > self.max_obs_length:
            obs = obs[:self.max_obs_length] + "\n... (truncated)"

        return obs

    def project_action(self, action: str) -> str:
        """
        Project action to standardized format
        将动作投影到标准格式

        Args:
            action: Raw action text

        Returns:
            str: Projected action
        """
        # Clean and standardize action text
        action = action.strip()

        # Remove any special tokens that might interfere
        action = action.replace("<|endoftext|>", "")
        action = action.replace("<|pad|>", "")

        return action

    def project_reward(self, reward: float) -> float:
        """
        Project reward to standardized range
        将奖励投影到标准范围

        Args:
            reward: Raw reward value

        Returns:
            float: Projected reward
        """
        # Clip to reasonable range
        return max(min(reward, 1.0), -1.0)

    def batch_project_observations(self, obs_list: List[str]) -> List[str]:
        """
        Project a batch of observations
        批量投影观测

        Args:
            obs_list: List of observations

        Returns:
            List[str]: Projected observations
        """
        return [self.project_observation(obs) for obs in obs_list]

    def batch_project_actions(self, action_list: List[str]) -> List[str]:
        """
        Project a batch of actions
        批量投影动作

        Args:
            action_list: List of actions

        Returns:
            List[str]: Projected actions
        """
        return [self.project_action(action) for action in action_list]

    def batch_project_rewards(self, reward_list: List[float]) -> List[float]:
        """
        Project a batch of rewards
        批量投影奖励

        Args:
            reward_list: List of rewards

        Returns:
            List[float]: Projected rewards
        """
        return [self.project_reward(r) for r in reward_list]
