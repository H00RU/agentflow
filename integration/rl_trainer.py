"""
RL Trainer for End-to-End Training
端到端训练的 RL 训练器

This module implements:
1. Trajectory collection from AFlow environments
2. Advantage computation using standard GRPO
3. Policy and value loss computation
4. Gradient updates

此模块实现：
1. 从 AFlow 环境收集轨迹
2. 使用标准 GRPO 计算优势
3. 策略和价值损失计算
4. 梯度更新
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict
import sys
import os

# REMOVED: GiGPO import - GiGPO found to be unstable, using GRPO uniformly
# sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'verl-agent'))
# from gigpo.workflow_gigpo import compute_workflow_gigpo_advantage


class RolloutBuffer:
    """
    Buffer for storing trajectories
    存储轨迹的缓冲区
    """

    def __init__(self):
        self.observations = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.response_masks = []
        self.input_ids = []  # Store complete input_ids (prompt + response) for consistent re-computation
        self.attention_masks = []  # Store attention masks

        # Workflow-specific information
        self.workflow_nodes = []
        self.workflow_states = []
        self.episode_indices = []
        self.trajectory_indices = []

    def add(
        self,
        obs: str,
        action: str,
        log_prob: torch.Tensor,
        reward: float,
        done: bool,
        response_mask: torch.Tensor,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        workflow_node: Optional[str] = None,
        workflow_state: Optional[Any] = None,
        episode_idx: int = 0,
        traj_idx: int = 0
    ):
        """Add a step to the buffer (GRPO: no values)"""
        self.observations.append(obs)
        self.actions.append(action)
        self.log_probs.append(log_prob.detach().cpu())
        self.rewards.append(reward)
        self.dones.append(done)
        self.response_masks.append(response_mask.detach().cpu())
        # Store complete sequences for consistent re-computation during updates
        if input_ids is not None:
            self.input_ids.append(input_ids.detach().cpu())
        if attention_mask is not None:
            self.attention_masks.append(attention_mask.detach().cpu())

        self.workflow_nodes.append(workflow_node)
        self.workflow_states.append(workflow_state)
        self.episode_indices.append(episode_idx)
        self.trajectory_indices.append(traj_idx)

    def clear(self):
        """Clear buffer"""
        self.__init__()

    def get(self) -> Dict[str, Any]:
        """Get all data from buffer (GRPO: no values)"""
        # Handle variable-length sequences by padding
        log_probs_padded = None
        response_masks_padded = None

        if self.log_probs:
            # Pad log_probs to max length in batch
            # Each log_prob is shape [1, seq_len], we need to pad seq_len dimension
            max_len = max(lp.shape[1] for lp in self.log_probs)
            log_probs_list = []
            for lp in self.log_probs:
                if lp.shape[1] < max_len:
                    # Pad with zeros
                    padding = torch.zeros(lp.shape[0], max_len - lp.shape[1], dtype=lp.dtype)
                    lp_padded = torch.cat([lp, padding], dim=1)
                else:
                    lp_padded = lp
                log_probs_list.append(lp_padded)
            log_probs_padded = torch.stack(log_probs_list)

        if self.response_masks:
            # Pad response_masks to max length
            max_len = max(rm.shape[1] for rm in self.response_masks)
            response_masks_list = []
            for rm in self.response_masks:
                if rm.shape[1] < max_len:
                    # Pad with zeros (False for masks)
                    padding = torch.zeros(rm.shape[0], max_len - rm.shape[1], dtype=rm.dtype)
                    rm_padded = torch.cat([rm, padding], dim=1)
                else:
                    rm_padded = rm
                response_masks_list.append(rm_padded)
            response_masks_padded = torch.stack(response_masks_list)

        # Pad input_ids and attention_masks if available
        input_ids_padded = None
        attention_masks_padded = None
        if self.input_ids:
            max_len = max(ids.shape[1] for ids in self.input_ids)
            input_ids_list = []
            attention_masks_list = []
            for ids, mask in zip(self.input_ids, self.attention_masks):
                # Squeeze out batch dimension if present (1, seq_len) -> (seq_len,)
                if ids.dim() == 2 and ids.shape[0] == 1:
                    ids = ids.squeeze(0)
                    mask = mask.squeeze(0)

                if ids.shape[0] < max_len:
                    # Pad to max length
                    padding = torch.zeros(max_len - ids.shape[0], dtype=ids.dtype)
                    ids_padded = torch.cat([ids, padding], dim=0)
                    mask_padded = torch.cat([mask, torch.zeros(max_len - mask.shape[0], dtype=mask.dtype)], dim=0)
                else:
                    ids_padded = ids
                    mask_padded = mask
                input_ids_list.append(ids_padded)
                attention_masks_list.append(mask_padded)
            input_ids_padded = torch.stack(input_ids_list)  # Now (bs, seq_len)
            attention_masks_padded = torch.stack(attention_masks_list)  # Now (bs, seq_len)

        return {
            'observations': self.observations,
            'actions': self.actions,
            'log_probs': log_probs_padded,
            'rewards': torch.tensor(self.rewards, dtype=torch.float32),
            'dones': torch.tensor(self.dones, dtype=torch.bool),
            'response_masks': response_masks_padded,
            'input_ids': input_ids_padded,  # Complete sequences for re-computation
            'attention_masks': attention_masks_padded,
            'workflow_nodes': np.array(self.workflow_nodes) if self.workflow_nodes else None,
            'workflow_states': self.workflow_states,
            'episode_indices': np.array(self.episode_indices),
            'trajectory_indices': np.array(self.trajectory_indices)
        }


class RLTrainer:
    """
    RL Trainer for end-to-end policy training
    端到端策略训练的 RL 训练器
    """

    def __init__(
        self,
        policy,
        learning_rate: float = 1e-5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 1.0,
        gamma: float = 0.99,  # Kept for compatibility, not used in GRPO
        ppo_epochs: int = 4,
        ppo_clip: float = 0.2,
        batch_size: int = 32,
        device: str = "cuda"
    ):
        """
        Initialize RL trainer for GRPO (no value loss)

        Args:
            policy: Trainable policy (TrainableQwenPolicy)
            learning_rate: Learning rate for optimizer
            entropy_coef: Coefficient for entropy bonus
            max_grad_norm: Max gradient norm for clipping
            gamma: Discount factor (kept for compatibility, not used in GRPO)
            ppo_epochs: Number of PPO update epochs
            ppo_clip: PPO clipping parameter
            batch_size: Batch size for updates
            device: Training device
        """
        self.policy = policy
        self.device = device

        # Hyperparameters
        self.learning_rate = learning_rate
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.gamma = gamma  # Kept for compatibility, not used in GRPO
        self.ppo_epochs = ppo_epochs
        self.ppo_clip = ppo_clip
        self.batch_size = batch_size

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.policy.parameters(),
            lr=learning_rate,
            weight_decay=0.01
        )

        # Rollout buffer
        self.buffer = RolloutBuffer()

        # Statistics (GRPO: no value loss)
        self.train_stats = {
            'policy_loss': [],
            'entropy': [],
            'total_loss': [],
            'approx_kl': [],
            'clip_fraction': []
        }

        print(f"[RLTrainer] Initialized for GRPO:")
        print(f"  - Learning rate: {learning_rate}")
        print(f"  - Entropy coef: {entropy_coef}")
        print(f"  - PPO epochs: {ppo_epochs}, Clip: {ppo_clip}")
        print(f"  - Batch size: {batch_size}")
        print(f"  - Advantage Estimator: GRPO")

    def collect_rollout(
        self,
        env,
        num_episodes: int,
        max_steps_per_episode: int = 20
    ) -> Dict[str, float]:
        """
        Collect rollouts from environment
        从环境收集 rollouts

        Args:
            env: AFlow environment
            num_episodes: Number of episodes to collect
            max_steps_per_episode: Maximum steps per episode

        Returns:
            Dict: Collection statistics
        """
        collection_stats = {
            'total_steps': 0,
            'total_reward': 0.0,
            'avg_episode_length': 0.0,
            'num_episodes': 0
        }

        for episode in range(num_episodes):
            obs_list, info_list = env.reset()

            done_list = [False] * len(obs_list)
            episode_rewards = [0.0] * len(obs_list)
            episode_steps = [0] * len(obs_list)

            for step in range(max_steps_per_episode):
                if all(done_list):
                    break

                # Get actions from policy (GRPO: no values)
                actions = []
                log_probs_list = []
                response_masks_list = []
                input_ids_list = []
                attention_masks_list = []

                for i, (obs, done) in enumerate(zip(obs_list, done_list)):
                    if not done:
                        action, log_probs, response_mask, input_ids, attention_mask = self.policy.get_action_and_value(obs)

                        actions.append(action)
                        log_probs_list.append(log_probs)
                        response_masks_list.append(response_mask)
                        input_ids_list.append(input_ids)
                        attention_masks_list.append(attention_mask)
                    else:
                        actions.append("")
                        log_probs_list.append(None)
                        response_masks_list.append(None)
                        input_ids_list.append(None)
                        attention_masks_list.append(None)

                # Step environment
                next_obs_list, reward_list, done_list, info_list = env.step(actions)

                # Store transitions (GRPO: no values)
                for i in range(len(obs_list)):
                    if log_probs_list[i] is not None:
                        self.buffer.add(
                            obs=obs_list[i],
                            action=actions[i],
                            log_prob=log_probs_list[i],
                            reward=reward_list[i],
                            done=done_list[i],
                            response_mask=response_masks_list[i],
                            input_ids=input_ids_list[i],
                            attention_mask=attention_masks_list[i],
                            workflow_node=info_list[i].get('mcts_node_id'),
                            workflow_state=info_list[i].get('state_id'),
                            episode_idx=episode * len(obs_list) + i,
                            traj_idx=step
                        )

                        episode_rewards[i] += reward_list[i]
                        episode_steps[i] += 1
                        collection_stats['total_steps'] += 1

                # Update observations
                obs_list = next_obs_list

            # Update stats
            collection_stats['total_reward'] += sum(episode_rewards)
            collection_stats['avg_episode_length'] += sum(episode_steps) / len(episode_steps)
            collection_stats['num_episodes'] += len(obs_list)

        # Average stats
        if collection_stats['num_episodes'] > 0:
            collection_stats['avg_episode_length'] /= num_episodes
            collection_stats['avg_reward'] = collection_stats['total_reward'] / collection_stats['num_episodes']

        return collection_stats

    # REMOVED: GiGPO advantage computation - GiGPO found to be unstable, using GRPO uniformly
    # def compute_advantages_gigpo(...):
    #     """Compute advantages using workflow-specific GiGPO"""
    #     ...

    def compute_advantages_grpo(
        self,
        rewards: torch.Tensor,
        response_masks: torch.Tensor,
        episode_indices: np.array,
        trajectory_indices: np.array,
        epsilon: float = 1e-6
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute advantages using standard GRPO (Group Relative Policy Optimization)
        使用标准 GRPO 计算优势

        Args:
            rewards: Rewards (bs, seq_len) or (bs,)
            response_masks: Response masks (bs, seq_len)
            episode_indices: Episode indices (bs,) for grouping
            trajectory_indices: Trajectory indices (bs,) for grouping
            epsilon: Small value to avoid division by zero

        Returns:
            Tuple: (advantages, returns)
        """
        print(f"DEBUG compute_advantages_grpo: rewards shape = {rewards.shape}, response_masks shape = {response_masks.shape}")

        # Fix: Handle 3D response_masks (bs, 1, seq_len) -> squeeze to (bs, seq_len)
        if len(response_masks.shape) == 3 and response_masks.shape[1] == 1:
            response_masks = response_masks.squeeze(1)
            print(f"DEBUG: Squeezed response_masks to shape {response_masks.shape}")

        # Handle reward shape
        if len(rewards.shape) == 1:
            # rewards is (bs,), expand to (bs, seq_len)
            token_level_rewards = rewards.unsqueeze(-1) * response_masks
        elif len(rewards.shape) == 2:
            # rewards is already (bs, seq_len)
            token_level_rewards = rewards * response_masks
        else:
            raise ValueError(f"Unexpected rewards shape: {rewards.shape}")

        # Compute episode-level rewards (sum over tokens)
        episode_rewards = (token_level_rewards * response_masks).sum(dim=1) / response_masks.sum(dim=1).clamp(min=1)

        print(f"DEBUG compute_advantages_grpo: episode_rewards shape = {episode_rewards.shape}")

        # Group by episode_indices and trajectory_indices for relative comparison
        # Combine indices for grouping
        group_ids = [f"{ep}_{traj}" for ep, traj in zip(episode_indices, trajectory_indices)]
        unique_groups = list(set(group_ids))

        advantages = torch.zeros_like(episode_rewards)

        # Normalize within each group (GRPO core idea)
        for group_id in unique_groups:
            group_mask = torch.tensor([gid == group_id for gid in group_ids], dtype=torch.bool, device=rewards.device)
            group_rewards = episode_rewards[group_mask]

            if len(group_rewards) > 1:
                # Normalize by mean and std within group
                group_mean = group_rewards.mean()
                group_std = group_rewards.std()
                advantages[group_mask] = (group_rewards - group_mean) / (group_std + epsilon)
            else:
                # Single item in group, set advantage to 0
                advantages[group_mask] = 0.0

        # Expand advantages to token level
        advantages = advantages.unsqueeze(-1).expand_as(token_level_rewards)
        returns = advantages  # For GRPO, returns = advantages

        print(f"DEBUG compute_advantages_grpo: advantages shape = {advantages.shape}")

        return advantages, returns

    # REMOVED: GAE advantage computation - using GRPO uniformly
    # def compute_advantages_gae(
    #     self,
    #     rewards: torch.Tensor,
    #     values: torch.Tensor,
    #     dones: torch.Tensor
    # ) -> Tuple[torch.Tensor, torch.Tensor]:
    #     """
    #     Compute advantages using GAE
    #     使用 GAE 计算优势
    #
    #     Args:
    #         rewards: Rewards (T,)
    #         values: Values (T,)
    #         dones: Done flags (T,)
    #
    #     Returns:
    #         Tuple: (advantages, returns)
    #     """
    #     advantages = torch.zeros_like(rewards)
    #     last_gae = 0
    #
    #     for t in reversed(range(len(rewards))):
    #         if t == len(rewards) - 1:
    #             next_value = 0
    #         else:
    #             next_value = values[t + 1]
    #
    #         delta = rewards[t] + self.gamma * next_value * (1 - dones[t].float()) - values[t]
    #         last_gae = delta + self.gamma * self.gae_lambda * (1 - dones[t].float()) * last_gae
    #         advantages[t] = last_gae
    #
    #     returns = advantages + values
    #     return advantages, returns

    def update(self) -> Dict[str, float]:
        """
        Update policy using collected rollouts
        使用收集的 rollouts 更新策略

        Returns:
            Dict: Training statistics
        """
        # Get data from buffer
        data = self.buffer.get()

        if data['log_probs'] is None or len(data['log_probs']) == 0:
            print("[RLTrainer] Warning: Empty buffer, skipping update")
            return {}

        # Move to device (GRPO: no values)
        log_probs_old = data['log_probs'].to(self.device)
        rewards = data['rewards'].to(self.device)
        dones = data['dones'].to(self.device)
        response_masks = data['response_masks'].to(self.device)

        # Compute advantages using standard GRPO
        # NOTE: GiGPO was found to be unstable and has been replaced with GRPO
        print("[RLTrainer] Computing advantages using GRPO")
        advantages, returns = self.compute_advantages_grpo(
            rewards=rewards,
            response_masks=response_masks,
            episode_indices=data['episode_indices'],
            trajectory_indices=data['trajectory_indices']
        )

        advantages = advantages.to(self.device)
        returns = returns.to(self.device)

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO update (GRPO: no value loss)
        update_stats = {
            'policy_loss': 0.0,
            'entropy': 0.0,
            'total_loss': 0.0,
            'approx_kl': 0.0,
            'clip_fraction': 0.0
        }

        # Get stored input_ids and attention_masks for consistent re-computation
        input_ids = data['input_ids']  # Already padded and batched
        attention_masks = data['attention_masks']  # Already padded and batched

        for epoch in range(self.ppo_epochs):
            # ==================================================
            # GRPO PPO UPDATE (No value loss)
            # ==================================================

            # 1. Re-compute log_probs with gradients using stored input_ids
            # This ensures consistent sequence lengths with stored response_masks
            input_ids_device = input_ids.to(self.device)
            attention_masks_device = attention_masks.to(self.device)

            # Batch forward pass (GRPO: no values)
            new_log_probs_list = []
            entropies_list = []

            for i in range(input_ids_device.shape[0]):
                # Forward pass with gradients using stored input_ids
                outputs = self.policy.forward(
                    input_ids=input_ids_device[i:i+1],
                    attention_mask=attention_masks_device[i:i+1],
                    response_mask=response_masks[i:i+1]
                )

                new_log_probs_list.append(outputs['log_probs'])

                # Compute entropy from logits
                logits = outputs['logits'][:, :-1, :]  # Shift for next-token prediction
                probs = F.softmax(logits, dim=-1)
                entropy = -(probs * F.log_softmax(logits, dim=-1)).sum(dim=-1)  # (bs, seq_len-1)
                entropy = F.pad(entropy, (0, 1), value=0.0)  # Pad to match seq_len
                entropy = entropy * response_masks[i:i+1]  # Apply mask
                entropies_list.append(entropy)

            # Stack all outputs
            new_log_probs = torch.cat(new_log_probs_list, dim=0)  # (bs, seq_len)
            new_entropies = torch.cat(entropies_list, dim=0)  # (bs, seq_len)

            # 2. Compute PPO ratio
            # ratio = exp(new_log_probs - old_log_probs)
            ratio = torch.exp(new_log_probs - log_probs_old)  # (bs, seq_len)

            # 3. Compute policy loss with PPO clipping
            # Unclipped policy loss
            policy_loss_unclipped = -advantages * ratio

            # Clipped policy loss
            ratio_clipped = torch.clamp(ratio, 1.0 - self.ppo_clip, 1.0 + self.ppo_clip)
            policy_loss_clipped = -advantages * ratio_clipped

            # Take the maximum (most conservative)
            policy_loss_per_token = torch.max(policy_loss_unclipped, policy_loss_clipped)

            # Apply response mask and average
            policy_loss = (policy_loss_per_token * response_masks).sum() / response_masks.sum()

            # 4. Compute entropy bonus
            entropy = (new_entropies * response_masks).sum() / response_masks.sum()

            # 5. Total loss (GRPO: no value loss)
            total_loss = policy_loss - self.entropy_coef * entropy

            # 7. Compute KL divergence for monitoring
            with torch.no_grad():
                approx_kl = ((ratio - 1) - torch.log(ratio)) * response_masks
                approx_kl = approx_kl.sum() / response_masks.sum()

                # Clip fraction
                clip_fraction = ((ratio < 1.0 - self.ppo_clip) | (ratio > 1.0 + self.ppo_clip)).float() * response_masks
                clip_fraction = clip_fraction.sum() / response_masks.sum()

            # 8. Backpropagation
            self.optimizer.zero_grad()
            total_loss.backward()

            # 9. Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)

            # 10. Update weights
            self.optimizer.step()

            # Record stats
            update_stats['policy_loss'] += policy_loss.item()
            update_stats['entropy'] += entropy.item()
            update_stats['total_loss'] += total_loss.item()
            update_stats['approx_kl'] += approx_kl.item()
            update_stats['clip_fraction'] += clip_fraction.item()

        # Average over epochs
        for key in update_stats:
            update_stats[key] /= self.ppo_epochs

        # Clear buffer
        self.buffer.clear()

        print(f"[RLTrainer] ✅ GRPO Update completed:")
        print(f"  - Policy loss: {update_stats['policy_loss']:.4f}")
        print(f"  - Entropy: {update_stats['entropy']:.4f}")
        print(f"  - Total loss: {update_stats['total_loss']:.4f}")
        print(f"  - Approx KL: {update_stats['approx_kl']:.6f}")
        print(f"  - Clip fraction: {update_stats['clip_fraction']:.4f}")
        print(f"[RLTrainer] 🎉 Policy (Qwen LoRA) updated with GRPO!")

        return update_stats

    def save_checkpoint(self, path: str):
        """Save trainer checkpoint"""
        checkpoint = {
            'optimizer': self.optimizer.state_dict(),
            'train_stats': self.train_stats
        }
        torch.save(checkpoint, path)
        print(f"[RLTrainer] Checkpoint saved to {path}")

    def load_checkpoint(self, path: str):
        """Load trainer checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.train_stats = checkpoint['train_stats']
        print(f"[RLTrainer] Checkpoint loaded from {path}")
