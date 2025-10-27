"""
RL Trainer for End-to-End Training
端到端训练的 RL 训练器

This module implements:
1. Trajectory collection from AFlow environments
2. Advantage computation using workflow-specific GiGPO
3. Policy and value loss computation
4. Gradient updates

此模块实现：
1. 从 AFlow 环境收集轨迹
2. 使用工作流特定 GiGPO 计算优势
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

# Import GiGPO
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'verl-agent'))
from gigpo.workflow_gigpo import compute_workflow_gigpo_advantage


class RolloutBuffer:
    """
    Buffer for storing trajectories
    存储轨迹的缓冲区
    """

    def __init__(self):
        self.observations = []
        self.actions = []
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.dones = []
        self.response_masks = []

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
        value: torch.Tensor,
        reward: float,
        done: bool,
        response_mask: torch.Tensor,
        workflow_node: Optional[str] = None,
        workflow_state: Optional[Any] = None,
        episode_idx: int = 0,
        traj_idx: int = 0
    ):
        """Add a step to the buffer"""
        self.observations.append(obs)
        self.actions.append(action)
        self.log_probs.append(log_prob.detach().cpu())
        self.values.append(value.detach().cpu())
        self.rewards.append(reward)
        self.dones.append(done)
        self.response_masks.append(response_mask.detach().cpu())

        self.workflow_nodes.append(workflow_node)
        self.workflow_states.append(workflow_state)
        self.episode_indices.append(episode_idx)
        self.trajectory_indices.append(traj_idx)

    def clear(self):
        """Clear buffer"""
        self.__init__()

    def get(self) -> Dict[str, Any]:
        """Get all data from buffer"""
        # Handle variable-length sequences by padding
        log_probs_padded = None
        values_padded = None
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

        if self.values:
            # Values should all be same shape already, but handle just in case
            if all(v.shape == self.values[0].shape for v in self.values):
                values_padded = torch.stack(self.values)
            else:
                # Pad if needed
                max_len = max(v.shape[1] if len(v.shape) > 1 else 1 for v in self.values)
                values_list = []
                for v in self.values:
                    if len(v.shape) == 1:
                        v = v.unsqueeze(1)
                    if v.shape[1] < max_len:
                        padding = torch.zeros(v.shape[0], max_len - v.shape[1], dtype=v.dtype)
                        v_padded = torch.cat([v, padding], dim=1)
                    else:
                        v_padded = v
                    values_list.append(v_padded)
                values_padded = torch.stack(values_list)

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

        return {
            'observations': self.observations,
            'actions': self.actions,
            'log_probs': log_probs_padded,
            'values': values_padded,
            'rewards': torch.tensor(self.rewards, dtype=torch.float32),
            'dones': torch.tensor(self.dones, dtype=torch.bool),
            'response_masks': response_masks_padded,
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
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 1.0,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        ppo_epochs: int = 4,
        ppo_clip: float = 0.2,
        batch_size: int = 32,
        use_gigpo: bool = True,
        gigpo_config: Optional[Dict] = None,
        device: str = "cuda"
    ):
        """
        Initialize RL trainer

        Args:
            policy: Trainable policy (TrainableQwenPolicy)
            learning_rate: Learning rate for optimizer
            value_coef: Coefficient for value loss
            entropy_coef: Coefficient for entropy bonus
            max_grad_norm: Max gradient norm for clipping
            gamma: Discount factor
            gae_lambda: GAE lambda for advantage estimation
            ppo_epochs: Number of PPO update epochs
            ppo_clip: PPO clipping parameter
            batch_size: Batch size for updates
            use_gigpo: Use workflow-specific GiGPO
            gigpo_config: Configuration for GiGPO
            device: Training device
        """
        self.policy = policy
        self.device = device

        # Hyperparameters
        self.learning_rate = learning_rate
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.ppo_epochs = ppo_epochs
        self.ppo_clip = ppo_clip
        self.batch_size = batch_size

        # GiGPO
        self.use_gigpo = use_gigpo
        self.gigpo_config = gigpo_config or {
            'epsilon': 1e-6,
            'step_advantage_w': 1.0,
            'mode': 'mean_norm',
            'enable_similarity': True,
            'similarity_thresh': 0.95,
            'workflow_similarity_thresh': 0.8
        }

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.policy.parameters(),
            lr=learning_rate,
            weight_decay=0.01
        )

        # Rollout buffer
        self.buffer = RolloutBuffer()

        # Statistics
        self.train_stats = {
            'policy_loss': [],
            'value_loss': [],
            'entropy': [],
            'total_loss': [],
            'approx_kl': [],
            'clip_fraction': []
        }

        print(f"[RLTrainer] Initialized with:")
        print(f"  - Learning rate: {learning_rate}")
        print(f"  - Value coef: {value_coef}, Entropy coef: {entropy_coef}")
        print(f"  - PPO epochs: {ppo_epochs}, Clip: {ppo_clip}")
        print(f"  - Batch size: {batch_size}")
        print(f"  - Use GiGPO: {use_gigpo}")

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

                # Get actions from policy
                actions = []
                log_probs_list = []
                values_list = []
                response_masks_list = []

                for i, (obs, done) in enumerate(zip(obs_list, done_list)):
                    if not done:
                        action, log_probs, values, response_mask = self.policy.get_action_and_value(obs)

                        actions.append(action)
                        log_probs_list.append(log_probs)
                        values_list.append(values)
                        response_masks_list.append(response_mask)
                    else:
                        actions.append("")
                        log_probs_list.append(None)
                        values_list.append(None)
                        response_masks_list.append(None)

                # Step environment
                next_obs_list, reward_list, done_list, info_list = env.step(actions)

                # Store transitions
                for i in range(len(obs_list)):
                    if log_probs_list[i] is not None:
                        self.buffer.add(
                            obs=obs_list[i],
                            action=actions[i],
                            log_prob=log_probs_list[i],
                            value=values_list[i],
                            reward=reward_list[i],
                            done=done_list[i],
                            response_mask=response_masks_list[i],
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

    def compute_advantages_gigpo(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        response_masks: torch.Tensor,
        workflow_nodes: np.array,
        workflow_states: List,
        episode_indices: np.array,
        trajectory_indices: np.array,
        anchor_obs: np.array
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute advantages using workflow-specific GiGPO
        使用工作流特定 GiGPO 计算优势

        Args:
            rewards: Rewards (bs, seq_len)
            values: Values (bs, seq_len)
            response_masks: Response masks (bs, seq_len)
            workflow_nodes: Workflow node IDs (bs,)
            workflow_states: List of workflow states
            episode_indices: Episode indices (bs,)
            trajectory_indices: Trajectory indices (bs,)
            anchor_obs: Anchor observations for grouping (bs,)

        Returns:
            Tuple: (advantages, returns)
        """
        # Compute token-level rewards (approximation: broadcast step rewards)
        # Handle both (bs,) and (bs, seq_len) reward shapes
        print(f"DEBUG compute_advantages_gigpo: rewards shape = {rewards.shape}, response_masks shape = {response_masks.shape}")

        # Squeeze response_masks if it has extra dimensions
        if len(response_masks.shape) == 3 and response_masks.shape[1] == 1:
            response_masks = response_masks.squeeze(1)
            print(f"DEBUG compute_advantages_gigpo: squeezed response_masks shape = {response_masks.shape}")

        if len(rewards.shape) == 1:
            # rewards is (bs,), expand to (bs, seq_len)
            token_level_rewards = rewards.unsqueeze(-1) * response_masks
        elif len(rewards.shape) == 2:
            # rewards is already (bs, seq_len)
            token_level_rewards = rewards * response_masks
        else:
            raise ValueError(f"Unexpected rewards shape: {rewards.shape}")

        print(f"DEBUG compute_advantages_gigpo: token_level_rewards shape = {token_level_rewards.shape}")

        # Compute step rewards (mean over tokens)
        step_rewards = (token_level_rewards * response_masks).sum(dim=1) / response_masks.sum(dim=1).clamp(min=1)
        print(f"DEBUG compute_advantages_gigpo: step_rewards shape = {step_rewards.shape}")

        # Use GiGPO
        advantages, scores = compute_workflow_gigpo_advantage(
            token_level_rewards=token_level_rewards,
            step_rewards=step_rewards,
            response_mask=response_masks,
            anchor_obs=anchor_obs,
            index=episode_indices,
            traj_index=trajectory_indices,
            workflow_nodes=workflow_nodes,
            workflow_states=workflow_states,
            **self.gigpo_config
        )

        # Compute returns
        returns = advantages + values

        return advantages, returns

    def compute_advantages_gae(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        dones: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute advantages using GAE
        使用 GAE 计算优势

        Args:
            rewards: Rewards (T,)
            values: Values (T,)
            dones: Done flags (T,)

        Returns:
            Tuple: (advantages, returns)
        """
        advantages = torch.zeros_like(rewards)
        last_gae = 0

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]

            delta = rewards[t] + self.gamma * next_value * (1 - dones[t].float()) - values[t]
            last_gae = delta + self.gamma * self.gae_lambda * (1 - dones[t].float()) * last_gae
            advantages[t] = last_gae

        returns = advantages + values
        return advantages, returns

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

        # Move to device
        log_probs_old = data['log_probs'].to(self.device)
        values = data['values'].to(self.device)
        rewards = data['rewards'].to(self.device)
        dones = data['dones'].to(self.device)
        response_masks = data['response_masks'].to(self.device)

        # Compute advantages
        if self.use_gigpo and data['workflow_nodes'] is not None:
            print("[RLTrainer] Computing advantages using workflow-specific GiGPO")
            advantages, returns = self.compute_advantages_gigpo(
                rewards=rewards,
                values=values,
                response_masks=response_masks,
                workflow_nodes=data['workflow_nodes'],
                workflow_states=data['workflow_states'],
                episode_indices=data['episode_indices'],
                trajectory_indices=data['trajectory_indices'],
                anchor_obs=np.array(data['observations'])
            )
        else:
            print("[RLTrainer] Computing advantages using GAE")
            # Flatten for GAE
            values_flat = values.mean(dim=1)  # (bs,)
            advantages, returns = self.compute_advantages_gae(rewards, values_flat, dones)
            # Expand for token-level - match log_probs shape
            advantages = advantages.unsqueeze(-1).expand_as(log_probs_old)
            returns = returns.unsqueeze(-1).unsqueeze(-1).expand_as(values)  # FIX: add extra unsqueeze

        advantages = advantages.to(self.device)
        returns = returns.to(self.device)

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO update
        update_stats = {
            'policy_loss': 0.0,
            'value_loss': 0.0,
            'entropy': 0.0,
            'total_loss': 0.0,
            'approx_kl': 0.0,
            'clip_fraction': 0.0
        }

        # Get observations for re-computing log_probs and values
        observations = data['observations']

        for epoch in range(self.ppo_epochs):
            # ==================================================
            # FULL PPO UPDATE IMPLEMENTATION
            # ==================================================

            # 1. Re-compute log_probs and values with gradients
            # Tokenize all observations
            inputs_list = []
            for obs in observations:
                inputs = self.policy.tokenizer(
                    obs,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=1024
                ).to(self.device)
                inputs_list.append(inputs)

            # Batch forward pass (simple version: process sequentially)
            new_log_probs_list = []
            new_values_list = []
            entropies_list = []

            for i, inputs in enumerate(inputs_list):
                # Forward pass with gradients
                outputs = self.policy.forward(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    response_mask=response_masks[i:i+1]
                )

                new_log_probs_list.append(outputs['log_probs'])
                new_values_list.append(outputs['values'])

                # Compute entropy from logits
                logits = outputs['logits'][:, :-1, :]  # Shift for next-token prediction
                probs = F.softmax(logits, dim=-1)
                entropy = -(probs * F.log_softmax(logits, dim=-1)).sum(dim=-1)  # (bs, seq_len-1)
                entropy = F.pad(entropy, (0, 1), value=0.0)  # Pad to match seq_len
                entropy = entropy * response_masks[i:i+1]  # Apply mask
                entropies_list.append(entropy)

            # Stack all outputs
            new_log_probs = torch.cat(new_log_probs_list, dim=0)  # (bs, seq_len)
            new_values = torch.cat(new_values_list, dim=0)  # (bs, seq_len)
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

            # 4. Compute value loss
            # Clip value predictions for stability
            values_clipped = values + torch.clamp(
                new_values - values,
                -self.ppo_clip,
                self.ppo_clip
            )

            # Value loss (both clipped and unclipped)
            value_loss_unclipped = F.mse_loss(new_values, returns.detach(), reduction='none')
            value_loss_clipped = F.mse_loss(values_clipped, returns.detach(), reduction='none')
            value_loss_per_token = torch.max(value_loss_unclipped, value_loss_clipped)

            # Apply mask and average
            value_loss = (value_loss_per_token * response_masks).sum() / response_masks.sum()

            # 5. Compute entropy bonus
            entropy = (new_entropies * response_masks).sum() / response_masks.sum()

            # 6. Total loss
            total_loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

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
            update_stats['value_loss'] += value_loss.item()
            update_stats['entropy'] += entropy.item()
            update_stats['total_loss'] += total_loss.item()
            update_stats['approx_kl'] += approx_kl.item()
            update_stats['clip_fraction'] += clip_fraction.item()

        # Average over epochs
        for key in update_stats:
            update_stats[key] /= self.ppo_epochs

        # Clear buffer
        self.buffer.clear()

        print(f"[RLTrainer] ✅ PPO Update completed:")
        print(f"  - Policy loss: {update_stats['policy_loss']:.4f}")
        print(f"  - Value loss: {update_stats['value_loss']:.4f}")
        print(f"  - Entropy: {update_stats['entropy']:.4f}")
        print(f"  - Total loss: {update_stats['total_loss']:.4f}")
        print(f"  - Approx KL: {update_stats['approx_kl']:.6f}")
        print(f"  - Clip fraction: {update_stats['clip_fraction']:.4f}")
        print(f"[RLTrainer] 🎉 Policy (Qwen LoRA) and Value Head both updated!")

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
