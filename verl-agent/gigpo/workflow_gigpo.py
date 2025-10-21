"""
Workflow-Specific GiGPO Implementation
工作流特化的 GiGPO 实现

This module extends GiGPO for workflow optimization by:
1. Mapping MCTS nodes to episode-level groups
2. Using workflow similarity for step-level groups
3. Integrating with WorkflowState from AFlow
4. Providing workflow-aware advantage computation

Key innovations:
- Episode groups = MCTS nodes (same parent workflow)
- Step groups = similar workflow states (operators, structure)
- Advantage computation considers workflow performance
"""

import numpy as np
import torch
from collections import defaultdict, Counter
from typing import List, Dict, Any, Optional, Tuple
import sys
import os

# Import from core GiGPO
from gigpo.core_gigpo import (
    to_hashable,
    summarize_group_size,
    are_similar
)

# Try to import WorkflowState
try:
    INTEGRATION_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'integration'))
    sys.path.insert(0, INTEGRATION_PATH)
    from unified_state import WorkflowState
    WORKFLOW_STATE_AVAILABLE = True
except ImportError:
    WORKFLOW_STATE_AVAILABLE = False
    WorkflowState = None


def compute_workflow_gigpo_advantage(
    token_level_rewards: torch.Tensor,
    step_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    anchor_obs: np.array,
    index: np.array,
    traj_index: np.array,
    workflow_nodes: Optional[np.array] = None,
    workflow_states: Optional[List] = None,
    epsilon: float = 1e-6,
    step_advantage_w: float = 1.0,
    mode: str = "mean_norm",
    enable_similarity: bool = False,
    similarity_thresh: float = 0.95,
    workflow_similarity_thresh: float = 0.8,
):
    """
    Compute advantages for GiGPO with workflow-specific grouping
    使用工作流特定分组的 GiGPO 优势计算

    Key differences from standard GiGPO:
    1. Episode groups use MCTS node information
    2. Step groups consider workflow structure similarity
    3. Advantages weighted by workflow performance

    Args:
        token_level_rewards: Token-level rewards (bs, response_length)
        step_rewards: Step-level rewards (bs,)
        response_mask: Mask for response tokens (bs, response_length)
        anchor_obs: Anchor observations for grouping (bs,)
        index: Episode group indices (bs,)
        traj_index: Trajectory indices (bs,)
        workflow_nodes: MCTS node IDs for each sample (bs,), optional
        workflow_states: List of WorkflowState objects, optional
        epsilon: Small value to avoid division by zero
        step_advantage_w: Weight for step-level advantages
        mode: Normalization mode ("mean_norm" or "mean_std_norm")
        enable_similarity: Enable similarity-based grouping
        similarity_thresh: Threshold for text similarity
        workflow_similarity_thresh: Threshold for workflow similarity

    Returns:
        Tuple: (advantages, scores)
    """
    if mode == "mean_std_norm":
        remove_std = False
    elif mode == "mean_norm":
        remove_std = True
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # Compute episode relative advantages with MCTS node grouping
    if workflow_nodes is not None:
        episode_advantages = compute_episode_advantage_by_node(
            token_level_rewards=token_level_rewards,
            response_mask=response_mask,
            index=index,
            traj_index=traj_index,
            workflow_nodes=workflow_nodes,
            epsilon=epsilon,
            remove_std=remove_std
        )
    else:
        # Fallback to standard episode advantage
        episode_advantages = episode_norm_reward(
            token_level_rewards=token_level_rewards,
            response_mask=response_mask,
            index=index,
            traj_index=traj_index,
            epsilon=epsilon,
            remove_std=remove_std
        )

    # Build step groups with workflow awareness
    if workflow_states is not None and WORKFLOW_STATE_AVAILABLE:
        step_group_uids = build_workflow_step_group(
            anchor_obs=anchor_obs,
            index=index,
            workflow_states=workflow_states,
            workflow_nodes=workflow_nodes,
            enable_similarity=enable_similarity,
            similarity_thresh=workflow_similarity_thresh
        )
    else:
        # Fallback to standard step grouping
        step_group_uids = build_step_group_standard(
            anchor_obs=anchor_obs,
            index=index,
            enable_similarity=enable_similarity,
            similarity_thresh=similarity_thresh
        )

    # Compute step relative advantages
    step_advantages = step_norm_reward(
        step_rewards=step_rewards,
        response_mask=response_mask,
        index=step_group_uids,
        epsilon=epsilon,
        remove_std=remove_std
    )

    # Ensure dimensions match before combining
    # episode_advantages: (bs, response_length)
    # step_advantages: (bs, response_length)
    # They should have the same response_length
    print(f"DEBUG: Before alignment - episode_advantages: {episode_advantages.shape}, step_advantages: {step_advantages.shape}")

    # Always check and align the last dimension
    target_length = episode_advantages.shape[-1]
    if step_advantages.shape[-1] != target_length:
        print(f"Warning: Shape mismatch - episode_advantages: {episode_advantages.shape}, step_advantages: {step_advantages.shape}")
        # Truncate or pad step_advantages
        if step_advantages.shape[-1] > target_length:
            step_advantages = step_advantages[:, :target_length]
            print(f"Truncated step_advantages to {step_advantages.shape}")
        else:
            padding = torch.zeros(
                step_advantages.shape[0],
                target_length - step_advantages.shape[-1],
                device=step_advantages.device,
                dtype=step_advantages.dtype
            )
            step_advantages = torch.cat([step_advantages, padding], dim=-1)
            print(f"Padded step_advantages to {step_advantages.shape}")

    # Compute joint advantages
    scores = episode_advantages + step_advantage_w * step_advantages

    return scores, scores


def compute_episode_advantage_by_node(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.array,
    traj_index: np.array,
    workflow_nodes: np.array,
    epsilon: float = 1e-6,
    remove_std: bool = True,
):
    """
    Compute episode-level advantage with MCTS node grouping
    使用 MCTS 节点分组计算 episode 级别优势

    Trajectories from the same MCTS node are grouped together,
    enabling credit assignment to workflow design choices.

    Args:
        token_level_rewards: Token-level rewards (bs, response_length)
        response_mask: Response mask (bs, response_length)
        index: Episode group indices (bs,)
        traj_index: Trajectory indices (bs,)
        workflow_nodes: MCTS node IDs (bs,)
        epsilon: Small value to avoid division by zero
        remove_std: Whether to remove std from normalization

    Returns:
        torch.Tensor: Episode advantages (bs, response_length)
    """
    # Use response_mask shape to get correct response_length
    response_length = response_mask.shape[-1]
    scores = token_level_rewards.sum(dim=-1)

    # Group by (index, workflow_node) instead of just index
    # This ensures samples from same MCTS node are grouped
    node_to_score = defaultdict(list)
    node_to_mean = {}
    node_to_std = {}

    with torch.no_grad():
        bsz = scores.shape[0]

        # Collect scores for each (index, node) group
        for i in range(bsz):
            group_key = (index[i], workflow_nodes[i])
            node_to_score[group_key].append(scores[i])

        # Compute mean and std for each group
        for group_key in node_to_score:
            if len(node_to_score[group_key]) == 1:
                node_to_mean[group_key] = torch.tensor(0.0)
                node_to_std[group_key] = torch.tensor(1.0)
            elif len(node_to_score[group_key]) > 1:
                node_to_mean[group_key] = torch.mean(torch.stack(node_to_score[group_key]))
                node_to_std[group_key] = torch.std(torch.stack(node_to_score[group_key]))
            else:
                raise ValueError(f"No score in group: {group_key}")

        # Normalize scores
        for i in range(bsz):
            group_key = (index[i], workflow_nodes[i])
            if remove_std:
                scores[i] = scores[i] - node_to_mean[group_key]
            else:
                scores[i] = (scores[i] - node_to_mean[group_key]) / (node_to_std[group_key] + epsilon)

        # Broadcast to token level
        episode_advantages = scores.unsqueeze(-1).tile([1, response_length]) * response_mask

    return episode_advantages


def build_workflow_step_group(
    anchor_obs: np.array,
    index: np.array,
    workflow_states: List,
    workflow_nodes: Optional[np.array] = None,
    enable_similarity: bool = False,
    similarity_thresh: float = 0.8,
    summarize: bool = False
) -> np.array:
    """
    Build step groups with workflow structure awareness
    构建具有工作流结构意识的步骤分组

    Groups states that have:
    1. Same episode index
    2. Similar MCTS node (if available)
    3. Similar workflow structure (operators, graph)

    Args:
        anchor_obs: Anchor observations (bs,)
        index: Episode group indices (bs,)
        workflow_states: List of WorkflowState objects
        workflow_nodes: MCTS node IDs (bs,), optional
        enable_similarity: Enable similarity-based grouping
        similarity_thresh: Threshold for workflow similarity
        summarize: Whether to summarize group sizes

    Returns:
        np.array: Step group UIDs (bs,)
    """
    if not WORKFLOW_STATE_AVAILABLE or workflow_states is None:
        # Fallback to standard grouping
        return build_step_group_standard(
            anchor_obs=anchor_obs,
            index=index,
            enable_similarity=enable_similarity,
            similarity_thresh=similarity_thresh,
            summarize=summarize
        )

    import uuid

    step_group_uids = np.empty(len(anchor_obs), dtype=object)
    unique_indices = np.unique(index)
    group_size: List[int] = []

    for idx in unique_indices:
        # Get all samples for this episode index
        indices = np.where(index == idx)[0]

        if not enable_similarity:
            # Group by exact workflow structure match
            clusters = defaultdict(list)

            for i in indices:
                if i < len(workflow_states) and workflow_states[i] is not None:
                    state = workflow_states[i]
                    # Create key from operators and node
                    key_parts = [
                        tuple(sorted(state.operators)),
                        state.parent_node_id if hasattr(state, 'parent_node_id') else None,
                        len(state.operators)
                    ]
                    key = to_hashable(key_parts)
                else:
                    # Fallback to observation
                    key = to_hashable(anchor_obs[i])

                clusters[key].append(i)

            # Assign UIDs
            for cluster_indices in clusters.values():
                uid = str(uuid.uuid4())
                group_size.append(len(cluster_indices))
                for original_idx in cluster_indices:
                    step_group_uids[original_idx] = uid

        else:
            # Similarity-based grouping for workflows
            clusters: List[Dict[str, Any]] = []

            for i in indices:
                if i < len(workflow_states) and workflow_states[i] is not None:
                    state = workflow_states[i]

                    # Try to place into existing cluster
                    placed = False
                    for cluster in clusters:
                        if are_workflows_similar(
                            state,
                            cluster["rep_state"],
                            similarity_thresh
                        ):
                            cluster["locs"].append(i)
                            placed = True
                            break

                    # Create new cluster if not placed
                    if not placed:
                        clusters.append({
                            "rep_state": state,
                            "locs": [i]
                        })
                else:
                    # Fallback: create single-element cluster
                    clusters.append({
                        "rep_state": None,
                        "locs": [i]
                    })

            # Assign UIDs
            for cluster in clusters:
                uid = str(uuid.uuid4())
                group_size.append(len(cluster["locs"]))
                for loc in cluster["locs"]:
                    step_group_uids[loc] = uid

    # Validate
    if None in step_group_uids or np.any(step_group_uids == None):
        missing_indices = np.where(step_group_uids == None)[0]
        raise ValueError(f"Failed to assign UIDs. Missing at indices: {missing_indices}")

    if summarize:
        summarize_group_size(group_size)

    print(f"Workflow step-level group: avg size = {np.mean(group_size):.2f}")

    return step_group_uids


def are_workflows_similar(
    state1: 'WorkflowState',
    state2: 'WorkflowState',
    threshold: float = 0.8
) -> bool:
    """
    Check if two workflow states are similar
    检查两个工作流状态是否相似

    Considers:
    1. Operator overlap (Jaccard similarity)
    2. Performance similarity (score difference)
    3. Structure similarity (graph code)

    Args:
        state1: First workflow state
        state2: Second workflow state
        threshold: Similarity threshold

    Returns:
        bool: True if similar
    """
    if state1 is None or state2 is None:
        return False

    # Operator similarity (Jaccard)
    ops1 = set(state1.operators)
    ops2 = set(state2.operators)

    if len(ops1) == 0 and len(ops2) == 0:
        operator_sim = 1.0
    elif len(ops1.union(ops2)) == 0:
        operator_sim = 0.0
    else:
        operator_sim = len(ops1.intersection(ops2)) / len(ops1.union(ops2))

    # Parent similarity (same parent = more similar)
    parent_sim = 1.0 if state1.parent_node_id == state2.parent_node_id else 0.5

    # Score similarity (closer scores = more similar)
    score_diff = abs(state1.score - state2.score)
    score_sim = 1.0 - min(score_diff, 1.0)

    # Combined similarity
    combined_sim = 0.5 * operator_sim + 0.3 * parent_sim + 0.2 * score_sim

    return combined_sim >= threshold


def build_step_group_standard(
    anchor_obs: np.array,
    index: np.array,
    enable_similarity: bool = False,
    similarity_thresh: float = 0.95,
    summarize: bool = False
) -> np.array:
    """
    Standard step group building (fallback)
    标准步骤分组构建（后备方案）

    This is the standard implementation from core_gigpo,
    used when workflow-specific information is not available.

    Args:
        anchor_obs: Anchor observations
        index: Episode indices
        enable_similarity: Enable similarity-based grouping
        similarity_thresh: Similarity threshold
        summarize: Whether to summarize

    Returns:
        np.array: Step group UIDs
    """
    import uuid

    step_group_uids = np.empty(len(anchor_obs), dtype=object)
    unique_indices = np.unique(index)
    group_size: List[int] = []

    for idx in unique_indices:
        indices = np.where(index == idx)[0]
        obs_group = anchor_obs[indices]

        if not enable_similarity:
            # Exact match clustering
            clusters = defaultdict(list)
            for i, obs in enumerate(obs_group):
                clusters[to_hashable(obs)].append(indices[i])

            for cluster_indices in clusters.values():
                uid = str(uuid.uuid4())
                group_size.append(len(cluster_indices))
                for original_idx in cluster_indices:
                    step_group_uids[original_idx] = uid

        else:
            # Similarity-based clustering
            clusters: List[Dict[str, Any]] = []

            for obs, loc in zip(obs_group, indices):
                placed = False
                for cluster in clusters:
                    if are_similar(obs, cluster["rep"], similarity_thresh):
                        cluster["locs"].append(loc)
                        placed = True
                        break

                if not placed:
                    clusters.append({"rep": obs, "locs": [loc]})

            for cluster in clusters:
                uid = str(uuid.uuid4())
                group_size.append(len(cluster["locs"]))
                for loc in cluster["locs"]:
                    step_group_uids[loc] = uid

    if None in step_group_uids or np.any(step_group_uids == None):
        missing_indices = np.where(step_group_uids == None)[0]
        raise ValueError(f"Failed to assign UIDs. Missing at indices: {missing_indices}")

    if summarize:
        summarize_group_size(group_size)

    return step_group_uids


def episode_norm_reward(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.array,
    traj_index: np.array,
    epsilon: float = 1e-6,
    remove_std: bool = True,
):
    """
    Standard episode-level advantage (from core_gigpo)
    标准 episode 级别优势（来自 core_gigpo）

    This is kept for compatibility and fallback.
    """
    response_length = token_level_rewards.shape[-1]
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}
    seen_pairs = set()

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            if (index[i], traj_index[i]) in seen_pairs:
                continue
            id2score[index[i]].append(scores[i])
            seen_pairs.add((index[i], traj_index[i]))

        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
                id2std[idx] = torch.tensor(1.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.stack(id2score[idx]))
                id2std[idx] = torch.std(torch.stack([id2score[idx]]))
            else:
                raise ValueError(f"No score in prompt index: {idx}")

        for i in range(bsz):
            if remove_std:
                scores[i] = scores[i] - id2mean[index[i]]
            else:
                scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)

        episode_advantages = scores.unsqueeze(-1).tile([1, response_length]) * response_mask

    return episode_advantages


def step_norm_reward(
    step_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.array,
    epsilon: float = 1e-6,
    remove_std: bool = True,
):
    """
    Standard step-level advantage (from core_gigpo)
    标准步骤级别优势（来自 core_gigpo）

    This is kept for compatibility and fallback.
    """
    response_length = response_mask.shape[-1]
    scores = step_rewards.clone()

    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])

        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.mean(torch.stack(id2score[idx]))
                id2std[idx] = torch.tensor(1.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.stack(id2score[idx]))
                id2std[idx] = torch.std(torch.stack([id2score[idx]]))
            else:
                raise ValueError(f"No score in prompt index: {idx}")

        for i in range(bsz):
            if remove_std:
                scores[i] = scores[i] - id2mean[index[i]]
            else:
                scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)

        step_advantages = scores.unsqueeze(-1).tile([1, response_length]) * response_mask

    return step_advantages


def extract_workflow_info_from_batch(batch: Any) -> Tuple[Optional[np.array], Optional[List]]:
    """
    Extract workflow-specific information from batch
    从批次中提取工作流特定信息

    Args:
        batch: Training batch (DataProto or similar)

    Returns:
        Tuple: (workflow_nodes, workflow_states)
    """
    workflow_nodes = None
    workflow_states = None

    # Try to extract from batch
    if hasattr(batch, 'non_tensor_batch'):
        ntb = batch.non_tensor_batch

        if 'workflow_nodes' in ntb:
            workflow_nodes = ntb['workflow_nodes']

        if 'workflow_states' in ntb:
            workflow_states = ntb['workflow_states']

    return workflow_nodes, workflow_states
