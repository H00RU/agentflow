# Copyright 2025 Nanyang Technological University (NTU), Singapore
# and the verl-agent (GiGPO) team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from agent_system.environments.env_manager import EnvironmentManagerBase, make_envs


# AFlow integration
import os
import sys

# Add AFlow path
VERL_AGENT_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
INTEGRATION_PATH = os.path.join(os.path.dirname(VERL_AGENT_PATH), 'integration')
AFLOW_PATH = os.path.join(os.path.dirname(VERL_AGENT_PATH), 'AFlow')

sys.path.insert(0, AFLOW_PATH)
sys.path.insert(0, INTEGRATION_PATH)


def make_aflow_envs(config):
    """
    Create AFlow environments for training and validation

    Args:
        config: Training configuration

    Returns:
        envs: Dict of training environments
        val_envs: Dict of validation environments
    """
    from agent_system.environments.env_package.aflow_integrated import build_aflow_envs

    aflow_config = config.get('aflow_env', {})

    # Get training datasets
    train_datasets = aflow_config.get('train_datasets', ['HumanEval'])
    test_datasets = aflow_config.get('test_datasets', ['HumanEval'])

    # Common parameters
    opt_llm_config = aflow_config.get('opt_llm_config')
    exec_llm_config = aflow_config.get('exec_llm_config')
    operators = aflow_config.get('operators', ['Custom', 'CustomCodeGenerate'])
    max_rounds = aflow_config.get('max_rounds', 10)
    validation_rounds = aflow_config.get('validation_rounds', 3)
    sample = aflow_config.get('sample', 3)
    workspace_path = aflow_config.get('workspace_path', '/root/aflow_integration/AFlow/workspace')

    env_num = aflow_config.get('env_num', 4)
    group_n = aflow_config.get('group_n', 2)

    print(f"\n[make_aflow_envs] Creating environments:")
    print(f"  Train datasets: {train_datasets}")
    print(f"  Test datasets: {test_datasets}")
    print(f"  Parallel workers: {env_num} x {group_n} = {env_num * group_n}")
    print(f"  Max rounds: {max_rounds}")
    print(f"  Operators: {operators}\n")

    # Create training environments
    train_envs = {}
    for dataset in train_datasets:
        print(f"[make_aflow_envs] Creating training environment for {dataset}...")

        question_type = _get_question_type(dataset)

        env = build_aflow_envs(
            dataset=dataset,
            question_type=question_type,
            opt_llm_config=opt_llm_config,
            exec_llm_config=exec_llm_config,
            operators=operators,
            sample=sample,
            optimized_path=workspace_path,
            max_rounds=max_rounds,
            validation_rounds=validation_rounds,
            seed=42,
            env_num=env_num,
            group_n=group_n,
            resources_per_worker={'num_cpus': 0.5, 'num_gpus': 0.0},  # Reduced CPU to leave resources for ActorRollout
            is_train=True
        )

        train_envs[dataset] = env
        print(f"  ✓ Created training env for {dataset}")

    # Create test environments
    test_envs = {}
    for dataset in test_datasets:
        print(f"[make_aflow_envs] Creating test environment for {dataset}...")

        question_type = _get_question_type(dataset)

        env = build_aflow_envs(
            dataset=dataset,
            question_type=question_type,
            opt_llm_config=opt_llm_config,
            exec_llm_config=exec_llm_config,
            operators=operators,
            sample=sample,
            optimized_path=workspace_path,
            max_rounds=max_rounds,
            validation_rounds=validation_rounds,
            seed=100,
            env_num=max(env_num // 2, 1),
            group_n=group_n,
            resources_per_worker={'num_cpus': 0.5, 'num_gpus': 0.0},  # Reduced CPU to leave resources for ActorRollout
            is_train=False
        )

        test_envs[dataset] = env
        print(f"  ✓ Created test env for {dataset}")

    print(f"\n[make_aflow_envs] Environment creation complete!")
    print(f"  Training envs: {list(train_envs.keys())}")
    print(f"  Test envs: {list(test_envs.keys())}\n")

    return train_envs, test_envs


def _get_question_type(dataset: str) -> str:
    """
    Map dataset name to question type

    Args:
        dataset: Dataset name

    Returns:
        str: Question type
    """
    if dataset in ['HumanEval', 'MBPP']:
        return 'code'
    elif dataset in ['GSM8K', 'MATH']:
        return 'math'
    else:
        return 'qa'