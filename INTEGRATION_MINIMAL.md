# 最小化修改的 verl-agent + AFlow 集成方案

## 设计原则

✅ **保持 AFlow 核心不变** - 不修改 optimizer.py, evaluator.py 等核心文件
✅ **保持 verl-agent 核心不变** - 不修改 ray_trainer.py, rollout_loop.py 等
✅ **创建适配层** - 在两者之间建立轻量级接口

## 核心理念

```
┌─────────────────────────────────────────────────────────────┐
│                      适配层架构                               │
│                                                              │
│  verl-agent (RL框架)        Adapter         AFlow (工作流)   │
│  ┌────────────┐           ┌──────┐         ┌────────────┐  │
│  │   Actor    │──观察────>│      │─提示──>│ Optimizer  │  │
│  │  (Policy)  │           │Adapter│         │  (MCTS)    │  │
│  │            │<──状态────│      │<─结果──│            │  │
│  └────────────┘           └──────┘         └────────────┘  │
│                               │                             │
│                          转换动作                            │
│                          提取状态                            │
│                          计算奖励                            │
└─────────────────────────────────────────────────────────────┘
```

---

# 一、文件组织（新增文件）

```
agent worflow/
├── AFlow/                          # 保持不变
│   ├── run.py
│   ├── scripts/
│   └── ...
│
├── verl-agent/                     # 保持不变
│   ├── verl/
│   ├── agent_system/
│   └── ...
│
└── integration/                    # 新增：适配层
    ├── __init__.py
    ├── aflow_wrapper.py           # AFlow 包装器
    ├── workflow_env.py            # Gym 环境适配
    ├── env_manager.py             # verl-agent 环境管理器
    ├── prompts.py                 # 提示词模板
    ├── reward_fn.py               # 奖励函数
    ├── config.yaml                # 配置文件
    └── run_training.sh            # 训练脚本
```

---

# 二、核心组件实现

## 2.1 AFlow 包装器（零修改 AFlow）

```python
# integration/aflow_wrapper.py

"""
AFlow 包装器 - 将 AFlow 的功能包装成可调用的接口
完全不修改 AFlow 的源代码
"""

import sys
import os
import asyncio
from typing import Dict, List, Optional
from pathlib import Path

# 动态添加 AFlow 到 Python 路径
AFLOW_PATH = Path(__file__).parent.parent / "AFlow"
sys.path.insert(0, str(AFLOW_PATH))

# 导入 AFlow 模块（不修改它们）
from scripts.optimizer import Optimizer
from scripts.async_llm import LLMsConfig, create_llm_instance
from scripts.evaluator import Evaluator
from data.download_data import download


class AFlowWrapper:
    """
    包装 AFlow，提供统一接口供 RL agent 使用
    """

    def __init__(
        self,
        dataset: str = "GSM8K",
        optimized_path: str = "workspace_rl",
        sample: int = 4,
        operators: List[str] = None,
        exec_model_name: str = "gpt-4o-mini"
    ):
        """
        Args:
            dataset: 数据集名称
            optimized_path: 工作流保存路径
            sample: 采样数量
            operators: 可用操作符
            exec_model_name: 执行模型名称
        """
        self.dataset = dataset
        self.optimized_path = optimized_path
        self.sample = sample

        # 数据集类型映射
        self.question_type_map = {
            "GSM8K": "math",
            "MATH": "math",
            "HumanEval": "code",
            "MBPP": "code",
            "HotpotQA": "qa",
            "DROP": "qa"
        }
        self.question_type = self.question_type_map.get(dataset, "qa")

        # 默认操作符
        if operators is None:
            if self.question_type == "math":
                self.operators = ["Custom", "ScEnsemble", "Programmer"]
            elif self.question_type == "code":
                self.operators = ["Custom", "CustomCodeGenerate", "ScEnsemble", "Test"]
            else:
                self.operators = ["Custom", "AnswerGenerate", "ScEnsemble"]
        else:
            self.operators = operators

        # 加载 LLM 配置（使用 AFlow 的配置系统）
        models_config = LLMsConfig.default()
        self.exec_llm_config = models_config.get(exec_model_name)

        # 下载数据集（只下载一次）
        download(["datasets"], force_download=False)

        # 初始化评估器
        self.evaluator = Evaluator(eval_path=self.optimized_path)

        # 当前状态
        self.current_round = 1
        self.workflow_history = []

    def reset(self) -> Dict:
        """
        重置环境，生成初始 workflow

        Returns:
            state: {
                'round': int,
                'workflow': dict,
                'score': float,
                'history': list
            }
        """
        self.current_round = 1
        self.workflow_history = []

        # 创建初始 workflow（直接复制 AFlow 的 round_1）
        # 不需要修改 AFlow，只需确保 round_1 存在
        initial_workflow = self._load_workflow(round_num=1)
        initial_score = self._evaluate_workflow(initial_workflow)

        self.workflow_history.append({
            'round': 1,
            'workflow': initial_workflow,
            'score': initial_score
        })

        return {
            'round': 1,
            'workflow': initial_workflow,
            'score': initial_score,
            'history': []
        }

    def step(self, modification: str) -> Dict:
        """
        执行一步修改

        Args:
            modification: RL agent 提供的修改建议（自然语言）

        Returns:
            next_state: dict
            reward: float
            done: bool
            info: dict
        """
        # Step 1: 使用 AFlow 的 LLM 生成新 workflow
        # 这里我们手动调用 AFlow 的组件，而不修改它们
        new_workflow = asyncio.run(
            self._generate_workflow_with_modification(modification)
        )

        # Step 2: 评估新 workflow
        try:
            new_score = self._evaluate_workflow(new_workflow)
            eval_success = True
        except Exception as e:
            print(f"Evaluation failed: {e}")
            new_score = 0.0
            eval_success = False

        # Step 3: 计算奖励
        old_score = self.workflow_history[-1]['score']
        reward = self._compute_reward(old_score, new_score, eval_success)

        # Step 4: 更新状态
        self.current_round += 1
        self.workflow_history.append({
            'round': self.current_round,
            'workflow': new_workflow,
            'score': new_score,
            'modification': modification
        })

        # Step 5: 判断是否结束
        done = (
            self.current_round >= 20 or  # 最多20轮
            new_score >= 0.95 or         # 达到目标分数
            (self.current_round > 5 and new_score < 0.1)  # 早期失败
        )

        # Step 6: 构建返回值
        next_state = {
            'round': self.current_round,
            'workflow': new_workflow,
            'score': new_score,
            'history': self.workflow_history[-5:]  # 最近5步历史
        }

        info = {
            'eval_success': eval_success,
            'improvement': new_score - old_score,
            'best_score': max(h['score'] for h in self.workflow_history)
        }

        return next_state, reward, done, info

    def _load_workflow(self, round_num: int) -> Dict:
        """
        加载指定轮次的 workflow
        直接读取 AFlow 生成的文件，不修改 AFlow
        """
        workflow_path = os.path.join(
            self.optimized_path,
            self.dataset,
            "workflows",
            f"round_{round_num}"
        )

        # 读取 graph.py
        graph_file = os.path.join(workflow_path, "graph.py")
        with open(graph_file, 'r') as f:
            graph_code = f.read()

        # 读取 prompt.py
        prompt_file = os.path.join(workflow_path, "prompt.py")
        with open(prompt_file, 'r') as f:
            prompt_code = f.read()

        return {
            'graph': graph_code,
            'prompt': prompt_code,
            'round': round_num
        }

    async def _generate_workflow_with_modification(self, modification: str) -> Dict:
        """
        根据修改建议生成新 workflow
        使用 AFlow 的 LLM，但不调用完整的 Optimizer
        """
        # 获取当前 workflow
        current_workflow = self.workflow_history[-1]['workflow']
        current_score = self.workflow_history[-1]['score']

        # 构建优化提示（使用 AFlow 风格的提示）
        from scripts.prompts.optimize_prompt import WORKFLOW_OPTIMIZE_PROMPT

        optimization_prompt = f"""
Current Workflow (Score: {current_score:.4f}):
{current_workflow['graph']}

Suggested Modification:
{modification}

Please generate an improved workflow based on this suggestion.
Output the new workflow code in <graph></graph> tags.
"""

        # 调用 AFlow 的 LLM（不修改 AFlow 的代码）
        exec_llm = create_llm_instance(self.exec_llm_config)

        try:
            response = await exec_llm(optimization_prompt)

            # 解析响应
            import re
            graph_match = re.search(r'<graph>(.*?)</graph>', response, re.DOTALL)

            if graph_match:
                new_graph = graph_match.group(1).strip()
            else:
                # 如果没有标签，使用整个响应
                new_graph = response

            return {
                'graph': new_graph,
                'prompt': current_workflow['prompt'],  # 暂时保持不变
                'round': self.current_round + 1
            }

        except Exception as e:
            print(f"Workflow generation failed: {e}")
            # 返回当前 workflow（不变）
            return current_workflow

    def _evaluate_workflow(self, workflow: Dict) -> float:
        """
        评估 workflow 性能
        使用 AFlow 的 Evaluator，不修改它
        """
        # 将 workflow 保存到临时文件
        temp_round = 9999  # 使用特殊的轮次号
        temp_path = os.path.join(
            self.optimized_path,
            self.dataset,
            "workflows",
            f"round_{temp_round}"
        )
        os.makedirs(temp_path, exist_ok=True)

        # 写入文件
        with open(os.path.join(temp_path, "graph.py"), 'w') as f:
            f.write(workflow['graph'])

        with open(os.path.join(temp_path, "prompt.py"), 'w') as f:
            f.write(workflow['prompt'])

        with open(os.path.join(temp_path, "__init__.py"), 'w') as f:
            f.write("")

        # 动态加载 workflow 类
        from scripts.optimizer_utils.graph_utils import GraphUtils
        graph_utils = GraphUtils(f"{self.optimized_path}/{self.dataset}")

        try:
            graph_class = graph_utils.load_graph(
                temp_round,
                f"{self.optimized_path}/{self.dataset}/workflows"
            )

            # 使用 AFlow 的评估器
            params = {
                'llm_config': self.exec_llm_config,
                'dataset': {}
            }

            score, _, _ = asyncio.run(
                self.evaluator.graph_evaluate(
                    dataset=self.dataset,
                    graph=graph_class,
                    params=params,
                    path=temp_path,
                    is_test=False
                )
            )

            return score

        except Exception as e:
            print(f"Evaluation error: {e}")
            return 0.0

    def _compute_reward(
        self,
        old_score: float,
        new_score: float,
        eval_success: bool
    ) -> float:
        """计算奖励"""
        if not eval_success:
            return -1.0  # 评估失败大惩罚

        improvement = new_score - old_score

        # 获取历史最佳
        best_score = max(h['score'] for h in self.workflow_history)

        if new_score > best_score:
            # 突破历史最佳
            reward = improvement * 10 + 2.0
        elif improvement > 0:
            # 性能提升
            reward = improvement * 10
        elif improvement == 0:
            # 无变化
            reward = -0.1
        else:
            # 性能下降
            reward = improvement * 10 - 0.5

        return reward
```

---

## 2.2 Gym 环境适配器

```python
# integration/workflow_env.py

"""
Gym 环境接口 - 连接 AFlowWrapper 和 verl-agent
"""

import gym
import numpy as np
from typing import Dict, Tuple
from .aflow_wrapper import AFlowWrapper


class WorkflowDesignEnv(gym.Env):
    """
    Workflow 设计环境（Gym 接口）
    """

    def __init__(self, dataset: str = "GSM8K", **kwargs):
        super().__init__()

        # 创建 AFlow 包装器
        self.aflow = AFlowWrapper(dataset=dataset, **kwargs)

        # 定义观察空间和动作空间（文本形式）
        # 实际上 verl-agent 不使用这些，但 gym 需要定义
        self.observation_space = gym.spaces.Dict({
            'text': gym.spaces.Text(max_length=10000),
            'score': gym.spaces.Box(low=0, high=1, shape=(1,))
        })

        self.action_space = gym.spaces.Text(max_length=2000)

    def reset(self):
        """重置环境"""
        state = self.aflow.reset()
        obs = self._state_to_obs(state)
        info = {'dataset': self.aflow.dataset}
        return obs, info

    def step(self, action: str):
        """执行动作"""
        # action 是 RL agent 生成的修改建议（自然语言）
        next_state, reward, done, info = self.aflow.step(action)

        next_obs = self._state_to_obs(next_state)

        return next_obs, reward, done, info

    def _state_to_obs(self, state: Dict) -> Dict:
        """
        将 AFlow 状态转换为观察
        """
        # 构建文本观察
        text_obs = f"""Workflow Design Task - Dataset: {self.aflow.dataset}

Current Round: {state['round']}
Current Score: {state['score']:.4f}

Current Workflow:
{state['workflow']['graph']}

Recent History:
"""
        for entry in state['history']:
            text_obs += f"\nRound {entry['round']}: Score = {entry['score']:.4f}"

        text_obs += f"""

Your task: Suggest a modification to improve the workflow.
Available operators: {', '.join(self.aflow.operators)}

Format your suggestion clearly and concisely.
"""

        return {
            'text': text_obs,
            'score': np.array([state['score']], dtype=np.float32),
            'anchor': {
                'round': state['round'],
                'score': state['score'],
                'workflow_hash': hash(state['workflow']['graph'])
            }
        }
```

---

## 2.3 verl-agent 环境管理器（最小修改）

```python
# integration/env_manager.py

"""
verl-agent 环境管理器
遵循 verl-agent 的接口规范
"""

import sys
from pathlib import Path

# 添加 verl-agent 到路径
VERL_PATH = Path(__file__).parent.parent / "verl-agent"
sys.path.insert(0, str(VERL_PATH))

from agent_system.environments.base import EnvironmentManagerBase, to_numpy
from agent_system.memory import SimpleMemory
import numpy as np
from typing import List, Dict, Any

# 导入我们的环境
from .workflow_env import WorkflowDesignEnv


class WorkflowDesignEnvironmentManager(EnvironmentManagerBase):
    """
    Workflow 设计环境管理器
    适配 verl-agent 的接口
    """

    def __init__(self, envs, projection_f, config):
        self.memory = SimpleMemory()
        super().__init__(envs, projection_f, config)

    def reset(self, kwargs) -> Dict[str, Any]:
        """重置环境"""
        # 调用底层环境
        obs_list, infos = self.envs.reset()

        # 初始化记忆
        self.memory.reset(batch_size=len(obs_list))

        # 构建观察
        observations = {
            'text': [obs['text'] for obs in obs_list],
            'image': None,
            'anchor': [obs['anchor'] for obs in obs_list]
        }

        return observations, infos

    def step(self, text_actions: List[str]):
        """执行动作"""
        # text_actions 是 RL agent 生成的修改建议列表

        # 解析和验证动作（简单检查）
        actions, valids = self.projection_f(text_actions)

        # 环境步进
        next_obs_list, rewards, dones, infos = self.envs.step(actions)

        # 存储历史
        self.memory.store({
            'observation': [obs['text'] for obs in next_obs_list],
            'action': text_actions
        })

        # 构建返回值
        next_observations = {
            'text': [obs['text'] for obs in next_obs_list],
            'image': None,
            'anchor': [obs['anchor'] for obs in next_obs_list]
        }

        # 添加动作有效性
        for i, info in enumerate(infos):
            info['is_action_valid'] = valids[i]

        rewards = to_numpy(rewards)
        dones = to_numpy(dones)

        return next_observations, rewards, dones, infos

    def success_evaluator(self, total_infos, total_batch_list,
                         episode_rewards, episode_lengths):
        """评估成功率"""
        from collections import defaultdict
        success = defaultdict(list)

        for batch_idx in range(len(total_batch_list)):
            # 获取最终信息
            final_info = total_infos[batch_idx][-1]

            # 计算改进率
            best_score = final_info.get('best_score', 0)
            success['success_rate'].append(float(best_score > 0.7))
            success['best_score'].append(best_score)

        return success


def workflow_design_projection(text_actions: List[str]) -> tuple:
    """
    动作投影函数
    简单验证动作有效性
    """
    actions = []
    valids = []

    for text in text_actions:
        # 简单检查：至少50个字符
        if len(text) > 50:
            actions.append(text)
            valids.append(True)
        else:
            actions.append("")
            valids.append(False)

    return actions, valids
```

---

## 2.4 并行环境包装器

```python
# integration/parallel_envs.py

"""
并行环境包装器
让多个 WorkflowDesignEnv 并行运行
"""

import ray
from typing import List
from .workflow_env import WorkflowDesignEnv


class ParallelWorkflowEnvs:
    """
    并行环境管理器
    """

    def __init__(self, datasets: List[str], num_envs_per_dataset: int = 4):
        """
        Args:
            datasets: 数据集列表
            num_envs_per_dataset: 每个数据集的并行环境数
        """
        self.datasets = datasets
        self.num_envs_per_dataset = num_envs_per_dataset

        # 创建环境
        self.envs = []
        for dataset in datasets:
            for _ in range(num_envs_per_dataset):
                env = WorkflowDesignEnv(dataset=dataset)
                self.envs.append(env)

        self.num_envs = len(self.envs)

    def reset(self, kwargs=None):
        """重置所有环境"""
        obs_list = []
        info_list = []

        for env in self.envs:
            obs, info = env.reset()
            obs_list.append(obs)
            info_list.append(info)

        return obs_list, info_list

    def step(self, actions: List[str]):
        """并行执行动作"""
        results = []

        for env, action in zip(self.envs, actions):
            result = env.step(action)
            results.append(result)

        # 解包结果
        obs_list = [r[0] for r in results]
        rewards = [r[1] for r in results]
        dones = [r[2] for r in results]
        infos = [r[3] for r in results]

        return obs_list, rewards, dones, infos
```

---

## 2.5 配置文件

```yaml
# integration/config.yaml

# 数据配置
data:
  train_batch_size: 4  # 同时训练4个workflow设计
  val_batch_size: 2
  max_prompt_length: 4096
  max_response_length: 1024
  return_raw_chat: true

# 环境配置
env:
  env_name: workflow_design
  seed: 42
  max_steps: 15  # 每个episode最多15步修改
  rollout:
    n: 4  # 每个数据集4个并行尝试

  # Workflow 设计配置
  workflow_design:
    datasets:
      - GSM8K
      - MATH
    operators:
      - Custom
      - ScEnsemble
      - Programmer
    exec_model_name: gpt-4o-mini

# 模型配置
actor_rollout_ref:
  model:
    path: Qwen/Qwen2.5-7B-Instruct
    use_remove_padding: true

  actor:
    optim:
      lr: 1e-6
    ppo_mini_batch_size: 64
    ppo_micro_batch_size_per_gpu: 8
    use_kl_loss: true
    kl_loss_coef: 0.01

  rollout:
    name: vllm
    tensor_model_parallel_size: 2
    temperature: 0.7

# 算法配置
algorithm:
  adv_estimator: gigpo
  gamma: 0.95
  gigpo:
    step_advantage_w: 0.5
    mode: mean_std_norm

# 训练配置
trainer:
  project_name: workflow_design_rl
  experiment_name: minimal_integration
  total_epochs: 50
  n_gpus_per_node: 4
  nnodes: 1
  save_freq: 5
  test_freq: 5
```

---

## 2.6 训练入口（最小修改）

```python
# integration/train.py

"""
训练入口
连接所有组件
"""

import sys
from pathlib import Path

# 添加路径
VERL_PATH = Path(__file__).parent.parent / "verl-agent"
sys.path.insert(0, str(VERL_PATH))

from omegaconf import OmegaConf
from transformers import AutoTokenizer

# 导入 verl-agent 组件（不修改它们）
from verl.trainer.ppo.ray_trainer import RayPPOTrainer, ResourcePoolManager, Role
from agent_system.multi_turn_rollout import TrajectoryCollector

# 导入我们的适配组件
from integration.env_manager import (
    WorkflowDesignEnvironmentManager,
    workflow_design_projection
)
from integration.parallel_envs import ParallelWorkflowEnvs


def make_workflow_design_envs(config):
    """
    创建 workflow 设计环境
    遵循 verl-agent 的接口规范
    """
    from functools import partial

    # 创建并行环境
    datasets = config.env.workflow_design.datasets
    _envs = ParallelWorkflowEnvs(
        datasets=datasets,
        num_envs_per_dataset=config.env.rollout.n
    )

    _val_envs = ParallelWorkflowEnvs(
        datasets=datasets,
        num_envs_per_dataset=1
    )

    # 创建投影函数
    projection_f = partial(workflow_design_projection)

    # 创建环境管理器
    envs = WorkflowDesignEnvironmentManager(_envs, projection_f, config)
    val_envs = WorkflowDesignEnvironmentManager(_val_envs, projection_f, config)

    return envs, val_envs


def main():
    # 加载配置
    config = OmegaConf.load('integration/config.yaml')

    # 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config.actor_rollout_ref.model.path
    )

    # 创建环境
    envs, val_envs = make_workflow_design_envs(config)

    # 创建轨迹收集器
    traj_collector = TrajectoryCollector(
        config=config,
        tokenizer=tokenizer
    )

    # 定义奖励函数（简单包装）
    def reward_fn(batch, return_dict=False):
        # 奖励已经在环境中计算
        rewards = batch.non_tensor_batch.get('rewards')
        import torch
        reward_tensor = torch.tensor(rewards, dtype=torch.float32)

        if return_dict:
            return {'reward_tensor': reward_tensor}
        return reward_tensor

    # 创建 worker mapping（使用 verl-agent 的标准组件）
    from verl.trainer.ppo.workers import FSDPWorker

    role_worker_mapping = {
        Role.ActorRollout: FSDPWorker,
    }

    # 创建资源池
    resource_pool_manager = ResourcePoolManager(
        resource_pool_spec={'default': [4]},
        mapping={Role.ActorRollout: 'default'}
    )

    # 创建 trainer（完全使用 verl-agent 的 trainer，不修改）
    trainer = RayPPOTrainer(
        config=config,
        tokenizer=tokenizer,
        role_worker_mapping=role_worker_mapping,
        resource_pool_manager=resource_pool_manager,
        envs=envs,
        val_envs=val_envs,
        traj_collector=traj_collector,
        reward_fn=reward_fn,
        val_reward_fn=reward_fn,
    )

    # 初始化并训练
    trainer.init_workers()
    trainer.fit()


if __name__ == '__main__':
    main()
```

---

## 2.7 训练脚本

```bash
# integration/run_training.sh

#!/bin/bash

set -x

# 确保在正确的目录
cd "$(dirname "$0")/.."

# 设置环境变量
export PYTHONPATH="${PYTHONPATH}:$(pwd)/AFlow:$(pwd)/verl-agent"
export VLLM_ATTENTION_BACKEND=XFORMERS

# 运行训练
python integration/train.py
```

---

# 三、使用方法

## 3.1 环境准备

```bash
# 1. 确保两个项目都已经安装好依赖
cd AFlow
pip install -r requirements.txt

cd ../verl-agent
pip install -e .

# 2. 创建 integration 目录并复制文件
cd ..
mkdir -p integration
# 复制上面的所有 Python 文件到 integration/

# 3. 配置 LLM API
# 在 AFlow/config/config2.yaml 中配置好 API keys
```

## 3.2 启动训练

```bash
# 方法1: 直接运行 Python
cd integration
python train.py

# 方法2: 使用 shell 脚本
bash run_training.sh

# 方法3: 使用 Ray（如果有多机）
ray start --head
python train.py
```

## 3.3 监控训练

```bash
# Wandb（如果配置）
wandb login
# 在 https://wandb.ai 查看训练曲线

# 本地日志
tail -f integration/logs/training.log
```

---

# 四、工作流程图

```
训练循环:

1. 初始化
   ├─> AFlowWrapper 创建初始 workflow
   ├─> 评估初始性能
   └─> 返回状态给 RL agent

2. 循环 (每个 episode)
   ├─> RL agent 观察当前状态
   │   (workflow代码 + 得分 + 历史)
   │
   ├─> RL agent 生成修改建议
   │   (自然语言描述)
   │
   ├─> AFlowWrapper 应用修改
   │   ├─> 使用 AFlow 的 LLM 生成新代码
   │   └─> 使用 AFlow 的 Evaluator 评估
   │
   ├─> 计算奖励
   │   (性能提升 = 新分数 - 旧分数)
   │
   ├─> verl-agent 计算优势
   │   (GiGPO 的两级分组)
   │
   └─> 更新策略网络
       (PPO 更新)

3. 验证
   ├─> 在测试集上评估最佳 workflow
   └─> 记录性能指标
```

---

# 五、优势分析

## 5.1 零修改原框架

✅ **AFlow 完全不变**
- 使用 AFlow 的 Evaluator
- 使用 AFlow 的 LLM 配置
- 使用 AFlow 的数据加载
- 只是把它们包装起来

✅ **verl-agent 完全不变**
- 使用标准的 RayPPOTrainer
- 使用标准的 TrajectoryCollector
- 使用标准的 GiGPO 算法
- 只是添加新的环境类型

## 5.2 灵活扩展

- 可以随时切换数据集（修改 config.yaml）
- 可以调整操作符列表
- 可以更换评估策略
- 可以修改奖励函数

## 5.3 易于调试

- 可以单独测试 AFlowWrapper
- 可以单独测试环境
- 可以单独测试 RL 训练
- 日志清晰，问题定位快

---

# 六、文件清单

新增文件（全部在 integration/ 目录下）：

```
integration/
├── __init__.py                    # 包初始化
├── aflow_wrapper.py              # AFlow 包装器（核心）
├── workflow_env.py               # Gym 环境接口
├── env_manager.py                # verl-agent 环境管理器
├── parallel_envs.py              # 并行环境
├── train.py                      # 训练入口
├── config.yaml                   # 配置文件
└── run_training.sh               # 训练脚本
```

修改文件：**0 个** ❌

---

# 七、测试方法

## 7.1 单元测试

```python
# integration/test_wrapper.py

from integration.aflow_wrapper import AFlowWrapper

def test_aflow_wrapper():
    # 测试初始化
    wrapper = AFlowWrapper(dataset="GSM8K")

    # 测试 reset
    state = wrapper.reset()
    assert 'workflow' in state
    assert 'score' in state
    print(f"Initial score: {state['score']}")

    # 测试 step
    modification = "Add ScEnsemble operator to aggregate solutions"
    next_state, reward, done, info = wrapper.step(modification)
    print(f"New score: {next_state['score']}, Reward: {reward}")

if __name__ == '__main__':
    test_aflow_wrapper()
```

## 7.2 集成测试

```python
# integration/test_integration.py

from integration.train import make_workflow_design_envs
from omegaconf import OmegaConf

def test_integration():
    config = OmegaConf.load('integration/config.yaml')

    envs, val_envs = make_workflow_design_envs(config)

    # 测试 reset
    obs, infos = envs.reset(kwargs=None)
    print(f"Observation: {obs['text'][0][:200]}...")

    # 测试 step
    actions = ["Add ScEnsemble operator"] * len(obs['text'])
    next_obs, rewards, dones, infos = envs.step(actions)
    print(f"Rewards: {rewards}")

if __name__ == '__main__':
    test_integration()
```

---

# 总结

这个方案的核心优势：

1. **零修改** - 不动 AFlow 和 verl-agent 的任何源代码
2. **轻量级** - 只有 ~800 行适配代码
3. **可维护** - 清晰的接口，易于调试
4. **可扩展** - 可以轻松添加新功能

实现路径：
1. 创建 `integration/` 目录
2. 复制上面的 7 个文件
3. 配置 config.yaml
4. 运行 `bash run_training.sh`

预期效果：
- RL agent 学会如何修改 workflow
- 性能逐步提升
- 最终生成比 AFlow 基线更好的 workflow
