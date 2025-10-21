# verl-agent + AFlow 深度整合设计方案

## 核心理念

将 **AFlow 的 workflow 设计空间** 作为 **verl-agent 的环境**，训练一个智能体自动生成和优化 workflow，彻底消除人工参与。

```
┌────────────────────────────────────────────────────────────────┐
│                    整合架构 (Meta-Learning)                      │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │               verl-agent (RL Training)                    │ │
│  │                                                           │ │
│  │  ┌────────┐    ┌────────┐    ┌────────┐    ┌────────┐  │ │
│  │  │ Actor  │───>│Rollout │───>│Critic  │───>│Trainer │  │ │
│  │  └────────┘    └────────┘    └────────┘    └────────┘  │ │
│  │       │             │              │             │       │ │
│  └───────┼─────────────┼──────────────┼─────────────┼───────┘ │
│          │             │              │             │          │
│          ↓             ↓              ↓             ↓          │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │        Workflow Design Environment (基于 AFlow)           │ │
│  │                                                           │ │
│  │  State:  当前workflow代码 + 历史经验 + 评估结果           │ │
│  │  Action: 添加/删除/修改操作符                             │ │
│  │  Reward: workflow在验证集上的性能                        │ │
│  └──────────────────────────────────────────────────────────┘ │
└────────────────────────────────────────────────────────────────┘
```

---

# 一、系统设计

## 1.1 核心组件映射

| verl-agent 组件 | 功能 | AFlow 对应部分 |
|----------------|------|---------------|
| **Environment** | Workflow设计空间 | AFlow的优化器 + 评估器 |
| **State** | 当前workflow状态 | graph.py + prompt.py + 经验 |
| **Action** | 修改workflow | 操作符的增删改 |
| **Reward** | 性能提升 | 验证集得分 |
| **Policy** | 设计策略 | 学习如何修改workflow |

---

## 1.2 详细架构设计

### 环境定义：WorkflowDesignEnvironment

```python
# agent_system/environments/env_package/workflow_design/workflow_env.py

class WorkflowDesignEnvironment:
    """
    Workflow 设计环境
    继承自 gym.Env
    """

    def __init__(self, dataset: str, operators: List[str], config):
        """
        Args:
            dataset: 目标数据集 (GSM8K, MATH, HumanEval等)
            operators: 可用的操作符列表
            config: 环境配置
        """
        self.dataset = dataset
        self.operators = operators
        self.config = config

        # 加载数据集
        self.train_data = load_dataset(dataset, split='train')
        self.val_data = load_dataset(dataset, split='validation')
        self.test_data = load_dataset(dataset, split='test')

        # 初始化评估器
        self.evaluator = Evaluator(eval_path=config.eval_path)

        # 初始化 LLM
        self.exec_llm_config = config.exec_llm_config
        self.exec_llm = create_llm_instance(self.exec_llm_config)

        # 当前workflow状态
        self.current_workflow = None
        self.workflow_history = []
        self.best_score = 0.0

    def reset(self):
        """
        重置环境，返回初始状态

        Returns:
            observation: {
                'text': str,        # workflow的文本描述
                'image': None,      # 不需要图像
                'anchor': dict      # workflow的结构化表示
            }
            info: dict
        """
        # 生成初始workflow（简单的baseline）
        self.current_workflow = self._generate_initial_workflow()

        # 评估初始workflow
        initial_score = self._evaluate_workflow(self.current_workflow)
        self.best_score = initial_score

        # 重置历史
        self.workflow_history = [{
            'workflow': self.current_workflow,
            'score': initial_score,
            'step': 0
        }]

        # 构建观察
        observation = self._build_observation()

        info = {
            'current_score': initial_score,
            'best_score': self.best_score,
            'step': 0
        }

        return observation, info

    def step(self, action: Dict):
        """
        执行一个workflow修改动作

        Args:
            action: {
                'operation': str,    # 'add', 'delete', 'modify', 'reorder'
                'operator': str,     # 操作符名称
                'position': int,     # 位置
                'params': dict       # 参数
            }

        Returns:
            next_observation: dict
            reward: float
            done: bool
            info: dict
        """
        # Step 1: 应用动作修改workflow
        new_workflow, valid = self._apply_action(action)

        if not valid:
            # 无效动作，返回负奖励
            reward = -0.1
            done = False
            info = {
                'current_score': self.workflow_history[-1]['score'],
                'best_score': self.best_score,
                'step': len(self.workflow_history),
                'action_valid': False
            }
            return self._build_observation(), reward, done, info

        # Step 2: 评估新workflow
        try:
            new_score = self._evaluate_workflow(new_workflow)
        except Exception as e:
            # 评估失败（如代码错误）
            reward = -0.5
            done = False
            info = {
                'current_score': self.workflow_history[-1]['score'],
                'best_score': self.best_score,
                'step': len(self.workflow_history),
                'action_valid': True,
                'eval_failed': True,
                'error': str(e)
            }
            return self._build_observation(), reward, done, info

        # Step 3: 计算奖励
        old_score = self.workflow_history[-1]['score']
        improvement = new_score - old_score

        # 奖励函数设计
        if new_score > self.best_score:
            # 突破历史最佳，给予额外奖励
            reward = improvement + 1.0
            self.best_score = new_score
        elif improvement > 0:
            # 性能提升
            reward = improvement
        else:
            # 性能下降或不变
            reward = improvement - 0.1  # 轻微惩罚

        # Step 4: 更新状态
        self.current_workflow = new_workflow
        self.workflow_history.append({
            'workflow': new_workflow,
            'score': new_score,
            'step': len(self.workflow_history)
        })

        # Step 5: 判断是否结束
        done = (
            len(self.workflow_history) >= self.config.max_steps or
            new_score >= self.config.target_score
        )

        # Step 6: 构建新观察
        next_observation = self._build_observation()

        info = {
            'current_score': new_score,
            'best_score': self.best_score,
            'step': len(self.workflow_history),
            'action_valid': True,
            'improvement': improvement
        }

        return next_observation, reward, done, info

    def _generate_initial_workflow(self) -> Dict:
        """
        生成初始baseline workflow
        """
        if self.dataset in ['GSM8K', 'MATH']:
            # 数学问题：简单的CoT + Programmer
            return {
                'graph': '''
async def solve(problem):
    # Step 1: Generate solution
    solution = await self.custom(
        input=problem,
        instruction="Solve this step by step."
    )

    # Step 2: Use programmer to verify
    code_result = await self.programmer(
        problem=problem,
        analysis=solution
    )

    return code_result["output"]
''',
                'operators': ['Custom', 'Programmer'],
                'prompt': 'Solve this step by step.'
            }
        elif self.dataset in ['HumanEval', 'MBPP']:
            # 代码生成：Generate + Test
            return {
                'graph': '''
async def solve(problem):
    # Generate code
    code = await self.code_generate(
        problem=problem,
        entry_point=entry_point,
        instruction="Generate a Python function."
    )

    # Test code
    result = await self.test(
        problem=problem,
        solution=code,
        entry_point=entry_point
    )

    return result["solution"]
''',
                'operators': ['CustomCodeGenerate', 'Test'],
                'prompt': 'Generate a Python function.'
            }
        else:
            # QA问题：简单生成
            return {
                'graph': '''
async def solve(problem):
    answer = await self.custom(
        input=problem,
        instruction="Answer the question."
    )
    return answer
''',
                'operators': ['Custom'],
                'prompt': 'Answer the question.'
            }

    def _apply_action(self, action: Dict) -> Tuple[Dict, bool]:
        """
        应用动作修改workflow

        Returns:
            new_workflow: 新的workflow
            valid: 动作是否有效
        """
        operation = action['operation']
        operator = action.get('operator')
        position = action.get('position', 0)
        params = action.get('params', {})

        # 复制当前workflow
        new_workflow = {
            'graph': self.current_workflow['graph'],
            'operators': self.current_workflow['operators'].copy(),
            'prompt': self.current_workflow['prompt']
        }

        try:
            if operation == 'add':
                # 添加新操作符
                if operator not in self.operators:
                    return new_workflow, False

                if operator in new_workflow['operators']:
                    return new_workflow, False  # 已存在

                new_workflow['operators'].append(operator)
                new_workflow['graph'] = self._insert_operator_to_graph(
                    new_workflow['graph'],
                    operator,
                    position,
                    params
                )

            elif operation == 'delete':
                # 删除操作符
                if operator not in new_workflow['operators']:
                    return new_workflow, False

                new_workflow['operators'].remove(operator)
                new_workflow['graph'] = self._remove_operator_from_graph(
                    new_workflow['graph'],
                    operator
                )

            elif operation == 'modify':
                # 修改操作符参数
                if operator not in new_workflow['operators']:
                    return new_workflow, False

                new_workflow['graph'] = self._modify_operator_in_graph(
                    new_workflow['graph'],
                    operator,
                    params
                )

            elif operation == 'reorder':
                # 重新排序操作符
                new_order = params.get('order', [])
                if not set(new_order).issubset(set(new_workflow['operators'])):
                    return new_workflow, False

                new_workflow['graph'] = self._reorder_operators_in_graph(
                    new_workflow['graph'],
                    new_order
                )

            else:
                return new_workflow, False

            return new_workflow, True

        except Exception as e:
            print(f"Action application failed: {e}")
            return new_workflow, False

    def _evaluate_workflow(self, workflow: Dict) -> float:
        """
        评估workflow性能

        Returns:
            score: 0-1之间的得分
        """
        # 将workflow转换为可执行的graph对象
        graph_class = self._workflow_to_graph_class(workflow)

        # 使用AFlow的评估器评估
        params = {
            'llm_config': self.exec_llm_config,
            'dataset': {}
        }

        # 在验证集的子集上评估（节省时间）
        score, _, _ = asyncio.run(
            self.evaluator.graph_evaluate(
                dataset=self.dataset,
                graph=graph_class,
                params=params,
                path=self.config.eval_path,
                is_test=False
            )
        )

        return score

    def _build_observation(self) -> Dict:
        """
        构建当前状态的观察
        """
        # 获取最近的历史
        recent_history = self.workflow_history[-5:]  # 最近5步

        # 构建文本观察
        text_obs = f"""Current Workflow Design Task:
Dataset: {self.dataset}
Available Operators: {', '.join(self.operators)}

Current Workflow:
{self.current_workflow['graph']}

Current Operators: {', '.join(self.current_workflow['operators'])}
Current Score: {self.workflow_history[-1]['score']:.4f}
Best Score: {self.best_score:.4f}

Recent History:
"""
        for i, entry in enumerate(recent_history):
            text_obs += f"\nStep {entry['step']}: Score = {entry['score']:.4f}"

        text_obs += f"""

Your task is to modify the workflow to improve its performance on the {self.dataset} dataset.
You can:
1. Add a new operator from the available list
2. Delete an existing operator
3. Modify operator parameters
4. Reorder operators

What action would you like to take?
"""

        # 结构化表示（用于GiGPO的anchor）
        anchor = {
            'operators': self.current_workflow['operators'],
            'score': self.workflow_history[-1]['score'],
            'graph_hash': hash(self.current_workflow['graph'])
        }

        return {
            'text': text_obs,
            'image': None,
            'anchor': anchor
        }

    def _workflow_to_graph_class(self, workflow: Dict):
        """
        将workflow字典转换为可执行的Graph类
        """
        # 动态创建Graph类
        graph_code = f'''
class Workflow:
    def __init__(self, name, llm_config, dataset):
        self.name = name
        self.dataset = dataset
        exec_llm = create_llm_instance(llm_config)

        # 初始化操作符
{self._generate_operator_init_code(workflow['operators'])}

{workflow['graph']}
'''

        # 执行代码创建类
        namespace = {
            'create_llm_instance': create_llm_instance,
            **{op: globals()[op] for op in workflow['operators']}
        }
        exec(graph_code, namespace)

        return namespace['Workflow']

    def _generate_operator_init_code(self, operators: List[str]) -> str:
        """生成操作符初始化代码"""
        code_lines = []
        for op in operators:
            var_name = op[0].lower() + op[1:]  # Custom -> custom
            code_lines.append(f"        self.{var_name} = {op}(exec_llm)")
        return '\n'.join(code_lines)

    # ... 其他辅助方法 ...
```

---

## 1.3 动作空间设计

智能体的动作是**修改workflow的指令**，我们需要将其转换为结构化的动作。

### 方案1: 离散动作空间

```python
# 定义动作类型枚举
class ActionType:
    ADD_OPERATOR = 0
    DELETE_OPERATOR = 1
    MODIFY_PARAMETER = 2
    REORDER = 3

# 动作空间
action_space = {
    'operation': Discrete(4),           # 操作类型
    'operator': Discrete(len(operators)), # 操作符索引
    'position': Discrete(10),            # 位置
    'param_key': Discrete(param_keys),   # 参数键
    'param_value': Discrete(param_values) # 参数值
}
```

### 方案2: 文本动作（推荐）

```python
# 智能体生成自然语言动作描述
action_text = """
<think>
The current workflow uses only Custom operator with basic prompting.
To improve performance on the MATH dataset, I should:
1. Add ScEnsemble to aggregate multiple solutions
2. Add Programmer to verify answers with code

I'll add ScEnsemble operator after Custom.
</think>

<action>
{
    "operation": "add",
    "operator": "ScEnsemble",
    "position": 1,
    "params": {
        "num_solutions": 5
    }
}
</action>
"""

# 解析动作
action = parse_action_from_text(action_text)
```

---

## 1.4 提示词设计

```python
# agent_system/environments/prompts/workflow_design.py

WORKFLOW_DESIGN_TEMPLATE = """You are an expert AI agent specialized in designing optimal workflows for solving problems.

Your current task: Design a workflow for the {dataset} dataset.

Available Operators:
{operator_descriptions}

Current Workflow:
```python
{current_workflow}
```

Current Performance:
- Score: {current_score:.4f}
- Best Score: {best_score:.4f}

Recent History:
{history}

Your goal is to modify the workflow to improve its performance. You can:
1. **Add** a new operator from the available list
2. **Delete** an existing operator that might be redundant
3. **Modify** operator parameters (e.g., number of samples, temperature)
4. **Reorder** operators to optimize the execution flow

Think step-by-step:
- What are the weaknesses of the current workflow?
- Which operator would address these weaknesses?
- Where should it be placed in the workflow?

Your reasoning MUST be enclosed in <think></think> tags.
Your action MUST be enclosed in <action></action> tags in JSON format.

Example:
<think>
The current workflow lacks diversity in solutions. Adding ScEnsemble would help aggregate multiple attempts and select the best one.
</think>

<action>
{{
    "operation": "add",
    "operator": "ScEnsemble",
    "position": 2,
    "params": {{"num_solutions": 5}}
}}
</action>
"""
```

---

# 二、实现步骤

## 2.1 阶段1: 环境实现

### 文件结构

```
agent_system/environments/env_package/workflow_design/
├── __init__.py
├── workflow_env.py          # 核心环境类
├── workflow_parser.py       # Workflow解析器
├── workflow_modifier.py     # Workflow修改器
├── workflow_executor.py     # Workflow执行器
└── utils.py                 # 工具函数
```

### 核心代码框架

```python
# agent_system/environments/env_package/workflow_design/__init__.py

from .workflow_env import WorkflowDesignEnvironment
from .workflow_parser import WorkflowParser
from .workflow_modifier import WorkflowModifier

__all__ = [
    'WorkflowDesignEnvironment',
    'WorkflowParser',
    'WorkflowModifier'
]
```

---

## 2.2 阶段2: 环境管理器

```python
# agent_system/environments/env_manager.py (添加新的管理器)

class WorkflowDesignEnvironmentManager(EnvironmentManagerBase):
    """
    Workflow设计环境管理器
    """
    def __init__(self, envs, projection_f, config):
        self.memory = SimpleMemory()
        super().__init__(envs, projection_f, config)

    def reset(self, kwargs) -> Dict[str, Any]:
        """重置环境"""
        obs, infos = self.envs.reset()

        # 初始化记忆
        self.memory.reset(batch_size=len(obs))

        # 存储任务信息
        self.datasets = [info['dataset'] for info in infos]

        observations = {
            'text': self.build_text_obs(obs, init=True),
            'image': None,
            'anchor': [o['anchor'] for o in obs]
        }

        return observations, infos

    def step(self, text_actions: List[str]):
        """执行动作"""
        # 解析文本动作为结构化动作
        actions, valids = self.projection_f(text_actions)

        # 环境步进
        next_obs, rewards, dones, infos = self.envs.step(actions)

        # 存储历史
        self.memory.store({
            'observation': [o['text'] for o in obs],
            'action': text_actions
        })

        # 构建新观察
        next_observations = {
            'text': self.build_text_obs(next_obs),
            'image': None,
            'anchor': [o['anchor'] for o in next_obs]
        }

        # 添加动作有效性
        for i, info in enumerate(infos):
            info['is_action_valid'] = valids[i]

        rewards = to_numpy(rewards)
        dones = to_numpy(dones)

        return next_observations, rewards, dones, infos

    def build_text_obs(self, obs_list, init=False) -> List[str]:
        """构建文本观察"""
        postprocess_text_obs = []

        if not init and self.config.env.history_length > 0:
            memory_contexts, valid_lens = self.memory.fetch(
                self.config.env.history_length,
                obs_key="observation",
                action_key="action"
            )

        for i, obs in enumerate(obs_list):
            if init or self.config.env.history_length <= 0:
                text = obs['text']
            else:
                # 添加历史上下文
                text = f"""Previous Actions:
{memory_contexts[i]}

{obs['text']}
"""

            postprocess_text_obs.append(text)

        return postprocess_text_obs

    def success_evaluator(self, total_infos, total_batch_list,
                         episode_rewards, episode_lengths):
        """评估成功率"""
        success = defaultdict(list)

        for batch_idx in range(len(total_batch_list)):
            # 获取最终得分
            final_info = total_infos[batch_idx][-1]
            final_score = final_info['current_score']
            best_score = final_info['best_score']

            # 定义成功：超过初始baseline 10%
            initial_score = total_infos[batch_idx][0]['current_score']
            improved = (final_score - initial_score) / (initial_score + 1e-6)

            success['success_rate'].append(float(improved > 0.1))
            success['improvement'].append(improved)
            success['best_score'].append(best_score)

        return success
```

---

## 2.3 阶段3: 动作解析

```python
# agent_system/environments/env_package/workflow_design/action_parser.py

import json
import re
from typing import Dict, Tuple

def workflow_design_projection(text_actions: List[str]) -> Tuple[List[Dict], List[bool]]:
    """
    将文本动作转换为结构化动作

    Args:
        text_actions: 智能体生成的文本动作列表

    Returns:
        actions: 结构化动作列表
        valids: 动作是否有效
    """
    actions = []
    valids = []

    for text in text_actions:
        try:
            # 提取 <action> 标签中的内容
            match = re.search(r'<action>(.*?)</action>', text, re.DOTALL)
            if not match:
                # 没有找到action标签
                actions.append({})
                valids.append(False)
                continue

            action_str = match.group(1).strip()

            # 解析JSON
            action = json.loads(action_str)

            # 验证动作格式
            required_keys = ['operation', 'operator']
            if not all(key in action for key in required_keys):
                actions.append({})
                valids.append(False)
                continue

            # 验证操作类型
            valid_operations = ['add', 'delete', 'modify', 'reorder']
            if action['operation'] not in valid_operations:
                actions.append({})
                valids.append(False)
                continue

            actions.append(action)
            valids.append(True)

        except json.JSONDecodeError:
            # JSON解析失败
            actions.append({})
            valids.append(False)
        except Exception as e:
            print(f"Action parsing error: {e}")
            actions.append({})
            valids.append(False)

    return actions, valids
```

---

## 2.4 阶段4: 训练配置

```yaml
# examples/workflow_design_trainer/config.yaml

# 数据配置
data:
  train_files: null  # 不需要，由环境动态生成
  val_files: null
  train_batch_size: 4  # 同时设计4个workflow
  val_batch_size: 4
  max_prompt_length: 4096
  max_response_length: 1024
  return_raw_chat: true

# 环境配置
env:
  env_name: workflow_design
  seed: 42
  max_steps: 20  # 最多20步修改
  rollout:
    n: 4  # 每个任务生成4个独立的设计尝试
  resources_per_worker:
    num_cpus: 1
    num_gpus: 0

  # Workflow设计特定配置
  workflow_design:
    datasets:
      - GSM8K
      - MATH
      - HumanEval
    operators:
      - Custom
      - AnswerGenerate
      - CustomCodeGenerate
      - ScEnsemble
      - Programmer
      - Test
      - Review
      - Revise
      - MdEnsemble
    eval_samples: 50  # 每次评估使用50个样本
    target_score: 0.9  # 目标得分

# 模型配置
actor_rollout_ref:
  model:
    path: Qwen/Qwen2.5-7B-Instruct
    use_remove_padding: true
    enable_gradient_checkpointing: true

  actor:
    optim:
      lr: 5e-7  # 较小的学习率
    ppo_mini_batch_size: 64
    ppo_micro_batch_size_per_gpu: 8
    use_kl_loss: true
    kl_loss_coef: 0.02

  rollout:
    name: vllm
    tensor_model_parallel_size: 2
    gpu_memory_utilization: 0.5
    temperature: 0.7
    do_sample: true

# 算法配置
algorithm:
  adv_estimator: gigpo
  gamma: 0.99  # 长期规划
  gigpo:
    step_advantage_w: 0.5
    mode: mean_std_norm
    enable_similarity: true  # 启用相似workflow分组
    similarity_thresh: 0.9

# 训练配置
trainer:
  project_name: workflow_design_rl
  experiment_name: gigpo_qwen2.5_7b
  total_epochs: 100
  n_gpus_per_node: 4
  nnodes: 1
  save_freq: 10
  test_freq: 5
  logger:
    - console
    - wandb
```

---

## 2.5 阶段5: 训练脚本

```bash
# examples/workflow_design_trainer/run_workflow_design.sh

#!/bin/bash

set -x

# 环境变量
export VLLM_ATTENTION_BACKEND=XFORMERS

# 训练参数
train_batch_size=4
val_batch_size=4
group_size=4
max_steps=20

# 运行训练
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=gigpo \
    data.train_batch_size=$train_batch_size \
    data.val_batch_size=$val_batch_size \
    data.max_prompt_length=4096 \
    data.max_response_length=1024 \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-7B-Instruct \
    actor_rollout_ref.actor.optim.lr=5e-7 \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.02 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.temperature=0.7 \
    algorithm.gamma=0.99 \
    algorithm.gigpo.step_advantage_w=0.5 \
    algorithm.gigpo.mode=mean_std_norm \
    algorithm.gigpo.enable_similarity=True \
    algorithm.gigpo.similarity_thresh=0.9 \
    env.env_name=workflow_design \
    env.seed=42 \
    env.max_steps=$max_steps \
    env.rollout.n=$group_size \
    trainer.project_name=workflow_design_rl \
    trainer.experiment_name=gigpo_qwen2.5_7b \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=10 \
    trainer.test_freq=5 \
    trainer.total_epochs=100 \
    trainer.logger=['console','wandb'] \
    $@
```

---

# 三、关键技术点

## 3.1 GiGPO 在 Workflow 设计中的应用

```python
# GiGPO 的两级分组机制非常适合workflow设计

# Episode-level 分组:
#   - 同一个数据集的多个设计尝试形成一组
#   - 组内比较：哪个设计策略更好？
#   - Episode优势 = 该workflow得分 - 组平均得分

# Step-level 分组:
#   - 相似的workflow状态形成一组（通过anchor_obs）
#   - 组内比较：在相似状态下，哪个修改动作更好？
#   - Step优势 = 该动作的即时奖励 - 组平均奖励

# 示例:
# Group 1 (数据集: GSM8K):
#   Workflow A: Custom -> Programmer (score: 0.75)
#   Workflow B: Custom -> ScEnsemble -> Programmer (score: 0.82)
#   Workflow C: Custom -> MdEnsemble (score: 0.78)
#
#   Episode优势: B > C > A
#
#   在 "Custom -> ?" 这个状态:
#     Action 1: 添加ScEnsemble (step reward: +0.07)
#     Action 2: 添加Programmer (step reward: +0.00)
#     Action 3: 添加MdEnsemble (step reward: +0.03)
#
#   Step优势: Action 1 > Action 3 > Action 2
```

## 3.2 Workflow 相似度计算

```python
# core_gigpo.py 中 enable_similarity=True 时使用

def compute_workflow_similarity(wf1: Dict, wf2: Dict) -> float:
    """
    计算两个workflow的相似度
    """
    # 方法1: 基于操作符集合
    ops1 = set(wf1['operators'])
    ops2 = set(wf2['operators'])
    jaccard = len(ops1 & ops2) / len(ops1 | ops2)

    # 方法2: 基于代码编辑距离
    from difflib import SequenceMatcher
    code_sim = SequenceMatcher(None, wf1['graph'], wf2['graph']).ratio()

    # 方法3: 基于性能
    score_diff = abs(wf1['score'] - wf2['score'])
    score_sim = 1.0 - min(score_diff, 1.0)

    # 综合相似度
    similarity = 0.4 * jaccard + 0.4 * code_sim + 0.2 * score_sim

    return similarity
```

## 3.3 奖励函数设计

```python
def compute_workflow_reward(
    old_score: float,
    new_score: float,
    best_score: float,
    action_valid: bool,
    eval_success: bool
) -> float:
    """
    设计良好的奖励函数
    """
    if not action_valid:
        return -0.1  # 无效动作小惩罚

    if not eval_success:
        return -0.5  # 评估失败中等惩罚

    # 性能提升奖励
    improvement = new_score - old_score

    if new_score > best_score:
        # 突破历史最佳：基础奖励 + 额外奖励
        reward = improvement + 1.0
    elif improvement > 0:
        # 性能提升：基础奖励
        reward = improvement
    elif improvement == 0:
        # 无变化：轻微惩罚
        reward = -0.05
    else:
        # 性能下降：惩罚
        reward = improvement - 0.1

    # 奖励缩放
    reward = reward * 10  # 放大信号

    return reward
```

---

# 四、预期效果

## 4.1 训练过程

```
Epoch 1:
  - 智能体随机尝试各种修改
  - 发现添加ScEnsemble通常能提升性能
  - Episode reward: +2.3

Epoch 10:
  - 智能体学会先添加生成类操作符，再添加集成类操作符
  - 开始尝试调整参数（如num_solutions）
  - Episode reward: +5.7

Epoch 50:
  - 智能体针对不同数据集使用不同策略
    - GSM8K: Custom -> Programmer -> ScEnsemble
    - HumanEval: CustomCodeGenerate -> Test -> Review
  - 能够识别何时停止修改（收益递减）
  - Episode reward: +8.4

Epoch 100:
  - 智能体找到接近最优的workflow
  - 平均性能提升: 25-35%
  - Episode reward: +10.2
```

## 4.2 最终能力

训练完成后，智能体能够：

1. **自动设计** - 给定新数据集，自动生成初始workflow
2. **迭代优化** - 持续改进workflow直到收敛
3. **迁移学习** - 将在一个数据集学到的经验应用到其他数据集
4. **自适应** - 根据数据集特点选择合适的操作符组合

---

# 五、实施计划

## 第一周: 环境开发
- [ ] 实现 WorkflowDesignEnvironment 基础类
- [ ] 实现动作解析器
- [ ] 实现workflow执行器
- [ ] 单元测试

## 第二周: 整合 verl-agent
- [ ] 创建 WorkflowDesignEnvironmentManager
- [ ] 修改 make_envs 函数
- [ ] 配置提示词模板
- [ ] 集成测试

## 第三周: 训练与调试
- [ ] 小规模训练（1个数据集，50个样本）
- [ ] 调试奖励函数
- [ ] 调整超参数
- [ ] 验证GiGPO有效性

## 第四周: 全规模实验
- [ ] 多数据集训练
- [ ] 性能评估
- [ ] 与AFlow基线对比
- [ ] 撰写报告

---

# 六、技术挑战与解决方案

## 挑战1: 评估速度慢

**问题**: 每次评估workflow需要在验证集上运行，非常耗时

**解决方案**:
1. 使用验证集的小子集（50-100个样本）进行快速评估
2. 缓存相似workflow的评估结果
3. 使用代理模型预测性能（训练一个小模型预测workflow得分）

## 挑战2: 动作空间复杂

**问题**: workflow修改的可能性太多，探索困难

**解决方案**:
1. 课程学习：先学习简单修改（添加/删除），再学习复杂修改
2. 分层RL：高层决策选择操作类型，低层决策选择具体操作符
3. 约束动作空间：限制每步只能修改一个操作符

## 挑战3: 稀疏奖励

**问题**: 只有workflow评估完成才有奖励，中间步骤无反馈

**解决方案**:
1. 设计中间奖励：代码语法正确 (+0.1)、新操作符成功调用 (+0.2)
2. 使用GiGPO的step-level优势估计
3. 奖励塑形：根据代码复杂度、操作符多样性等给予辅助奖励

## 挑战4: 训练不稳定

**问题**: RL训练可能不收敛

**解决方案**:
1. 使用较小的学习率（5e-7）
2. 增大KL惩罚系数（0.02）
3. 使用GiGPO而非PPO（更稳定）
4. 添加curriculum：从简单数据集开始训练

---

# 七、代码示例

## 完整训练示例

```python
# main.py

import asyncio
from agent_system.environments.env_manager import make_envs
from verl.trainer.ppo.ray_trainer import RayPPOTrainer

async def main():
    # 配置
    config = load_config('config.yaml')

    # 创建环境
    envs, val_envs = make_envs(config)
    # envs = WorkflowDesignEnvironmentManager(...)

    # 创建trainer
    trainer = RayPPOTrainer(
        config=config,
        tokenizer=tokenizer,
        role_worker_mapping=role_worker_mapping,
        resource_pool_manager=resource_pool_manager,
        envs=envs,
        val_envs=val_envs,
        traj_collector=traj_collector,
        reward_fn=workflow_reward_fn,
        val_reward_fn=workflow_val_reward_fn,
    )

    # 初始化workers
    trainer.init_workers()

    # 开始训练
    trainer.fit()

if __name__ == '__main__':
    asyncio.run(main())
```

---

# 总结

这个整合方案将 **verl-agent 的 RL 训练能力** 与 **AFlow 的 workflow 设计空间** 深度结合，实现了：

1. ✅ **完全自动化** - 无需人工参与workflow设计
2. ✅ **持续优化** - 通过RL不断改进
3. ✅ **泛化能力** - 学到的经验可迁移到新数据集
4. ✅ **高效探索** - GiGPO的细粒度信用分配加速学习

核心创新点：
- 将workflow设计作为RL环境
- 使用GiGPO实现episode和step两级优化
- 通过相似度匹配实现经验复用
- 设计良好的奖励函数引导学习

这是一个**元学习（Meta-Learning）**系统，智能体学习的是"如何设计workflow"，而不是"如何解决特定问题"。
