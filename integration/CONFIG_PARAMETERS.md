# 配置参数说明文档

## 完整参数列表

本文档说明所有可用的配置参数，包括方案B的新增参数。

---

## Environment 配置

### 基础参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `train_datasets` | List[str] | 必需 | 训练数据集列表，如`["AIME"]` |
| `test_datasets` | List[str] | 必需 | 测试数据集列表 |
| `data_path` | str | 必需 | 数据文件路径 |
| `train_test_split` | float | 0.8 | 训练/测试集划分比例（0-1） |

### LLM 配置

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `opt_llm_config` | Dict | 必需 | 优化LLM配置（用于workflow生成） |
| `exec_llm_config` | Dict | 必需 | 执行LLM配置（用于workflow中的LLM调用） |

**LLM配置子参数：**
```yaml
opt_llm_config:
  model: "gpt-4o-mini"  # 模型名称
  key: "${OPENAI_API_KEY}"  # API密钥
  base_url: "https://api.openai.com/v1"  # API基础URL
  temperature: 0.9  # 采样温度
```

### Operators 配置

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `operators` | List[str] | `["Custom", ...]` | 可用operators列表 |

**常用operators：**
- `Custom`: 通用LLM调用
- `CustomCodeGenerate`: 代码生成（HumanEval等）
- `ScEnsemble`: 自一致性集成
- `Test`: 测试
- `Review`: 审查
- `Revise`: 修订

**注意**：根据数据集选择合适的operators：
- AIME/MATH: `Custom`, `ScEnsemble`, `Review`, `Revise`
- HumanEval/MBPP: `Custom`, `CustomCodeGenerate`, `ScEnsemble`, `Test`

### 环境参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `env_num` | int | 2 | 并行环境数量 |
| `sample` | int | 3 | 每轮测试的样本数 |
| `max_rounds` | int | 10 | 最大训练轮次 |
| `workflow_sample_count` | int | None | Workflow内部采样数（如ScEnsemble） |

---

## 模式配置

### Static Mode（默认）

不启用Dynamic Mode，Qwen直接生成代码。

```yaml
environment:
  use_dynamic_optimizer: false  # 或不设置此参数
```

**特点**：
- ✅ Qwen直接生成完整Python代码
- ✅ 无Parser，无MCTS
- ✅ 训练Qwen学习代码生成
- ✅ 低成本

### Dynamic Mode（原版AFlow）

启用MCTS树搜索，使用GPT-4生成代码。

```yaml
environment:
  use_dynamic_optimizer: true
  validation_rounds: 3
  rl_weight: 0.5
```

**特点**：
- ✅ 完整MCTS树搜索
- ✅ UCB + Q-value融合
- ❌ 使用GPT-4生成代码（成本高）

### 方案B（推荐）

MCTS + Qwen代码生成。

```yaml
environment:
  use_dynamic_optimizer: true
  use_qwen_code_generation: true
  qwen_max_retries: 2
  validation_rounds: 3
  rl_weight: 0.5
```

**特点**：
- ✅ 完整MCTS树搜索
- ✅ Qwen生成代码（而非GPT-4）
- ✅ 低成本 + 高质量
- ✅ 语法验证 + 重试
- ✅ GPT-4 fallback

---

## Dynamic Mode 参数

仅在`use_dynamic_optimizer: true`时有效。

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `use_dynamic_optimizer` | bool | false | 是否启用MCTS优化 |
| `validation_rounds` | int | 3 | 每个workflow的验证轮次 |
| `rl_weight` | float | 0.5 | RL Q-value权重（0-1） |

**rl_weight说明**：
- `combined_score = (1-w) * ucb + w * q_value`
- `0.0`: 纯MCTS UCB
- `0.5`: UCB和Q-value均衡
- `1.0`: 纯RL Q-value

---

## 方案B参数（新增）

仅在`use_dynamic_optimizer: true`时有效。

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `use_qwen_code_generation` | bool | false | 是否使用Qwen生成代码 |
| `qwen_max_retries` | int | 2 | 语法错误时的最大重试次数 |

**use_qwen_code_generation说明**：
- `false`: 使用GPT-4生成代码（原版AFlow）
- `true`: 使用Qwen生成代码（方案B）

**qwen_max_retries说明**：
- Qwen生成的代码如果有语法错误，最多重试几次
- 如果所有尝试都失败，自动fallback到GPT-4

---

## RL 配置

### Policy 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `model_path` | str | 必需 | Qwen模型路径 |
| `use_lora` | bool | true | 是否使用LoRA |
| `lora_r` | int | 16 | LoRA rank |
| `lora_alpha` | int | 32 | LoRA alpha |
| `value_head_dim` | int | 1024 | Value head维度 |

### 训练参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `learning_rate` | float | 0.00003 | 学习率 |
| `batch_size` | int | 4 | 批次大小 |
| `ppo_epochs` | int | 4 | PPO训练轮次 |
| `ppo_clip` | float | 0.2 | PPO裁剪参数 |
| `gamma` | float | 0.99 | 折扣因子 |
| `gae_lambda` | float | 0.95 | GAE lambda |
| `value_coef` | float | 0.5 | Value loss系数 |
| `entropy_coef` | float | 0.03 | Entropy系数 |
| `gradient_clip` | float | 1.0 | 梯度裁剪 |

### GiGPO 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `gigpo.enable` | bool | true | 是否启用GiGPO |
| `gigpo.epsilon` | float | 0.000001 | Epsilon值 |
| `gigpo.step_advantage_w` | float | 1.0 | 步骤优势权重 |
| `gigpo.mode` | str | "mean_norm" | 归一化模式 |
| `gigpo.enable_similarity` | bool | true | 是否启用相似性检测 |
| `gigpo.similarity_thresh` | float | 0.95 | 相似性阈值 |

---

## 顶层参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `device` | str | "cuda" | 设备（cuda/cpu） |
| `total_epochs` | int | 10 | 总训练轮次 |
| `episodes_per_epoch` | int | 6 | 每轮的episode数 |
| `update_frequency` | int | 1 | 更新频率 |
| `eval_episodes` | int | 2 | 评估episode数 |
| `save_frequency` | int | 1 | 保存频率 |
| `output_dir` | str | 必需 | 输出目录 |
| `experience_pool_size` | int | 500 | 经验池大小 |

---

## 配置示例

### Static Mode 配置

```yaml
device: "cuda"
total_epochs: 10
output_dir: "/path/to/output"

environment:
  train_datasets: ["AIME"]
  data_path: "/path/to/data.jsonl"
  use_dynamic_optimizer: false  # Static Mode

  opt_llm_config:
    model: "gpt-4o-mini"
    key: "${OPENAI_API_KEY}"
    temperature: 0.9

  operators:
    - "Custom"
    - "ScEnsemble"

  sample: 24
  max_rounds: 5

rl:
  policy:
    model_path: "/path/to/qwen"
    use_lora: true
  learning_rate: 0.00003
```

### Dynamic Mode 配置

```yaml
device: "cuda"
total_epochs: 10
output_dir: "/path/to/output"

environment:
  train_datasets: ["AIME"]
  data_path: "/path/to/data.jsonl"

  # Dynamic Mode（原版）
  use_dynamic_optimizer: true
  validation_rounds: 3
  rl_weight: 0.5

  opt_llm_config:
    model: "gpt-4o-mini"
    key: "${OPENAI_API_KEY}"

  sample: 24

rl:
  policy:
    model_path: "/path/to/qwen"
```

### 方案B配置

```yaml
device: "cuda"
total_epochs: 10
output_dir: "/path/to/output"

environment:
  train_datasets: ["AIME"]
  data_path: "/path/to/data.jsonl"

  # 方案B = MCTS + Qwen
  use_dynamic_optimizer: true
  use_qwen_code_generation: true  # 方案B核心开关
  qwen_max_retries: 2

  validation_rounds: 3
  rl_weight: 0.5

  opt_llm_config:
    model: "gpt-4o-mini"  # 仅用于fallback
    key: "${OPENAI_API_KEY}"

  sample: 24

rl:
  policy:
    model_path: "/path/to/qwen"
```

---

## 参数优先级

1. **命令行参数** > 配置文件参数 > 默认值
2. **环境变量** 可用于API密钥等敏感信息
   - 使用`${VAR_NAME}`引用环境变量

---

## 最佳实践

### 训练流程

**阶段1：Static Mode训练Qwen**
```yaml
environment:
  use_dynamic_optimizer: false
  sample: 24
  max_rounds: 5
```

目标：让Qwen学会生成正确的代码

**阶段2：方案B优化**
```yaml
environment:
  use_dynamic_optimizer: true
  use_qwen_code_generation: true
  sample: 24
  max_rounds: 5
```

目标：使用MCTS搜索最佳workflow设计

### 参数调优

**小数据集（如AIME 30题）：**
- `learning_rate`: 0.00003（低）
- `batch_size`: 4（小）
- `entropy_coef`: 0.03（低，减少探索）

**大数据集（如HumanEval 164题）：**
- `learning_rate`: 0.0001（正常）
- `batch_size`: 8（正常）
- `entropy_coef`: 0.1（正常）

**rl_weight调整：**
- 训练初期：0.3-0.4（更依赖MCTS探索）
- 训练后期：0.5-0.7（更依赖RL经验）

---

## 故障排查

### 问题：方案B不生效

检查：
1. `use_dynamic_optimizer: true` 已设置？
2. `use_qwen_code_generation: true` 已设置？
3. 日志中有`"方案B: Using Qwen to generate code"`？

### 问题：Qwen语法错误太多

解决：
1. 增加`qwen_max_retries`
2. 先在Static Mode训练Qwen
3. 检查system prompt是否正确加载

### 问题：Dynamic Mode性能不如Static

原因：
- MCTS需要更多轮次才能收敛
- 增加`max_rounds`和`total_epochs`
- 调整`rl_weight`平衡探索和利用

---

## 相关文档

- **快速入门**: `SOLUTION_B_QUICKSTART.md`
- **实现文档**: `SOLUTION_B_IMPLEMENTATION.md`
- **Parser移除**: `PARSER_REMOVAL_SUMMARY.md`
- **MCTS方案**: `MCTS_SOLUTION.md`

---

**文档版本**: 1.0
**更新日期**: 2025-10-28
