# Schema 1: MCTS Optimization for Workflow Generation

## 概述 (Overview)

This project implements **Schema 1** - a clean separation of concerns between workflow optimization and policy learning.

**核心理念**: MCTS独立找最优workflow，GRPO（未来）学会生成好的修改建议。

---

## 项目结构 (Project Structure)

```
integration/
├── README.md                          # 本文件
├──
├── 核心训练脚本 (Core Training)
│   ├── deep_train_real_workflow.py   # 主训练脚本（纯MCTS优化）
│   └── deep_workflow_env.py          # MCTS环境包装器
│
├── 评估和适配 (Evaluation)
│   ├── workflow_evaluator.py         # 统一评估器（支持多数据集）
│   └── evaluation_adapter.py         # 评估适配器
│
├── 配置文件 (Configuration)
│   ├── aime_full_test.yaml           # 完整配置（5 epochs）
│   └── aime_minimal_test.yaml        # 最小化配置（1 epoch，快速验证）
│
└── ../AFlow/                          # 原生AFlow（MCTS优化器）
    └── scripts/optimizer.py          # 纯MCTS实现

```

---

## 快速开始 (Quick Start)

### 1. 验证环境
```bash
cd /content/agentflow/integration
python3 << 'EOF'
from scripts.logs import logger
from deep_workflow_env import create_deep_workflow_env
from scripts.optimizer import Optimizer
print("✅ All imports OK - Ready to train!")
EOF
```

### 2. 运行最小化测试（1-2分钟）
```bash
nohup python deep_train_real_workflow.py --config aime_minimal_test.yaml &
```

### 3. 运行完整训练（几小时）
```bash
nohup python deep_train_real_workflow.py --config aime_full_test.yaml &
```

---

## 配置参数说明 (Configuration Parameters)

### 核心参数

| 参数 | 说明 | 类型 |
|------|------|------|
| `device` | 计算设备（cuda/cpu） | string |
| `total_epochs` | 总训练epoch数 | int |
| `episodes_per_epoch` | 每个epoch的episode数 | int |
| `output_dir` | 输出目录 | string |

### 环境参数

| 参数 | 说明 | 类型 | 默认值 |
|------|------|------|--------|
| `dataset` | 数据集名称（AIME/HumanEval/etc） | string | - |
| `max_rounds` | MCTS树搜索最大轮数 | int | 5 |
| `sample` | 每轮评估的样本数 | int | 24 |
| `train_test_split` | 训练/测试集划分比例 | float | 0.8 |
| `mini_batch_size` | Mini-batch大小（None=全量） | int | 6 |

### Operators
```yaml
operators:
  - "Custom"       # 基础operator
  - "ScEnsemble"   # 自一致性集成
  - "Review"       # 审查
  - "Revise"       # 修订
```

### LLM配置
```yaml
opt_llm_config:      # MCTS优化中使用的LLM
  model: "gpt-4o-mini"
  temperature: 0.9
exec_llm_config:     # workflow执行时使用的LLM
  model: "gpt-4o-mini"
  temperature: 0.7
```

---

## Schema 1 vs Schema 2

### Schema 1 (当前) ✅ 推荐

**架构**:
```
MCTS优化器（AFlow）     Qwen + GRPO（未来）
    ↓                        ↓
  纯粹树搜索    ← 分离的数据流 → 从reward学习
    ↓                        ↓
  评估真实性能              梯度更新
```

**优势**:
- ✅ 职责清晰，易于调试
- ✅ 代码简洁（~700行）
- ✅ 参数少（12个关键参数）
- ✅ 标准MCTS + 标准GRPO

**文件**:
- `deep_workflow_env.py`: 使用原生 `Optimizer`
- `deep_train_real_workflow.py`: 纯MCTS评估循环

### Schema 2 (旧) ❌ 已移除

**问题**:
- ❌ 复杂架构（MCTS + RL权重融合）
- ❌ 代码臃肿（2400+行）
- ❌ 参数混乱（15+个参数）
- ❌ 两个系统相互影响

**移除的文件**:
- `rl_trainer.py` - GRPO训练器
- `trainable_qwen_policy.py` - Qwen policy
- `unified_state.py` - 状态管理
- `workflow_code_prompt_manager.py` - 代码生成
- ...等等

---

## 文件说明 (File Documentation)

### deep_train_real_workflow.py
主训练脚本，实现Schema 1的训练循环：

**关键方法**:
- `__init__(config)`: 初始化训练器
- `_create_environments()`: 为每个数据集创建环境
- `train_epoch(epoch)`: 训练一个epoch（纯MCTS）
- `train()`: 主训练循环
- `save_checkpoint(epoch)`: 保存检查点

**特点**:
- 纯MCTS优化（无RL融合）
- 多轮环境交互
- 保存最优workflow

### deep_workflow_env.py
环境包装器，用原生AFlow Optimizer：

**关键方法**:
- `__init__()`: 初始化环境和MCTS优化器
- `reset()`: 重置环境
- `step(actions)`: 执行一步MCTS优化
- `_step_mcts()`: MCTS树搜索实现

**特点**:
- 使用原生 `Optimizer` from `scripts.optimizer`
- 支持mini-batch评估
- 统一使用 `WorkflowEvaluator`

### workflow_evaluator.py
统一的评估器，支持多数据集：

**支持的数据集**:
- HumanEval
- MBPP
- AIME (通过改造)
- MATH
- GSM8K

**特点**:
- 支持mini-batch随机采样
- 支持train/test split
- 真实pass@k评估

### evaluation_adapter.py
将AFlow的evaluation_utils替换为WorkflowEvaluator：

**职责**:
- 适配评估器接口
- 缓存评估结果

---

## 训练流程 (Training Flow)

```
Start Training
    ↓
For each Epoch:
    ├─ Create environment
    ├─ Reset environment state
    └─ For each Episode:
       ├─ env.reset() → 初始workflow观测
       ├─ env.step() → MCTS树搜索
       │  ├─ Optimizer._optimize_graph()
       │  ├─ 评估workflow性能
       │  └─ 返回pass@k分数
       └─ 记录episode统计
    └─ Save checkpoint if needed
End Training
```

---

## 输出目录 (Output Structure)

```
output/schema1_mcts/
├── checkpoints/
│   ├── epoch_1.pt
│   ├── epoch_2.pt
│   └── best.pt          # 最好的epoch
└── logs/
    └── training.log
```

**Checkpoint格式**:
```python
checkpoint = {
    'epoch': int,
    'stats': {
        'avg_score': float,
        'max_score': float,
        'total_episodes': int,
        ...
    },
    'config': dict
}
```

---

## 常见问题 (FAQ)

### Q1: 如何修改数据集？
编辑YAML配置文件中的 `environment.dataset`:
```yaml
environment:
  dataset: "HumanEval"  # 改为其他数据集
```

### Q2: 如何加快训练？
使用 `aime_minimal_test.yaml` 或调整参数：
```yaml
total_epochs: 1          # 减少epoch数
episodes_per_epoch: 1    # 减少episode数
max_rounds: 2            # 减少MCTS轮数
sample: 6                # 减少样本数
```

### Q3: 如何添加GRPO训练？
Schema 1已预留接口。未来可在 `train_epoch()` 中添加：
```python
# 伪代码
for episode in episodes:
    obs, reward = env.step()  # 获得reward
    grpo_trainer.update(obs, reward)  # GRPO学习
```

### Q4: 如何使用自己的LLM？
编辑YAML中的 `opt_llm_config` 和 `exec_llm_config`：
```yaml
opt_llm_config:
  model: "your-model"
  key: "${YOUR_API_KEY}"
  base_url: "https://your-endpoint"
```

---

## 注意事项 (Important Notes)

### ⚠️ 必须设置API密钥
```bash
export OPENAI_API_KEY="your-key-here"
```

### ⚠️ 确保AFlow是原生版本
```bash
ls -la ../AFlow/scripts/optimizer.py  # 应该存在
ls -la ../AFlow/scripts/optimizer_rl.py  # 应该NOT存在
```

### ⚠️ 配置文件要求
- `environment.dataset` 必须指定
- `opt_llm_config` 必须配置
- `exec_llm_config` 必须配置

---

## 参考资源 (References)

- [AFlow Documentation](../AFlow/README.md)
- [Original Paper](../paper.pdf)
- [Schema Comparison](../SCHEMA_COMPARISON.md)

---

## 许可证 (License)

MIT License - See LICENSE file

---

**最后更新**: 2025-10-30
**维护者**: AI Assistant
**状态**: ✅ Schema 1 Production Ready
