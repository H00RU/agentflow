# Quick Start Guide
# 快速开始指南

立即开始使用AFlow + verl-agent深度集成系统！

Get started with the AFlow + verl-agent deep integration system right away!

---

## 🚀 5分钟快速开始 5-Minute Quick Start

### 1. 安装依赖 Install Dependencies

```bash
# 基础依赖 Basic dependencies
pip install torch numpy pyyaml ray

# OpenAI API (可选，用于LLM调用)
pip install openai

# 其他可选依赖 Optional dependencies
pip install tensorboard wandb
```

### 2. 设置环境变量 Set Environment Variables

```bash
# OpenAI API密钥（如果使用OpenAI）
export OPENAI_API_KEY="sk-your-api-key-here"

# 可选: 自定义API端点
export OPENAI_BASE_URL="https://api.openai.com/v1"
```

### 3. 修改配置 Modify Configuration

编辑 `integration/deep_config.yaml`:

```yaml
# 最小修改 - 只需修改这几项
environment:
  train_datasets:
    - "HumanEval"  # 或您的数据集

  opt_llm_config:
    model: "gpt-4"  # 或您的模型
    api_key: null   # 使用环境变量

# 其他保持默认即可
total_epochs: 5  # 快速测试用5轮
```

### 4. 启动训练 Start Training

```bash
cd integration
python deep_train.py --config deep_config.yaml
```

### 5. 监控进度 Monitor Progress

```bash
# 实时查看日志
tail -f output/deep_integration/logs/training.log

# 查看统计数据
cat output/deep_integration/logs/training_stats.json
```

---

## 📋 详细步骤 Detailed Steps

### 步骤1: 环境准备 Environment Setup

#### 1.1 检查Python版本 Check Python Version

```bash
python --version
# 需要 Python 3.8+
```

#### 1.2 创建虚拟环境（推荐）Create Virtual Environment (Recommended)

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或 venv\Scripts\activate  # Windows
```

#### 1.3 安装依赖 Install Dependencies

```bash
# 方式1: 直接安装 Direct installation
pip install torch numpy pyyaml ray openai

# 方式2: 使用requirements.txt (如果提供)
# pip install -r requirements.txt
```

#### 1.4 验证安装 Verify Installation

```bash
python -c "import torch, numpy, yaml, ray; print('All dependencies installed!')"
```

### 步骤2: 配置系统 Configure System

#### 2.1 查看配置文件 View Configuration

```bash
cd integration
cat deep_config.yaml
```

#### 2.2 最小配置修改 Minimal Configuration Changes

创建您自己的配置文件 `my_config.yaml`:

```yaml
# my_config.yaml - 最小配置示例
device: "cuda"  # 或 "cpu"
output_dir: "./my_output"
total_epochs: 10

environment:
  train_datasets:
    - "HumanEval"

  env_num: 2  # 并行环境数量
  group_n: 2  # GiGPO分组数量

  opt_llm_config:
    model: "gpt-4"
    temperature: 0.7
    max_tokens: 2000

rl:
  initial_weight: 0.5
  weight_schedule: "linear_increase"
```

#### 2.3 高级配置（可选）Advanced Configuration (Optional)

```yaml
# 调整RL参数
rl:
  gigpo:
    enable_similarity: true
    similarity_thresh: 0.95
    workflow_similarity_thresh: 0.8

# 调整并行性
environment:
  env_num: 4
  group_n: 2

# 启用实验跟踪
experiment:
  use_wandb: true
  wandb_project: "my-aflow-project"
```

### 步骤3: 运行训练 Run Training

#### 3.1 基础训练 Basic Training

```bash
cd integration
python deep_train.py --config deep_config.yaml
```

#### 3.2 使用自定义配置 Use Custom Configuration

```bash
python deep_train.py --config my_config.yaml
```

#### 3.3 覆盖配置参数 Override Configuration

```bash
python deep_train.py \
  --config deep_config.yaml \
  --output_dir ./custom_output \
  --device cuda
```

#### 3.4 调试模式 Debug Mode

在配置文件中启用调试:

```yaml
debug:
  enable_debug_mode: true
  debug_episodes: 5
  verbose_env: true
  verbose_rl: true
```

然后运行:

```bash
python deep_train.py --config deep_config.yaml
```

### 步骤4: 监控训练 Monitor Training

#### 4.1 实时日志 Real-time Logs

```bash
# 终端1: 运行训练
python deep_train.py --config deep_config.yaml

# 终端2: 监控日志
tail -f output/deep_integration/logs/training.log
```

#### 4.2 查看统计 View Statistics

```bash
# 训练统计
cat output/deep_integration/logs/training_stats.json

# 评估结果
cat output/deep_integration/logs/eval_epoch_5.json
```

#### 4.3 可视化（Python）Visualization (Python)

```python
import json
import matplotlib.pyplot as plt

# 加载统计数据
with open('output/deep_integration/logs/training_stats.json') as f:
    stats = json.load(f)

# 绘制训练曲线
plt.figure(figsize=(10, 6))
plt.plot(stats['avg_scores'])
plt.xlabel('Epoch')
plt.ylabel('Average Score')
plt.title('Training Progress')
plt.grid(True)
plt.savefig('training_progress.png')
plt.show()
```

#### 4.4 TensorBoard（可选）TensorBoard (Optional)

```bash
# 启用TensorBoard (在配置中)
# logging:
#   use_tensorboard: true

# 运行TensorBoard
tensorboard --logdir output/tensorboard
```

### 步骤5: 检查结果 Check Results

#### 5.1 查看输出目录 View Output Directory

```bash
tree output/deep_integration/

# 结构 Structure:
# output/deep_integration/
# ├── checkpoints/
# │   ├── best.pt
# │   ├── best_pool.pkl
# │   └── epoch_5.pt
# ├── logs/
# │   ├── training.log
# │   ├── training_stats.json
# │   └── eval_epoch_5.json
# └── optimized_workflows/
#     ├── train/
#     └── test/
```

#### 5.2 加载最佳检查点 Load Best Checkpoint

```python
import torch

# 加载检查点
checkpoint = torch.load('output/deep_integration/checkpoints/best.pt')

print(f"Best epoch: {checkpoint['epoch']}")
print(f"Best score: {checkpoint['stats']['best_score']}")
```

#### 5.3 查看优化的工作流 View Optimized Workflows

```bash
# 查看某个数据集的优化结果
cat output/deep_integration/optimized_workflows/train/HumanEval/workflows/round_10/graph.py
```

---

## 🔧 常见问题解决 Troubleshooting

### 问题1: ImportError

**错误 Error:**
```
ImportError: No module named 'scripts.optimizer_rl'
```

**解决 Solution:**
```bash
# 确保路径正确
export PYTHONPATH="${PYTHONPATH}:$(pwd)/AFlow"
export PYTHONPATH="${PYTHONPATH}:$(pwd)/verl-agent"
export PYTHONPATH="${PYTHONPATH}:$(pwd)/integration"

# 或在Python中添加
import sys
sys.path.insert(0, '/path/to/AFlow')
sys.path.insert(0, '/path/to/verl-agent')
sys.path.insert(0, '/path/to/integration')
```

### 问题2: Ray初始化失败

**错误 Error:**
```
RuntimeError: Ray has not been started yet
```

**解决 Solution:**
```python
# 在代码中
import ray
if not ray.is_initialized():
    ray.init()
```

### 问题3: OpenAI API错误

**错误 Error:**
```
openai.error.AuthenticationError: Invalid API key
```

**解决 Solution:**
```bash
# 检查环境变量
echo $OPENAI_API_KEY

# 重新设置
export OPENAI_API_KEY="sk-your-correct-api-key"

# 或在配置文件中直接指定
# opt_llm_config:
#   api_key: "sk-your-api-key"
```

### 问题4: CUDA内存不足

**错误 Error:**
```
RuntimeError: CUDA out of memory
```

**解决 Solution:**
```yaml
# 在配置中减少并行数
environment:
  env_num: 2  # 从4降到2
  group_n: 1  # 从2降到1

# 或使用CPU
device: "cpu"
```

### 问题5: 训练很慢

**解决 Solution:**

1. 减少环境数量:
```yaml
environment:
  env_num: 2
  max_rounds: 10  # 从20降到10
```

2. 使用更快的LLM:
```yaml
opt_llm_config:
  model: "gpt-3.5-turbo"  # 代替gpt-4
```

3. 减少验证轮次:
```yaml
environment:
  validation_rounds: 3  # 从5降到3
```

---

## 📊 验证安装 Verify Installation

运行以下脚本验证所有组件正常:

```python
# verify_installation.py
import sys
import os

print("Verifying installation...")

# 1. Check imports
try:
    from integration.unified_state import WorkflowState, StateManager
    print("✓ unified_state imported")
except ImportError as e:
    print(f"✗ unified_state import failed: {e}")

try:
    from AFlow.scripts.shared_experience import SharedExperiencePool
    print("✓ shared_experience imported")
except ImportError as e:
    print(f"✗ shared_experience import failed: {e}")

try:
    from AFlow.scripts.optimizer_rl import RLEnhancedOptimizer
    print("✓ optimizer_rl imported")
except ImportError as e:
    print(f"✗ optimizer_rl import failed: {e}")

try:
    from verl_agent.gigpo.workflow_gigpo import compute_workflow_gigpo_advantage
    print("✓ workflow_gigpo imported")
except ImportError as e:
    print(f"✗ workflow_gigpo import failed: {e}")

# 2. Check Ray
try:
    import ray
    if not ray.is_initialized():
        ray.init()
    print("✓ Ray initialized")
except Exception as e:
    print(f"✗ Ray initialization failed: {e}")

# 3. Check configuration
try:
    import yaml
    with open('integration/deep_config.yaml') as f:
        config = yaml.safe_load(f)
    print("✓ Configuration loaded")
except Exception as e:
    print(f"✗ Configuration loading failed: {e}")

print("\nInstallation verification complete!")
```

运行验证:
```bash
python verify_installation.py
```

---

## 🎯 示例工作流 Example Workflow

### 完整示例: 训练HumanEval

```bash
# 1. 准备环境
cd "/Users/zhangmingda/Desktop/agent worflow"
export OPENAI_API_KEY="your-key-here"

# 2. 创建自定义配置
cat > integration/humaneval_config.yaml << EOF
device: "cuda"
output_dir: "./output/humaneval_experiment"
total_epochs: 10
episodes_per_epoch: 20

environment:
  train_datasets:
    - "HumanEval"
  test_datasets:
    - "HumanEval"

  env_num: 2
  group_n: 2
  max_rounds: 15

  opt_llm_config:
    model: "gpt-4"
    temperature: 0.7

rl:
  initial_weight: 0.4
  weight_schedule: "linear_increase"
EOF

# 3. 启动训练
cd integration
python deep_train.py --config humaneval_config.yaml 2>&1 | tee training.log

# 4. 监控（另一个终端）
watch -n 5 'tail -20 output/humaneval_experiment/logs/training.log'

# 5. 训练完成后查看结果
cat output/humaneval_experiment/logs/training_stats.json | jq '.best_score'

# 6. 可视化
python << EOF
import json
import matplotlib.pyplot as plt

with open('output/humaneval_experiment/logs/training_stats.json') as f:
    stats = json.load(f)

plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.plot(stats['avg_scores'])
plt.title('Average Score')
plt.xlabel('Epoch')
plt.ylabel('Score')

plt.subplot(1, 3, 2)
plt.plot(stats['experience_pool_size'])
plt.title('Experience Pool Size')
plt.xlabel('Epoch')

plt.subplot(1, 3, 3)
plt.plot(stats['state_manager_size'])
plt.title('State Manager Size')
plt.xlabel('Epoch')

plt.tight_layout()
plt.savefig('training_analysis.png')
print("Saved to training_analysis.png")
EOF
```

---

## 📚 更多资源 More Resources

### 文档 Documentation

- **完整文档**: `integration/README.md`
- **实现总结**: `IMPLEMENTATION_SUMMARY.md`
- **交付清单**: `DELIVERABLES_CHECKLIST.md`

### 配置示例 Configuration Examples

- **基础配置**: `integration/deep_config.yaml`
- **最小配置**: 见上文 "Minimal Configuration Changes"

### 代码示例 Code Examples

```python
# 自定义RL策略
class MyPolicy:
    def get_q_value(self, state_repr: str) -> float:
        # 您的Q值估计逻辑
        return 0.5

    def suggest_action(self, state_repr: str) -> str:
        # 您的动作建议逻辑
        return "Optimize the workflow structure"

    def get_action(self, obs: str) -> str:
        # 您的动作生成逻辑
        return "Add ScEnsemble operator"

# 使用自定义策略
from integration.deep_train import DeepIntegratedTrainer

trainer = DeepIntegratedTrainer(config)
trainer.set_rl_policy(MyPolicy())
trainer.train()
```

---

## 🆘 获取帮助 Get Help

### 检查日志 Check Logs

```bash
# 详细错误信息
cat output/deep_integration/logs/training.log | grep ERROR

# 警告信息
cat output/deep_integration/logs/training.log | grep WARNING
```

### 启用详细日志 Enable Verbose Logging

```yaml
# 在配置中
logging:
  level: "DEBUG"

debug:
  verbose_env: true
  verbose_rl: true
  verbose_gigpo: true
```

### 常见错误模式 Common Error Patterns

1. **路径问题**: 确保所有`sys.path`设置正确
2. **依赖问题**: 运行`pip list`检查已安装包
3. **配置问题**: 验证YAML语法正确
4. **API问题**: 检查API密钥和网络连接
5. **资源问题**: 监控CPU/GPU/内存使用

---

## ✅ 检查清单 Checklist

开始训练前，确保:

- [ ] Python 3.8+ 已安装
- [ ] 所有依赖已安装 (`pip install ...`)
- [ ] 环境变量已设置 (OPENAI_API_KEY)
- [ ] 配置文件已准备
- [ ] 输出目录有写权限
- [ ] (可选) GPU可用且CUDA配置正确
- [ ] (可选) Ray集群可用

---

## 🎉 开始训练！Start Training!

一切准备就绪，开始您的深度集成训练之旅！

Everything is ready, start your deep integration training journey!

```bash
cd integration
python deep_train.py --config deep_config.yaml
```

祝训练顺利！Good luck with your training!

---

**快速参考 Quick Reference:**
- 配置文件: `integration/deep_config.yaml`
- 训练脚本: `integration/deep_train.py`
- 完整文档: `integration/README.md`
- 实现总结: `IMPLEMENTATION_SUMMARY.md`
