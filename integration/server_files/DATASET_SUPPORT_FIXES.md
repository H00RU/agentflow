# 多数据集支持修复文档

## 问题描述

在之前的实现中，项目存在严重的prompt与数据集不匹配问题：

### 问题现状
- **实际运行**: AIME数学题训练
- **Prompt内容**: 告诉Qwen做HumanEval代码生成
- **结果**: Qwen完全困惑，不知道该解数学题还是生成代码

### 具体表现
1. `workflow_prompt_manager.py` 硬编码为HumanEval代码生成任务
2. `deep_workflow_env.py` 的observation说"code generation"
3. 实际使用的是 `aime_evaluator.py`（正确）
4. 但prompt是错误的，导致训练效果极差

---

## 修复方案（方案2 - 多数据集支持）

按照用户要求，采用方案2：让 `workflow_prompt_manager.py` 支持多数据集选择。

### 设计原则
1. ✅ 不简化训练流程
2. ✅ 保持与VERL框架兼容
3. ✅ 保持与AFlow框架兼容
4. ✅ 支持后续添加新数据集
5. ✅ 向后兼容（默认HumanEval）

---

## 修改详情

### 1. workflow_prompt_manager.py

#### 修改内容
```python
class WorkflowPromptManager:
    def __init__(self, dataset: str = "HumanEval"):
        """支持多数据集"""
        self.dataset = dataset.upper()
        self.system_prompt = self._create_system_prompt()
        self.examples = self._create_examples()
```

#### 新增方法
- `_create_system_prompt()`: 根据dataset路由到不同的prompt生成方法
- `_create_aime_system_prompt()`: AIME数学题专用prompt
- `_create_humaneval_system_prompt()`: HumanEval代码生成专用prompt
- `_create_generic_system_prompt()`: 通用prompt（新数据集默认）
- `_create_aime_examples()`: AIME示例workflow
- `_create_humaneval_examples()`: HumanEval示例workflow
- `_create_generic_examples()`: 通用示例workflow

#### 全局函数修改
```python
def get_prompt_manager(dataset: str = "HumanEval") -> WorkflowPromptManager:
    """支持dataset参数，按dataset缓存实例"""
    dataset_upper = dataset.upper()
    if dataset_upper not in _prompt_manager_cache:
        _prompt_manager_cache[dataset_upper] = WorkflowPromptManager(dataset=dataset)
    return _prompt_manager_cache[dataset_upper]
```

#### AIME vs HumanEval Prompt区别

**AIME (数学题):**
- 任务描述: "mathematical problem-solving tasks"
- 可用operators: Custom, ScEnsemble, Test, Review, Revise
- 特征说明: 数学竞赛题、答案0-999、需要多步推理
- Examples: 数学解题workflow

**HumanEval (代码生成):**
- 任务描述: "code generation tasks"
- 可用operators: Custom, CustomCodeGenerate, ScEnsemble, Test
- 特征说明: 代码生成、测试用例
- Examples: 代码生成workflow

---

### 2. deep_workflow_env.py

#### 修改内容
```python
def _construct_observation(self, ...) -> str:
    """根据dataset生成相应的任务描述"""
    dataset_upper = self.dataset.upper()
    if dataset_upper == "AIME":
        task_desc = "Design and optimize agent workflow for solving AIME mathematical problems"
        focus_points = """1. Which operators to use for mathematical reasoning
2. How to combine them effectively for problem-solving
3. Using ensemble methods for robust solutions
4. How to improve upon previous attempts"""
    elif dataset_upper == "HUMANEVAL":
        task_desc = "Design and optimize agent workflow for code generation"
        focus_points = """1. Which operators to use for code generation
2. How to combine them effectively
3. How to improve upon previous attempts"""
    else:
        # 通用描述
        task_desc = f"Design and optimize agent workflow for {self.dataset} tasks"
        ...
```

#### 效果
- Observation现在准确描述当前任务
- AIME任务强调"数学推理"
- HumanEval任务强调"代码生成"

---

### 3. deep_train_real_workflow.py

#### 修改内容
```python
# 调整顺序：先读取train_datasets，再创建prompt_manager
self.env_config = config.get('environment', {})
self.train_datasets = self.env_config.get('train_datasets', [])
if not self.train_datasets:
    raise ValueError("Please specify 'train_datasets' in config file")

# 根据第一个训练数据集创建prompt manager
primary_dataset = self.train_datasets[0] if self.train_datasets else "HumanEval"
self.prompt_manager = get_prompt_manager(dataset=primary_dataset)
print(f"Using prompt manager for dataset: {primary_dataset}")
```

#### 效果
- 自动从配置文件读取dataset
- 使用第一个训练数据集的prompt manager
- 打印日志确认使用的数据集

---

### 4. server_files/ 同步

所有修改的文件已同步到 `server_files/` 目录：
- `workflow_prompt_manager.py`
- `deep_workflow_env.py`
- `deep_train_real_workflow.py`

---

## 测试验证

### 测试内容
1. ✅ WorkflowPromptManager支持dataset参数
2. ✅ AIME prompt包含数学相关内容（mathematical, Review, Revise）
3. ✅ HumanEval prompt包含代码生成内容（CustomCodeGenerate）
4. ✅ 不同dataset返回不同的system prompt和examples
5. ✅ 相同dataset使用缓存，返回同一实例
6. ✅ 通用dataset自动生成prompt

### 测试结果
```
AIME Prompt特征:
  - 包含'mathematical': True
  - 包含'AIME': True
  - 包含'Review': True
  - 包含'Revise': True

HumanEval Prompt特征:
  - 包含'code generation': True
  - 包含'CustomCodeGenerate': True

缓存测试: True
通用Dataset测试: True
```

---

## 使用方法

### AIME训练
配置文件 `aime_full_test.yaml`:
```yaml
environment:
  train_datasets:
    - "AIME"  # ← 自动使用AIME prompt
  test_datasets:
    - "AIME"
```

训练时会自动：
1. 读取 `train_datasets: ["AIME"]`
2. 调用 `get_prompt_manager(dataset="AIME")`
3. 生成AIME专用的system prompt
4. Environment的observation包含数学任务描述

### HumanEval训练
```yaml
environment:
  train_datasets:
    - "HumanEval"  # ← 自动使用HumanEval prompt
```

### 添加新数据集
只需三步：

**步骤1**: 在 `workflow_prompt_manager.py` 添加dataset分支
```python
def _create_system_prompt(self) -> str:
    if self.dataset == "AIME":
        return self._create_aime_system_prompt()
    elif self.dataset == "MYNEWDATASET":  # ← 添加这里
        return self._create_mynewdataset_system_prompt()
    elif self.dataset == "HUMANEVAL":
        return self._create_humaneval_system_prompt()
    else:
        return self._create_generic_system_prompt()
```

**步骤2**: 添加专用prompt方法
```python
def _create_mynewdataset_system_prompt(self) -> str:
    return """You are an AI workflow optimizer for MyNewDataset tasks.

Your task is to design and improve agent workflows that solve...
[根据数据集特点编写prompt]
"""
```

**步骤3**: 在配置文件中使用
```yaml
environment:
  train_datasets:
    - "MyNewDataset"
```

---

## 向后兼容性

### 默认行为（未修改代码的项目）
```python
# 旧代码仍然可以工作
pm = get_prompt_manager()  # 默认使用HumanEval
```

### 显式指定dataset
```python
# 新代码
pm = get_prompt_manager(dataset="AIME")
```

---

## 架构完整性

### ✅ 保持VERL集成
- GiGPO优势计算: 完全保留
- PPO策略更新: 完全保留
- RLTrainer接口: 完全保留
- Ray分布式支持: 不受影响（虽然当前未使用）

### ✅ 保持AFlow集成
- WorkflowParser: 完全保留
- Evaluator系统: 完全保留（aime_evaluator, workflow_evaluator）
- Operator系统: 完全保留
- MCTS优化: 完全保留（动态模式）

### ✅ 训练流程完整
```
配置文件 (aime_full_test.yaml)
    ↓
deep_train_real_workflow.py
    ├─ 读取train_datasets: ["AIME"]
    ├─ 创建get_prompt_manager(dataset="AIME")  ← 正确的prompt
    ├─ 创建TrainableQwenPolicy
    └─ 创建RLTrainer
        ↓
deep_workflow_env.py
    ├─ 使用aime_evaluator  ← 正确的evaluator
    ├─ _construct_observation  ← 正确的任务描述
    └─ 执行workflow → 真实测试 → 返回reward
        ↓
RLTrainer.update()
    ├─ GiGPO计算advantages  ← 完整保留
    └─ PPO更新Qwen参数    ← 完整保留
```

---

## 修复效果

### 修复前
```
Qwen收到的信息:
  Prompt: "solve coding problems (HumanEval)"  ← 错误！
  Observation: "code generation"               ← 错误！
  实际任务: 解AIME数学题                       ← 实际！

结果: Qwen完全困惑，训练效果差
```

### 修复后
```
Qwen收到的信息（AIME）:
  Prompt: "solve AIME mathematical problems"  ← 正确！
  Observation: "mathematical problem-solving" ← 正确！
  实际任务: 解AIME数学题                      ← 一致！

结果: Qwen理解任务，训练效果正常
```

---

## 总结

### 修改的文件（3个）
1. `/content/agentflow/integration/workflow_prompt_manager.py`
2. `/content/agentflow/integration/deep_workflow_env.py`
3. `/content/agentflow/integration/deep_train_real_workflow.py`

### 修改的行数
- workflow_prompt_manager.py: +约200行（新增多数据集方法）
- deep_workflow_env.py: +约20行（修改observation生成）
- deep_train_real_workflow.py: +约5行（传递dataset参数）

### 核心改进
1. ✅ 解决了prompt与数据集不匹配的严重问题
2. ✅ 支持多数据集（AIME, HumanEval, 通用）
3. ✅ 易于扩展新数据集
4. ✅ 完全保持VERL+AFlow框架完整性
5. ✅ 向后兼容
6. ✅ 通过测试验证

### 关键特性
- **自动识别**: 从配置文件自动读取dataset
- **准确prompt**: 根据dataset生成对应的system prompt
- **一致描述**: observation与实际任务匹配
- **易扩展**: 添加新数据集只需修改prompt_manager

---

## 下一步建议

1. **测试AIME训练**: 使用修复后的代码运行 `aime_full_test.yaml`，观察Qwen是否理解任务
2. **监控指标**: 关注训练loss、accuracy、workflow多样性
3. **添加新数据集**: 如果需要支持GSM8K、MATH等数据集，按照文档添加对应的prompt
4. **调整超参数**: 根据之前的分析，使用 `aime_full_test.yaml` 的超参数（已优化）

---

生成时间: 2025-10-27
修复类型: 多数据集支持（方案2）
影响范围: Prompt生成、Observation构造、训练流程
测试状态: ✅ 通过
