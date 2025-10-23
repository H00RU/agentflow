# 📚 如何添加新数据集 - 完全无硬编码！

本文档说明如何在 AgentFlow 项目中添加新数据集，无需修改多处硬编码。

---

## 🎯 概述

AgentFlow 使用 `DatasetClassifier` 统一管理所有数据集的分类和配置。添加新数据集只需一行代码！

---

## 📝 方法1：动态添加（推荐用于测试）

在代码中动态注册新数据集：

```python
from workflow_parser import DatasetClassifier

# 添加新的数学数据集
DatasetClassifier.add_dataset(
    dataset='MYMATH',           # 数据集名称
    category='math',            # 类别: 'math', 'code', 或 'qa'
    sample_count=15             # 可选：自定义默认采样数
)

# 添加新的代码数据集
DatasetClassifier.add_dataset(
    dataset='MYCODE',
    category='code',
    sample_count=5
)

# 添加新的问答数据集
DatasetClassifier.add_dataset(
    dataset='MYQA',
    category='qa',
    sample_count=3
)
```

---

## 🔧 方法2：修改类定义（推荐用于生产）

编辑 `workflow_parser.py` 中的 `DatasetClassifier` 类：

```python
class DatasetClassifier:
    """
    数据集分类器 - 统一管理数据集类型判断
    """

    # 1️⃣ 在对应的集合中添加数据集名称
    CODE_DATASETS: Set[str] = {
        "HUMANEVAL", "MBPP", "CODEEVAL",
        "APPS", "CODEX",
        "MYCODE"  # ← 添加你的新代码数据集
    }

    MATH_DATASETS: Set[str] = {
        "AIME", "MATH", "GSM8K",
        "MATHQA", "SVAMP", "AQUA",
        "MYMATH"  # ← 添加你的新数学数据集
    }

    QA_DATASETS: Set[str] = {
        "HOTPOTQA", "DROP", "SQUAD", "NATURALQA",
        "MYQA"  # ← 添加你的新问答数据集
    }

    # 2️⃣ （可选）设置特定的默认采样数
    DEFAULT_SAMPLE_COUNTS: Dict[str, int] = {
        "AIME": 20,      # 最难的数学竞赛
        "MATH": 10,      # 中等难度数学
        "GSM8K": 5,      # 较简单的数学
        "HUMANEVAL": 3,  # 代码生成
        "MYMATH": 15,    # ← 添加自定义采样数
        "MYCODE": 5,     # ← 添加自定义采样数
    }
```

---

## 🚀 示例：添加 MMLU 数据集

### 步骤1：注册数据集

```python
from workflow_parser import DatasetClassifier, WorkflowParser

# 注册 MMLU 为问答数据集，默认采样5次
DatasetClassifier.add_dataset('MMLU', 'qa', sample_count=5)
```

### 步骤2：立即使用

```python
# 创建 parser
parser = WorkflowParser()

# 生成 MMLU 的 workflow
spec = parser.parse_qwen_output(
    '<operators>Custom, ScEnsemble</operators>',
    dataset_type='MMLU'
)

# 生成的 workflow 会自动：
#   - 使用 QA 类型的默认设置
#   - 采样 5 次
#   - 不需要 entry_point 参数（不是代码任务）
```

### 步骤3：在配置文件中使用

```yaml
environment:
  train_datasets:
    - "MMLU"  # ← 直接使用新数据集

  workflow_sample_count: 5  # 可选：覆盖默认采样数
```

---

## 📊 现有支持的数据集

### 代码生成 (CODE_DATASETS)

| 数据集 | 默认采样数 | 说明 |
|--------|-----------|------|
| HumanEval | 3 | Python 代码生成 |
| MBPP | 3 | Python 基础编程 |
| CODEEVAL | 3 | 代码评测 |
| APPS | 3 | 应用级编程 |
| CODEX | 3 | OpenAI Codex 测试集 |

**特点**：
- 需要 `entry_point` 参数
- 使用 `CustomCodeGenerate` operator
- 默认采样较少（代码生成成本高）

---

### 数学推理 (MATH_DATASETS)

| 数据集 | 默认采样数 | 难度 |
|--------|-----------|------|
| AIME | 20 | ⭐⭐⭐⭐⭐ 最难 |
| MATH | 10 | ⭐⭐⭐⭐ 中等 |
| GSM8K | 5 | ⭐⭐⭐ 较易 |
| MATHQA | 5 | ⭐⭐⭐ 较易 |
| SVAMP | 5 | ⭐⭐⭐ 较易 |
| AQUA | 5 | ⭐⭐⭐⭐ 中等 |

**特点**：
- 不需要 `entry_point` 参数
- 使用 `Custom` operator（让 LLM 直接推理）
- 采样数根据难度调整（越难采样越多）

---

### 问答任务 (QA_DATASETS)

| 数据集 | 默认采样数 | 类型 |
|--------|-----------|------|
| HOTPOTQA | 3 | 多跳推理 |
| DROP | 3 | 阅读理解+计算 |
| SQUAD | 3 | 阅读理解 |
| NATURALQA | 3 | 开放域问答 |

**特点**：
- 不需要 `entry_point` 参数
- 使用 `Custom` operator
- 默认采样适中

---

## 🎨 数据集类别说明

### 1. **CODE (代码生成)**

```python
category='code'
```

**适用于**：
- 需要生成可执行代码的任务
- 需要 `entry_point` 参数（函数名）
- 需要代码测试的任务

**自动行为**：
- Workflow 签名: `async def __call__(self, problem: str, entry_point: str)`
- 使用 `CustomCodeGenerate` operator
- 默认采样: 3 次

---

### 2. **MATH (数学推理)**

```python
category='math'
```

**适用于**：
- 数学问题求解
- 需要数值计算
- 需要符号推理

**自动行为**：
- Workflow 签名: `async def __call__(self, problem: str, entry_point: Optional[str] = None)`
- 使用 `Custom` operator（LLM 直接推理）
- 默认采样: 5 次（根据难度可调整）
- Instruction: "Solve this math problem step by step..."

---

### 3. **QA (问答任务)**

```python
category='qa'
```

**适用于**：
- 阅读理解
- 知识问答
- 推理任务

**自动行为**：
- Workflow 签名: `async def __call__(self, problem: str, entry_point: Optional[str] = None)`
- 使用 `Custom` operator
- 默认采样: 3 次

---

## ⚙️ 自定义采样数

### 全局默认采样数

在 `DatasetClassifier` 中设置：

```python
DEFAULT_SAMPLE_COUNTS: Dict[str, int] = {
    "AIME": 20,      # AIME 很难，需要20次采样
    "MYNEW": 25,     # 自定义数据集，需要25次
}
```

### 配置文件中覆盖

在 YAML 配置文件中：

```yaml
environment:
  train_datasets:
    - "AIME"

  # 覆盖默认采样数（对所有数据集生效）
  workflow_sample_count: 30  # AIME 将使用30次采样而非默认的20次
```

### 代码中动态指定

```python
spec = parser.parse_qwen_output(
    qwen_output,
    dataset_type='AIME',
    sample_count=25  # 本次使用25次采样
)
```

**优先级**：`代码指定 > 配置文件 > 数据集特定默认值 > 类别默认值`

---

## 🔍 未知数据集的处理

如果使用了未注册的数据集，系统会：

1. ✅ **不会崩溃**
2. ✅ **使用合理的默认行为**：
   - 不属于任何类别
   - 使用 `Custom` operator
   - 默认采样 3 次
   - 不需要 `entry_point`

```python
# 即使 "UNKNOWN" 未注册，也能正常工作
spec = parser.parse_qwen_output(
    '<operators>Custom</operators>',
    dataset_type='UNKNOWN'
)
# 生成的 workflow 会使用默认的通用设置
```

---

## 💡 最佳实践

### ✅ 推荐做法

1. **生产环境**：在 `DatasetClassifier` 类中添加（方法2）
   - 代码更清晰
   - 便于版本控制
   - 团队成员都能看到

2. **测试/实验**：使用动态添加（方法1）
   - 快速验证
   - 不影响主代码
   - 便于尝试不同配置

3. **设置合理的采样数**：
   - 简单任务：3-5 次
   - 中等任务：5-10 次
   - 困难任务：10-20 次
   - 极难任务：20+ 次

### ❌ 避免

1. ❌ 不要在多个地方硬编码数据集名称
2. ❌ 不要为每个数据集创建单独的处理函数
3. ❌ 不要忽略采样数的设置（会影响性能）

---

## 🧪 测试新数据集

添加数据集后，建议进行测试：

```python
from workflow_parser import DatasetClassifier, WorkflowParser

# 1. 添加数据集
DatasetClassifier.add_dataset('MYTEST', 'math', sample_count=10)

# 2. 测试分类
assert DatasetClassifier.is_math_dataset('MYTEST')
assert not DatasetClassifier.is_code_dataset('MYTEST')

# 3. 测试采样数
assert DatasetClassifier.get_default_sample_count('MYTEST') == 10

# 4. 测试 workflow 生成
parser = WorkflowParser()
spec = parser.parse_qwen_output(
    '<operators>Custom, ScEnsemble</operators>',
    dataset_type='MYTEST'
)

# 5. 验证生成的代码
assert 'range(10)' in spec.workflow_code  # 应该有10次采样
assert 'await self.custom(' in spec.workflow_code  # 应该使用 Custom
print("✅ 新数据集测试通过！")
```

---

## 📞 常见问题

### Q1: 如何修改现有数据集的采样数？

**A**: 有三种方法：

```python
# 方法1: 修改 DEFAULT_SAMPLE_COUNTS
DEFAULT_SAMPLE_COUNTS["AIME"] = 30

# 方法2: 在配置文件中覆盖
workflow_sample_count: 30

# 方法3: 代码中指定
spec = parser.parse_qwen_output(..., sample_count=30)
```

### Q2: 可以添加多个类别吗（既是数学又是代码）？

**A**: 不建议。每个数据集应该有明确的类别。如果任务混合，建议：
- 创建两个独立的数据集配置
- 或根据主要任务类型选择类别

### Q3: 如何知道数据集是否需要 entry_point？

**A**: 只有代码生成任务需要：
- 需要：HumanEval, MBPP 等代码数据集
- 不需要：AIME, MATH, GSM8K 等数学/问答数据集

### Q4: 未知数据集会使用什么默认值？

**A**:
- 类别：无（通用）
- Operator: Custom
- 采样数：3
- Entry point: 可选（Optional）

---

## 🎉 总结

### 优点

✅ **无硬编码**：所有数据集判断集中管理
✅ **易扩展**：添加新数据集只需一行代码
✅ **类型安全**：使用 Set 和 Dict 管理
✅ **智能默认值**：根据类别自动推断
✅ **向后兼容**：不影响现有代码
✅ **灵活配置**：多层级的采样数配置

### 核心文件

- `integration/workflow_parser.py` - DatasetClassifier 类定义
- `integration/deep_workflow_env.py` - 使用 DatasetClassifier
- 配置文件 (*.yaml) - 数据集配置

---

**祝您使用愉快！** 🚀

如有问题，请查看代码中的文档字符串或联系维护者。
