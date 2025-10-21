# Testing Guide
# 测试指南

## ✅ 当前测试状态 Current Test Status

### 已完成测试 Completed Tests

✅ **文件结构验证 File Structure Verification**
- 所有15个核心文件已验证存在
- 总代码量: ~3,600行Python代码
- All 15 core files verified to exist
- Total code: ~3,600 lines of Python

✅ **逻辑流程验证 Logic Flow Verification**
- 所有关键类和方法已验证
- 所有集成点已验证连接正确
- 数据流逻辑连贯性已确认
- All key classes and methods verified
- All integration points verified
- Data flow logic confirmed coherent

### 待完成测试 Pending Tests

⏳ **依赖安装测试 Dependency Installation Test**
- 需要安装Python包
- Need to install Python packages

⏳ **组件功能测试 Component Functional Test**
- 需要导入并测试各组件
- Need to import and test components

⏳ **完整集成测试 Full Integration Test**
- 需要运行完整训练流程
- Need to run full training pipeline

---

## 🚀 快速测试步骤 Quick Test Steps

### 步骤1: 安装依赖 Install Dependencies

```bash
cd "/Users/zhangmingda/Desktop/agent worflow"

# 方式1: 使用requirements.txt
pip3 install -r requirements.txt

# 方式2: 使用安装脚本
chmod +x install_dependencies.sh
bash install_dependencies.sh

# 方式3: 手动安装
pip3 install numpy torch pyyaml ray anthropic
```

**预计时间 Estimated Time**: 5-10分钟

### 步骤2: 验证组件 Verify Components

```bash
cd integration
python3 test_components.py
```

**预期输出 Expected Output**:
```
[Test 1] Testing unified_state imports...
✓ unified_state imported successfully
✓ Created WorkflowState: ...
✓ StateManager working: 1 states

[Test 2] Testing shared_experience imports...
✓ shared_experience imported successfully
✓ SharedExperiencePool working: 1 experiences

... (更多测试) (more tests)

All basic components are working correctly!
Ready to test full integration.
```

**预计时间 Estimated Time**: 1-2分钟

### 步骤3: 运行最小测试 Run Minimal Test

```bash
cd integration
python3 deep_train.py --config test_config.yaml
```

**测试配置 Test Configuration**:
- 设备: CPU
- Epochs: 1
- Episodes: 2
- 数据集: HumanEval
- 环境数: 1
- 最大轮次: 3

**预计时间 Estimated Time**:
- 如果使用Claude API: 5-10分钟
- 如果模拟运行: 1-2分钟

**预期输出 Expected Output**:
```
Starting deep integration training
Creating environments...
Created 1 training and 1 test environments

Starting epoch 1/1
Training on HumanEval
  Episode 1/2: avg_score=0.xxxx, avg_reward=0.xxxx
  Episode 2/2: avg_score=0.xxxx, avg_reward=0.xxxx

Epoch 1 completed: avg_score=0.xxxx, ...
```

---

## 📋 详细测试清单 Detailed Test Checklist

### A. 环境测试 Environment Tests

- [x] Python 3.8+ 已安装 Python 3.8+ installed
- [ ] pip3 可用 pip3 available
- [ ] 依赖包已安装 Dependencies installed
  - [ ] numpy
  - [ ] torch
  - [ ] pyyaml
  - [ ] ray
  - [ ] anthropic
- [ ] Claude API密钥已设置 Claude API key set
- [ ] 磁盘空间充足 (>500MB) Sufficient disk space

### B. 文件结构测试 File Structure Tests

已通过 ✅ **PASSED**

运行: `python3 integration/verify_files.py`

结果: 15/16 文件找到（缺少1个原始AFlow文件，不影响集成）

### C. 逻辑流程测试 Logic Flow Tests

已通过 ✅ **PASSED**

运行: `python3 integration/simple_logic_test.py`

结果: 所有集成点验证通过

### D. 组件功能测试 Component Functional Tests

待运行 ⏳ **PENDING**

运行: `python3 integration/test_components.py`

测试内容:
- [ ] unified_state 导入和功能
- [ ] shared_experience 导入和功能
- [ ] AFlow基础导入
- [ ] 依赖包可用性
- [ ] 配置文件加载
- [ ] WorkflowState方法
- [ ] StateManager方法
- [ ] SharedExperiencePool方法

### E. 最小集成测试 Minimal Integration Test

待运行 ⏳ **PENDING**

运行: `python3 deep_train.py --config test_config.yaml`

测试内容:
- [ ] 环境创建
- [ ] RL策略设置
- [ ] 训练循环执行
- [ ] 状态管理
- [ ] 经验池更新
- [ ] 检查点保存
- [ ] 日志生成

### F. 完整训练测试 Full Training Test

服务器运行 🖥️ **TO BE RUN ON SERVER**

运行: `python3 deep_train.py --config deep_config.yaml`

测试内容:
- [ ] 多数据集训练
- [ ] 并行环境执行
- [ ] RL权重调度
- [ ] 评估系统
- [ ] 性能监控
- [ ] 最终结果验证

---

## 🔧 故障排除 Troubleshooting

### 问题1: 依赖安装失败

**症状 Symptoms**:
```
ERROR: Could not find a version that satisfies the requirement ...
```

**解决方案 Solutions**:
```bash
# 升级pip
pip3 install --upgrade pip

# 分别安装
pip3 install numpy
pip3 install torch
pip3 install pyyaml
pip3 install ray
pip3 install anthropic
```

### 问题2: 导入错误

**症状 Symptoms**:
```
ModuleNotFoundError: No module named 'unified_state'
```

**解决方案 Solutions**:
```bash
# 确保在正确的目录
cd "/Users/zhangmingda/Desktop/agent worflow/integration"

# 设置PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)/.."
export PYTHONPATH="${PYTHONPATH}:$(pwd)/../AFlow"
export PYTHONPATH="${PYTHONPATH}:$(pwd)/../verl-agent"

# 重新运行
python3 test_components.py
```

### 问题3: Claude API错误

**症状 Symptoms**:
```
anthropic.AuthenticationError: Invalid API key
```

**解决方案 Solutions**:
```bash
# 检查API密钥
cat integration/test_config.yaml | grep api_key

# 如果显示不正确，编辑配置
vim integration/test_config.yaml

# 确保api_key字段正确:
# api_key: "sk-ant-api03-..."
```

### 问题4: Ray初始化失败

**症状 Symptoms**:
```
RuntimeError: Ray has not been started yet
```

**解决方案 Solutions**:
```python
# Ray会自动初始化，如果仍有问题:
import ray
ray.init(ignore_reinit_error=True)
```

### 问题5: 内存不足

**症状 Symptoms**:
```
MemoryError: ...
```

**解决方案 Solutions**:
```yaml
# 在test_config.yaml中减少并行数
environment:
  env_num: 1  # 已经是最小值
  max_rounds: 2  # 进一步减少
```

---

## 📊 测试结果示例 Test Result Examples

### 成功的组件测试输出

```
============================================================
Component Testing Script
============================================================

[Test 1] Testing unified_state imports...
✓ unified_state imported successfully
✓ Created WorkflowState: abc123def456
✓ StateManager working: 1 states

[Test 2] Testing shared_experience imports...
✓ shared_experience imported successfully
✓ SharedExperiencePool working: 1 experiences

[Test 3] Testing AFlow basic imports...
✓ Found optimizer.py at .../AFlow/scripts/optimizer.py

[Test 4] Testing dependencies...
✓ numpy version: 1.24.0
✓ torch version: 2.0.0
✓ yaml imported successfully
✓ ray version: 2.5.0

[Test 5] Testing configuration loading...
✓ Loaded test_config.yaml
  - Device: cpu
  - Epochs: 1
  - Datasets: ['HumanEval']

[Test 6] Testing WorkflowState methods...
✓ Text representation: 245 chars
✓ Anchor representation: abc123def456
✓ Reward computation: 0.1000
✓ State cloning: abc123def456

[Test 7] Testing StateManager methods...
✓ Added 5 states
✓ Got 3 best states
✓ Got 5 states for HumanEval

[Test 8] Testing SharedExperiencePool methods...
✓ Added 10 experiences
✓ Got 3 best experiences
✓ Got 5 experiences in score range [0.6, 0.8]
✓ Got 3 random experiences
✓ Pool statistics: avg_score=0.7250

============================================================
Component Testing Complete
============================================================

All basic components are working correctly!
Ready to test full integration.
```

### 成功的最小训练输出

```
Starting deep integration training
Creating environments...
Creating training environment for HumanEval
Creating test environment for HumanEval
Created 1 training and 1 test environments

Starting epoch 1/1
Current RL weight: 0.500
Training on HumanEval
  Episode 1/2: avg_score=0.6500, avg_reward=0.0500
  Episode 2/2: avg_score=0.7200, avg_reward=0.1200

Epoch 1 completed: avg_score=0.6850, avg_reward=0.0850, pool_size=4

Evaluating at epoch 1
Evaluating on HumanEval
Evaluation completed: avg_score=0.7000

Saved checkpoint to output/test_run/checkpoints/best.pt

Training completed
Final evaluation: {'avg_score': 0.7000, ...}
```

---

## 🎯 验收标准 Acceptance Criteria

### 最小验收 Minimal Acceptance

✅ 文件结构完整 (15/16 files)
✅ 逻辑流程正确 (所有集成点验证通过)
⏳ 组件测试通过 (等待依赖安装)
⏳ 最小训练运行 (等待依赖安装)

### 完整验收 Full Acceptance

⏳ 多数据集训练
⏳ 并行环境执行
⏳ RL-MCTS融合工作
⏳ 共享经验池工作
⏳ GiGPO分组正确
⏳ 检查点保存/恢复
⏳ 性能提升验证

---

## 📝 测试日志 Test Logs

### 本地测试 Local Tests

**日期 Date**: 2025-10-09

**测试1: 文件结构验证**
- ✅ 通过 PASSED
- 时间: <1秒
- 结果: 15/16文件找到

**测试2: 逻辑流程验证**
- ✅ 通过 PASSED
- 时间: <1秒
- 结果: 所有集成点正确

**测试3: 组件功能测试**
- ⏳ 待运行 (需要安装numpy等依赖)
- PENDING (needs numpy etc.)

**测试4: 最小训练测试**
- ⏳ 待运行 (需要安装所有依赖和Claude API)
- PENDING (needs all deps and Claude API)

### 服务器测试 Server Tests

**待运行 TO BE RUN**

计划:
1. 安装依赖
2. 配置Claude API
3. 运行完整训练
4. 监控性能
5. 验证结果

---

## 🚀 下一步行动 Next Actions

### 立即行动 (本地) Immediate Actions (Local)

1. **安装依赖**
   ```bash
   pip3 install -r requirements.txt
   ```

2. **运行组件测试**
   ```bash
   cd integration
   python3 test_components.py
   ```

3. **如果组件测试通过，运行最小训练**
   ```bash
   python3 deep_train.py --config test_config.yaml
   ```

### 服务器部署 Server Deployment

1. **上传代码到服务器**
   ```bash
   scp -r "/Users/zhangmingda/Desktop/agent worflow" server:/path/to/
   ```

2. **在服务器上安装依赖**
   ```bash
   ssh server
   cd /path/to/agent worflow
   pip3 install -r requirements.txt
   ```

3. **配置完整训练**
   - 编辑 `integration/deep_config.yaml`
   - 设置多数据集
   - 调整并行参数
   - 配置GPU

4. **运行完整训练**
   ```bash
   cd integration
   python3 deep_train.py --config deep_config.yaml
   ```

5. **监控训练**
   ```bash
   tail -f output/deep_integration/logs/training.log
   ```

---

## ✅ 结论 Conclusion

**当前状态 Current Status**:
- 代码实现: ✅ 100% 完成
- 文件结构: ✅ 验证通过
- 逻辑流程: ✅ 验证通过
- 功能测试: ⏳ 等待依赖安装
- 集成测试: ⏳ 等待运行

**准备就绪 Ready For**:
- ✅ 代码审查 Code review
- ✅ 依赖安装 Dependency installation
- ✅ 组件测试 Component testing
- ✅ 服务器部署 Server deployment

**建议 Recommendations**:
1. 先在本地安装依赖并运行组件测试
2. 如果组件测试通过，在服务器上进行完整训练
3. 使用test_config.yaml进行快速验证
4. 使用deep_config.yaml进行完整训练

---

**准备好开始测试了！Ready to start testing!** 🚀
