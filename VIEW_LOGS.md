# 🔍 训练日志查看指南

## 快速访问命令

### SSH连接
```bash
ssh root@6.tcp.ngrok.io -p 15577
# 密码: LtgyRHLSCrFm
```

---

## 📊 日志文件位置

| 文件 | 完整路径 | 说明 |
|------|----------|------|
| **主训练日志** | `/root/aflow_integration/integration/rl_training_final.log` | 当前运行的完整日志 |
| **历史日志** | `/root/aflow_integration/integration/rl_training.log` | 之前的训练日志 |
| **配置文件** | `/root/aflow_integration/integration/test_config.yaml` | 训练配置 |
| **输出目录** | `/root/aflow_integration/integration/output/test_run/` | 所有输出结果 |

---

## 🔧 常用查看命令

### 1. 实时监控（推荐）
```bash
tail -f /root/aflow_integration/integration/rl_training_final.log
```
> 按 Ctrl+C 退出

### 2. 查看最近100行
```bash
tail -100 /root/aflow_integration/integration/rl_training_final.log
```

### 3. 查看全部日志
```bash
cat /root/aflow_integration/integration/rl_training_final.log
```

### 4. 查看训练分数
```bash
grep "Average score" /root/aflow_integration/integration/rl_training_final.log | tail -20
```

### 5. 查看epoch进度
```bash
grep -E "epoch|Episode.*completed" /root/aflow_integration/integration/rl_training_final.log
```

### 6. 查看错误
```bash
grep -i "error\|exception\|failed" /root/aflow_integration/integration/rl_training_final.log | tail -20
```

### 7. 查看Qwen加载信息
```bash
grep -E "QwenPolicy|RL policy|Worker.*Loading" /root/aflow_integration/integration/rl_training_final.log | head -20
```

### 8. 查看Claude API调用
```bash
grep -E "Using Anthropic|Claude|Token usage" /root/aflow_integration/integration/rl_training_final.log | tail -20
```

---

## 💻 进程管理

### 检查训练是否运行
```bash
ps aux | grep deep_train.py | grep -v grep
```

### 查看进程详细信息
```bash
ps aux | grep python3 | grep deep_train
```

### 停止训练（如需要）
```bash
pkill -f deep_train.py
```

---

## 🖥️ GPU监控

### 查看GPU状态
```bash
nvidia-smi
```

### 持续监控GPU（每秒更新）
```bash
watch -n 1 nvidia-smi
```
> 按 Ctrl+C 退出

### 简洁的GPU内存查看
```bash
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
```

---

## 📈 训练输出文件

### 查看输出目录结构
```bash
ls -la /root/aflow_integration/integration/output/test_run/
```

### 输出目录包含：
```
output/test_run/
├── checkpoints/              # 训练检查点
├── logs/                     # 结构化日志
│   ├── training_stats.json  # 训练统计
│   └── eval_epoch_*.json    # 评估结果
├── optimized_workflows/     # 优化后的工作流
│   ├── train/
│   │   └── HumanEval/
│   │       └── worker_0/
│   │           └── workflows/
│   └── test/
└── [其他输出文件]
```

### 查看训练统计
```bash
cat /root/aflow_integration/integration/output/test_run/logs/training_stats.json | python3 -m json.tool
```

### 查看工作流结果
```bash
ls -la /root/aflow_integration/integration/output/test_run/optimized_workflows/train/HumanEval/worker_0/HumanEval/workflows/
```

---

## 🔎 高级日志分析

### 统计日志行数
```bash
wc -l /root/aflow_integration/integration/rl_training_final.log
```

### 查看特定时间的日志
```bash
grep "2025-10-09 17:2" /root/aflow_integration/integration/rl_training_final.log | tail -50
```

### 统计分数分布
```bash
grep "Average score" /root/aflow_integration/integration/rl_training_final.log | awk '{print $NF}' | sort | uniq -c
```

### 查看评估进度
```bash
grep "Evaluating HumanEval problems" /root/aflow_integration/integration/rl_training_final.log | tail -10
```

---

## 📱 一键查看脚本

创建快捷脚本：
```bash
# 在服务器上创建查看脚本
cat > ~/view_training.sh << 'SCRIPT'
#!/bin/bash
echo "=========================================="
echo "训练进程状态"
echo "=========================================="
ps aux | grep deep_train.py | grep -v grep

echo ""
echo "=========================================="
echo "GPU使用情况"
echo "=========================================="
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv

echo ""
echo "=========================================="
echo "最近10个训练分数"
echo "=========================================="
grep "Average score" /root/aflow_integration/integration/rl_training_final.log | tail -10

echo ""
echo "=========================================="
echo "最后20行日志"
echo "=========================================="
tail -20 /root/aflow_integration/integration/rl_training_final.log
SCRIPT

chmod +x ~/view_training.sh
```

**使用**:
```bash
ssh root@6.tcp.ngrok.io -p 15577
~/view_training.sh
```

---

## 🆘 故障排查

### 训练卡住了？
```bash
# 查看最后修改时间
ls -lh /root/aflow_integration/integration/rl_training_final.log

# 查看最后100行是否有重复
tail -100 /root/aflow_integration/integration/rl_training_final.log | tail -20
```

### 查看Ray进程
```bash
ps aux | grep ray | grep -v grep
```

### 查看完整进程树
```bash
pstree -p $(pgrep -f deep_train.py)
```

---

## 📞 远程连接信息

- **SSH地址**: `root@6.tcp.ngrok.io`
- **端口**: `15577`
- **密码**: `LtgyRHLSCrFm`
- **会话时长**: 24小时

---

## 💡 提示

1. **实时查看最方便**: `tail -f` 命令可以实时看到新产生的日志
2. **保持连接**: SSH可能会超时断开，需要重新连接
3. **日志很大**: 当前已有23,000+行，可以使用 `less` 命令分页查看
4. **颜色代码**: 日志中的 `[32m` 等是颜色代码，可以忽略

