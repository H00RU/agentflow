# Server Deployment Guide
# 服务器部署指南

**Status**: ✅ **Mac M4 local tests passed - Ready for server**
**Date**: 2025-10-09

---

## 🚀 Quick Server Deployment

### Step 1: Upload Code to Server

```bash
# From your Mac
cd "/Users/zhangmingda/Desktop"

# Upload to server
scp -r "agent worflow" username@server:/path/to/destination/

# Or use rsync for better performance
rsync -avz --progress "agent worflow" username@server:/path/to/destination/
```

### Step 2: SSH to Server

```bash
ssh username@server
cd /path/to/destination/agent worflow
```

### Step 3: Setup Environment

```bash
# Check Python version (need 3.8+)
python3 --version

# Install dependencies
pip3 install -r requirements.txt

# Or install individually
pip3 install numpy torch pyyaml ray anthropic

# Verify installation
python3 -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
```

### Step 4: Configure for Server

```bash
cd integration

# Edit configuration for server (if needed)
vim deep_config.yaml
```

**Key configurations to adjust**:

```yaml
# Change device to use GPU
device: "cuda"  # or "cuda:0" for specific GPU

# Adjust parallel environments based on server resources
environment:
  env_num: 8  # Increase for more parallelization
  group_n: 2  # Keep for GiGPO

# Add more datasets
environment:
  train_datasets:
    - "HumanEval"
    - "MBPP"
    - "GSM8K"
    - "MATH"

# Adjust training parameters
total_epochs: 20
episodes_per_epoch: 50
```

### Step 5: Run Tests on Server

```bash
# Test 1: Verify files
python3 integration/verify_files.py

# Test 2: Test components
python3 integration/test_components.py

# Test 3: Run integration test
python3 integration/test_integration_simple.py
```

### Step 6: Start Training

```bash
cd integration

# Option 1: Run in foreground (for testing)
python3 deep_train.py --config deep_config.yaml

# Option 2: Run in background with logging
nohup python3 deep_train.py --config deep_config.yaml > training.log 2>&1 &

# Option 3: Use screen (recommended)
screen -S aflow_training
python3 deep_train.py --config deep_config.yaml
# Press Ctrl+A then D to detach
# Use 'screen -r aflow_training' to reattach
```

### Step 7: Monitor Training

```bash
# Watch log file
tail -f output/deep_integration/logs/training.log

# Check statistics
cat output/deep_integration/logs/training_stats.json | jq '.'

# Monitor GPU usage (if using CUDA)
watch -n 1 nvidia-smi
```

---

## 📋 Server-Specific Configurations

### For Multi-GPU Servers

```yaml
# In deep_config.yaml
device: "cuda"  # PyTorch will use available GPUs

hardware:
  use_distributed: true
  world_size: 4  # Number of GPUs
```

### For CPU-Only Servers

```yaml
device: "cpu"

environment:
  env_num: 16  # More parallel environments to compensate
  resources_per_worker:
    num_cpus: 2.0
```

### For High-Memory Servers

```yaml
# Increase experience pool and states
experience_pool_size: 50000

advanced:
  state_tracking:
    max_states: 500000

# More parallel environments
environment:
  env_num: 16
  group_n: 4
```

---

## 🔧 Troubleshooting on Server

### Issue 1: CUDA Not Available

```bash
# Check CUDA
python3 -c "import torch; print(torch.cuda.is_available())"

# If False, check:
nvidia-smi  # Check GPU driver
nvcc --version  # Check CUDA toolkit

# Solution: Install PyTorch with CUDA support
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Issue 2: Ray Initialization Fails

```bash
# Check Ray
python3 -c "import ray; ray.init()"

# If error, try:
ray stop  # Stop any existing Ray processes
ray start --head  # Start Ray manually

# Or in config, reduce resources:
environment:
  env_num: 2  # Reduce if Ray has issues
```

### Issue 3: Out of Memory

```yaml
# Reduce batch size and parallel environments
rl:
  batch_size: 16  # From 64

environment:
  env_num: 2  # From 4
  max_rounds: 10  # From 20

# Or use CPU offloading
hardware:
  use_amp: true  # Mixed precision
```

---

## 📊 Expected Performance on Server

### Training Time Estimates

**Small Test** (1 epoch, 2 episodes, HumanEval):
- CPU: 10-15 minutes
- GPU: 5-8 minutes

**Medium Run** (5 epochs, 20 episodes, HumanEval + GSM8K):
- CPU: 2-3 hours
- GPU: 1-1.5 hours

**Full Training** (20 epochs, 50 episodes, all datasets):
- CPU: 1-2 days
- GPU: 6-12 hours

### Resource Requirements

**Minimum**:
- CPU: 4 cores
- RAM: 8GB
- Disk: 10GB

**Recommended**:
- CPU: 16+ cores
- RAM: 32GB+
- GPU: 1x NVIDIA GPU (8GB+ VRAM)
- Disk: 50GB+

**Optimal**:
- CPU: 32+ cores
- RAM: 64GB+
- GPU: 4x NVIDIA GPU (16GB+ VRAM each)
- Disk: 100GB+ SSD

---

## 🎯 Optimization Tips for Server

### 1. Parallel Environments

```yaml
# Scale based on CPU/GPU count
environment:
  env_num: <num_cpus / 2>  # e.g., 32 cores → 16 envs
  group_n: 2
```

### 2. Batch Processing

```yaml
# Increase batch size if GPU memory allows
rl:
  batch_size: 128  # From 64
  gradient_accumulation_steps: 2
```

### 3. Dataset Parallelization

Train multiple datasets simultaneously:

```bash
# Terminal 1: HumanEval
python3 deep_train.py --config config_humaneval.yaml

# Terminal 2: GSM8K
python3 deep_train.py --config config_gsm8k.yaml

# Terminal 3: MATH
python3 deep_train.py --config config_math.yaml
```

### 4. Checkpointing

```yaml
checkpoint:
  save_frequency: 5  # Save every 5 epochs
  save_best: true
  max_checkpoints: 10
```

---

## 📈 Monitoring and Logging

### Real-time Monitoring

```bash
# Use tmux for multiple panes
tmux new -s aflow

# Pane 1: Training
python3 deep_train.py --config deep_config.yaml

# Pane 2: Log monitoring (Ctrl+B then ")
tail -f output/deep_integration/logs/training.log

# Pane 3: GPU monitoring (Ctrl+B then ")
watch -n 1 nvidia-smi

# Pane 4: System monitoring
htop
```

### Log Analysis

```bash
# Extract scores
grep "avg_score" output/deep_integration/logs/training.log

# Plot progress (if matplotlib available)
python3 << EOF
import json
import matplotlib.pyplot as plt

with open('output/deep_integration/logs/training_stats.json') as f:
    stats = json.load(f)

plt.plot(stats['avg_scores'])
plt.xlabel('Epoch')
plt.ylabel('Score')
plt.title('Training Progress')
plt.savefig('training_progress.png')
print('Saved to training_progress.png')
EOF
```

---

## 🔄 Resume Training

If training is interrupted:

```bash
cd integration

# Training will automatically resume from latest checkpoint if:
python3 deep_train.py --config deep_config.yaml

# Or explicitly specify checkpoint:
python3 deep_train.py \
  --config deep_config.yaml \
  --resume_from output/deep_integration/checkpoints/epoch_10.pt
```

---

## ✅ Pre-Deployment Checklist

### Before Starting Full Training

- [ ] Code uploaded to server
- [ ] Dependencies installed (`pip3 list`)
- [ ] Python 3.8+ available
- [ ] GPU drivers working (if using GPU)
- [ ] Ray installed and tested
- [ ] Anthropic API key configured
- [ ] Sufficient disk space (50GB+)
- [ ] Sufficient RAM (32GB+ recommended)
- [ ] test_components.py passed
- [ ] test_integration_simple.py passed
- [ ] Configuration file adjusted for server
- [ ] Monitoring tools ready (screen/tmux)

---

## 🚀 Quick Start Commands for Server

```bash
# Complete deployment in one go
cd /path/to/agent worflow

# 1. Install
pip3 install -r requirements.txt

# 2. Test
python3 integration/test_components.py

# 3. Configure (edit as needed)
vim integration/deep_config.yaml

# 4. Train (in background)
cd integration
nohup python3 deep_train.py --config deep_config.yaml > ../training.log 2>&1 &

# 5. Monitor
tail -f ../training.log
```

---

## 📞 Support

If you encounter issues:

1. Check `TEST_RESULTS.md` - Mac M4 local tests
2. Check `TESTING_GUIDE.md` - Troubleshooting section
3. Check `QUICK_START.md` - Common issues FAQ
4. Check logs: `output/deep_integration/logs/training.log`
5. Check configuration: `integration/deep_config.yaml`

---

**Your Code Status**: ✅ **READY FOR DEPLOYMENT**

**Mac M4 Tests**: ✅ **ALL PASSED**

**Next Action**: Upload to server and start training! 🚀

---

**Deployment Date**: 2025-10-09
**Tested On**: Mac mini M4
**Ready For**: Production server deployment
