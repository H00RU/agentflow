#!/bin/bash
# 优化训练启动脚本 - 包含模型检查和输出目录设置

echo "=========================================="
echo "  AIME优化训练系统启动"
echo "=========================================="

# 切换到integration目录
cd /content/agentflow/integration

echo ""
echo "【1/5】检查配置文件..."
# 检查配置文件
if [ ! -f "优化运行.yaml" ]; then
    echo "❌ 错误: 找不到配置文件 优化运行.yaml"
    exit 1
fi

if [ ! -f "优化训练.py" ]; then
    echo "❌ 错误: 找不到训练脚本 优化训练.py"
    exit 1
fi
echo "✓ 配置文件检查通过"

echo ""
echo "【2/5】检查环境变量..."
# 检查API Key
if [ -z "$OPENAI_API_KEY" ]; then
    echo "❌ 错误: 未设置 OPENAI_API_KEY"
    echo ""
    echo "请先设置API Key:"
    echo "  export OPENAI_API_KEY='your-api-key-here'"
    exit 1
fi
echo "✓ OPENAI_API_KEY 已设置"

echo ""
echo "【3/5】检查Qwen模型..."
MODEL_PATH="/root/models/Qwen2.5-7B-Instruct"

if [ ! -d "$MODEL_PATH" ]; then
    echo "⚠️  模型不存在: $MODEL_PATH"
    echo ""
    read -p "是否自动下载 Qwen2.5-7B-Instruct 模型? (y/n): " -n 1 -r
    echo ""

    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "🔄 开始下载模型（约14GB，需要一些时间）..."
        mkdir -p /root/models

        # 使用 huggingface-cli 下载
        pip install -q huggingface_hub

        python3 -c "
from huggingface_hub import snapshot_download
import os

print('正在从 Hugging Face 下载 Qwen2.5-7B-Instruct...')
model_path = snapshot_download(
    repo_id='Qwen/Qwen2.5-7B-Instruct',
    local_dir='/root/models/Qwen2.5-7B-Instruct',
    local_dir_use_symlinks=False
)
print(f'✓ 模型下载完成: {model_path}')
"

        if [ $? -eq 0 ]; then
            echo "✓ 模型下载成功"
        else
            echo "❌ 模型下载失败"
            echo ""
            echo "手动下载方法："
            echo "  1. 访问: https://huggingface.co/Qwen/Qwen2.5-7B-Instruct"
            echo "  2. 下载所有文件到: /root/models/Qwen2.5-7B-Instruct/"
            exit 1
        fi
    else
        echo "❌ 训练需要模型文件，请手动下载后再启动"
        exit 1
    fi
else
    echo "✓ 模型已存在: $MODEL_PATH"
fi

echo ""
echo "【4/5】设置输出目录..."
OUTPUT_DIR="/content/drive/MyDrive/agentflow/outputs/optimized_training"

# 检查Google Drive是否挂载
if [ ! -d "/content/drive" ]; then
    echo "⚠️  Google Drive 未挂载"
    echo ""
    read -p "是否挂载 Google Drive? (y/n): " -n 1 -r
    echo ""

    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "🔄 正在挂载 Google Drive..."
        python3 -c "from google.colab import drive; drive.mount('/content/drive')"

        if [ $? -ne 0 ]; then
            echo "❌ Google Drive 挂载失败"
            echo "改用本地目录: /content/outputs/optimized_training"
            OUTPUT_DIR="/content/outputs/optimized_training"
        fi
    else
        echo "使用本地输出目录: /content/outputs/optimized_training"
        OUTPUT_DIR="/content/outputs/optimized_training"
    fi
fi

# 创建输出目录
mkdir -p "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR/checkpoints"
mkdir -p "$OUTPUT_DIR/workflows"

echo "✓ 输出目录: $OUTPUT_DIR"

# 如果输出目录改变了，更新配置文件中的路径
if [ "$OUTPUT_DIR" != "/content/drive/MyDrive/agentflow/outputs/optimized_training" ]; then
    echo "🔄 更新配置文件中的输出路径..."
    sed -i "s|/content/drive/MyDrive/agentflow/outputs/optimized_training|$OUTPUT_DIR|g" 优化运行.yaml
fi

echo ""
echo "【5/5】准备启动训练..."
echo ""
echo "=========================================="
echo "训练配置："
echo "  - 模型: Qwen2.5-7B-Instruct"
echo "  - 总Epoch数: 20"
echo "  - 课程学习: 4题→8题→12题"
echo "  - 奖励塑形: 启用"
echo "  - 学习率: 0.0003"
echo "  - 输出目录: $OUTPUT_DIR"
echo "=========================================="
echo ""

# 询问是否后台运行
read -p "是否在后台运行? (y/n): " -n 1 -r
echo ""
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    # 后台运行
    OUTPUT_LOG="$OUTPUT_DIR/training.log"
    echo "📝 日志将保存到: $OUTPUT_LOG"
    echo "🚀 开始后台训练..."

    nohup python3 优化训练.py --config 优化运行.yaml > "$OUTPUT_LOG" 2>&1 &
    PID=$!

    echo ""
    echo "✅ 训练已在后台启动 (PID: $PID)"
    echo ""
    echo "监控命令:"
    echo "  查看日志: tail -f $OUTPUT_LOG"
    echo "  实时监控: watch -n 5 'tail -20 $OUTPUT_LOG'"
    echo "  检查进程: ps aux | grep $PID"
    echo "  停止训练: kill $PID"
    echo ""
    echo "或使用监控脚本:"
    echo "  ./监控训练.sh"
else
    # 前台运行
    echo "🚀 开始训练 (前台运行，Ctrl+C可停止)..."
    echo ""
    python3 优化训练.py --config 优化运行.yaml
fi
