#!/bin/bash
# Python依赖检查和安装脚本

echo "=========================================="
echo "  检查和安装训练所需的Python依赖"
echo "=========================================="
echo ""

echo "【1】检查Python版本..."
PYTHON_VERSION=$(python3 --version)
echo "✓ $PYTHON_VERSION"

echo ""
echo "【2】检查CUDA..."
if command -v nvcc &> /dev/null; then
    CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $6}' | cut -c2-)
    echo "✓ CUDA $CUDA_VERSION"
else
    echo "⚠️  CUDA不可用（CPU模式）"
fi

echo ""
echo "【3】检查核心依赖包..."

# 检查PyTorch
if python3 -c "import torch; print(f'✓ PyTorch {torch.__version__}')" 2>/dev/null; then
    :
else
    echo "⚠️  PyTorch未安装，开始安装..."
    pip install -q torch torchvision torchaudio
fi

# 检查transformers
if python3 -c "import transformers; print(f'✓ transformers {transformers.__version__}')" 2>/dev/null; then
    :
else
    echo "⚠️  transformers未安装，开始安装..."
    pip install -q transformers
fi

# 检查peft (LoRA)
if python3 -c "import peft; print(f'✓ peft {peft.__version__}')" 2>/dev/null; then
    :
else
    echo "⚠️  peft未安装，开始安装..."
    pip install -q peft
fi

# 检查accelerate
if python3 -c "import accelerate; print(f'✓ accelerate {accelerate.__version__}')" 2>/dev/null; then
    :
else
    echo "⚠️  accelerate未安装，开始安装..."
    pip install -q accelerate
fi

echo ""
echo "【4】检查其他依赖..."

# OpenAI
if python3 -c "import openai; print(f'✓ openai {openai.__version__}')" 2>/dev/null; then
    :
else
    echo "⚠️  openai未安装，开始安装..."
    pip install -q openai
fi

# YAML
if python3 -c "import yaml; print('✓ PyYAML')" 2>/dev/null; then
    :
else
    echo "⚠️  PyYAML未安装，开始安装..."
    pip install -q pyyaml
fi

# huggingface_hub (for model download)
if python3 -c "import huggingface_hub; print(f'✓ huggingface_hub {huggingface_hub.__version__}')" 2>/dev/null; then
    :
else
    echo "⚠️  huggingface_hub未安装，开始安装..."
    pip install -q huggingface_hub
fi

# aiohttp (for async)
if python3 -c "import aiohttp; print(f'✓ aiohttp {aiohttp.__version__}')" 2>/dev/null; then
    :
else
    echo "⚠️  aiohttp未安装，开始安装..."
    pip install -q aiohttp
fi

echo ""
echo "【5】验证核心模块导入..."

python3 << 'PYEOF'
import sys
sys.path.insert(0, '/content/agentflow/integration')
sys.path.insert(0, '/content/agentflow/AFlow')

try:
    import torch
    print(f"✓ PyTorch {torch.__version__}")
    print(f"  - CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  - Device: {torch.cuda.get_device_name(0)}")

    import transformers
    print(f"✓ transformers {transformers.__version__}")

    import peft
    print(f"✓ peft {peft.__version__}")

    from scripts.logs import logger
    print("✓ AFlow scripts.logs")

    from workflow_parser import WorkflowParser
    print("✓ workflow_parser")

    print("\n✅ 所有模块验证通过！")

except Exception as e:
    print(f"\n❌ 模块导入错误: {e}")
    sys.exit(1)
PYEOF

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✅ 所有依赖检查和安装完成！"
    echo "=========================================="
    echo ""
    echo "下一步："
    echo "  1. export OPENAI_API_KEY='your-api-key'"
    echo "  2. cd /content/agentflow/integration"
    echo "  3. ./启动优化训练.sh"
else
    echo ""
    echo "❌ 依赖检查失败，请检查错误信息"
    exit 1
fi
