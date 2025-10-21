#!/bin/bash
# Installation script for deep integration dependencies
# 深度集成依赖安装脚本

echo "================================================"
echo "Installing Dependencies for Deep Integration"
echo "================================================"

# Check Python version
echo ""
echo "[1/6] Checking Python version..."
python3 --version

# Install basic dependencies
echo ""
echo "[2/6] Installing basic dependencies..."
pip3 install numpy

echo ""
echo "[3/6] Installing PyTorch..."
pip3 install torch

echo ""
echo "[4/6] Installing YAML..."
pip3 install pyyaml

echo ""
echo "[5/6] Installing Ray..."
pip3 install ray

echo ""
echo "[6/6] Installing optional dependencies..."
pip3 install anthropic  # For Claude API

echo ""
echo "================================================"
echo "Installation Complete!"
echo "================================================"
echo ""
echo "To verify installation, run:"
echo "  cd integration"
echo "  python3 test_components.py"
