#!/bin/bash
# 一键打包脚本 - 准备上传到新服务器
# 在本地 Mac 上运行此脚本

set -e

echo "========================================================================"
echo "AFlow + verl-agent 项目打包脚本"
echo "========================================================================"
echo ""

# 当前目录
CURRENT_DIR="/Users/zhangmingda/Desktop/agent worflow"
OUTPUT_FILE="aflow_verl_integration_fixed.tar.gz"

cd "$CURRENT_DIR"

# 检查目录是否存在
echo "[1/4] 检查目录..."
if [ ! -d "integration" ]; then
    echo "  ❌ 未找到 integration 目录"
    exit 1
fi

if [ ! -d "AFlow" ]; then
    echo "  ❌ 未找到 AFlow 目录"
    exit 1
fi

if [ ! -d "verl-agent" ]; then
    echo "  ❌ 未找到 verl-agent 目录"
    exit 1
fi

echo "  ✓ 所有目录存在"

# 统计文件
echo ""
echo "[2/4] 统计文件..."
INTEGRATION_FILES=$(find integration -type f | wc -l | tr -d ' ')
AFLOW_FILES=$(find AFlow -type f | wc -l | tr -d ' ')
VERL_FILES=$(find verl-agent -type f | wc -l | tr -d ' ')
TOTAL_FILES=$((INTEGRATION_FILES + AFLOW_FILES + VERL_FILES))

echo "  integration: $INTEGRATION_FILES 个文件"
echo "  AFlow: $AFLOW_FILES 个文件"
echo "  verl-agent: $VERL_FILES 个文件"
echo "  总计: $TOTAL_FILES 个文件"

# 删除旧的压缩包
if [ -f "$OUTPUT_FILE" ]; then
    echo ""
    echo "[3/4] 删除旧的压缩包..."
    rm -f "$OUTPUT_FILE"
    echo "  ✓ 已删除旧压缩包"
fi

# 创建压缩包
echo ""
echo "[4/4] 创建压缩包..."
echo "  这可能需要几分钟，请稍候..."

tar czf "$OUTPUT_FILE" \
    --exclude='*.pyc' \
    --exclude='__pycache__' \
    --exclude='.git' \
    --exclude='.DS_Store' \
    --exclude='*.log' \
    --exclude='output' \
    --exclude='data/aflow_train.parquet' \
    --exclude='data/aflow_val.parquet' \
    integration/ \
    AFlow/ \
    verl-agent/

# 显示结果
echo ""
echo "========================================================================"
echo "✅ 打包完成！"
echo "========================================================================"
echo ""

# 文件信息
FILE_SIZE=$(ls -lh "$OUTPUT_FILE" | awk '{print $5}')
FILE_PATH="$CURRENT_DIR/$OUTPUT_FILE"

echo "压缩包信息："
echo "  文件名: $OUTPUT_FILE"
echo "  大小: $FILE_SIZE"
echo "  路径: $FILE_PATH"
echo ""

# 验证压缩包
echo "验证压缩包内容..."
tar tzf "$OUTPUT_FILE" | head -20
echo "  ... (显示前 20 个文件)"
echo ""

ARCHIVE_FILES=$(tar tzf "$OUTPUT_FILE" | wc -l | tr -d ' ')
echo "  压缩包包含 $ARCHIVE_FILES 个文件"

# 下一步提示
echo ""
echo "========================================================================"
echo "下一步操作："
echo "========================================================================"
echo ""
echo "1. 上传压缩包到新服务器："
echo "   scp $OUTPUT_FILE root@YOUR_SERVER:/root/"
echo ""
echo "2. 登录新服务器并解压："
echo "   ssh root@YOUR_SERVER"
echo "   cd /root"
echo "   tar xzf $OUTPUT_FILE"
echo "   mkdir -p aflow_integration"
echo "   mv integration AFlow verl-agent aflow_integration/"
echo ""
echo "3. 运行环境设置脚本："
echo "   cd /root/aflow_integration/integration"
echo "   chmod +x setup_new_server.sh"
echo "   bash setup_new_server.sh"
echo ""
echo "4. 查看完整部署文档："
echo "   cat /root/aflow_integration/integration/DEPLOYMENT_GUIDE.md"
echo ""
echo "========================================================================"
