#!/bin/bash
# 日志同步脚本 - 独立运行，不依赖Claude连接

LOG_SOURCE_DIR="/content/agentflow/integration/logs/"
LOG_DEST_DIR="/content/drive/MyDrive/agentflow/outputs/minimal_test/logs/"

# 创建目标目录（如果不存在）
mkdir -p "$LOG_DEST_DIR"

echo "开始日志同步守护进程..."
echo "源目录: $LOG_SOURCE_DIR"
echo "目标目录: $LOG_DEST_DIR"
echo "同步间隔: 每5分钟"
echo "日志文件: /content/agentflow/integration/sync_logs_daemon.log"

# 无限循环，每5分钟同步一次
while true; do
    TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')

    # 检查源目录是否存在
    if [ -d "$LOG_SOURCE_DIR" ]; then
        # 复制整个logs目录下的所有文件
        cp -r "$LOG_SOURCE_DIR"* "$LOG_DEST_DIR"

        # 统计文件数和总大小
        FILE_COUNT=$(find "$LOG_SOURCE_DIR" -type f | wc -l)
        TOTAL_SIZE=$(du -sh "$LOG_SOURCE_DIR" | cut -f1)

        # 记录同步状态
        echo "[$TIMESTAMP] ✓ 已同步 $FILE_COUNT 个文件 (总大小: $TOTAL_SIZE)"
    else
        echo "[$TIMESTAMP] ✗ 源目录不存在"
    fi

    # 等待5分钟（300秒）
    sleep 300
done
