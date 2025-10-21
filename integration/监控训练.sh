#!/bin/bash
# 训练监控脚本 - 实时显示训练进度

LOG_FILE="/content/drive/MyDrive/agentworkflow/outputs/optimized_training.log"
STATS_DIR="/content/drive/MyDrive/agentworkflow/outputs/optimized_training"

echo "=========================================="
echo "  AIME训练监控"
echo "=========================================="
echo ""

# 检查训练是否在运行
TRAINING_PID=$(ps aux | grep "优化训练.py" | grep -v grep | awk '{print $2}')

if [ -z "$TRAINING_PID" ]; then
    echo "⚠️  未检测到正在运行的训练进程"
else
    echo "✅ 训练进程运行中 (PID: $TRAINING_PID)"
fi

echo ""
echo "选择监控选项:"
echo "  1) 实时查看日志"
echo "  2) 显示训练统计"
echo "  3) 显示最近的奖励"
echo "  4) 检查检查点"
echo "  5) 显示GPU使用情况"
echo "  6) 查看最近50行日志"
echo "  7) 停止训练"
echo ""
read -p "请选择 (1-7): " -n 1 -r
echo ""
echo ""

case $REPLY in
    1)
        echo "📊 实时查看日志 (Ctrl+C退出)..."
        if [ -f "$LOG_FILE" ]; then
            tail -f "$LOG_FILE"
        else
            echo "❌ 日志文件不存在: $LOG_FILE"
        fi
        ;;
    2)
        echo "📈 训练统计:"
        echo ""
        if [ -f "$LOG_FILE" ]; then
            echo "总Epoch数:"
            grep -oP "Epoch \K[0-9]+/[0-9]+" "$LOG_FILE" | tail -1
            echo ""
            echo "最佳分数:"
            grep -oP "新最佳分数: \K[0-9.]+" "$LOG_FILE" | tail -1
            echo ""
            echo "平均分数趋势:"
            grep -oP "平均分数: \K[0-9.]+" "$LOG_FILE" | tail -5
        else
            echo "❌ 日志文件不存在"
        fi
        ;;
    3)
        echo "💰 最近的奖励:"
        echo ""
        if [ -f "$LOG_FILE" ]; then
            grep "avg_reward" "$LOG_FILE" | tail -10
        else
            echo "❌ 日志文件不存在"
        fi
        ;;
    4)
        echo "💾 检查点列表:"
        echo ""
        if [ -d "$STATS_DIR/checkpoints" ]; then
            ls -lht "$STATS_DIR/checkpoints" | head -20
        else
            echo "❌ 检查点目录不存在"
        fi
        ;;
    5)
        echo "🖥️  GPU使用情况:"
        echo ""
        nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
        ;;
    6)
        echo "📝 最近50行日志:"
        echo ""
        if [ -f "$LOG_FILE" ]; then
            tail -50 "$LOG_FILE"
        else
            echo "❌ 日志文件不存在"
        fi
        ;;
    7)
        if [ -z "$TRAINING_PID" ]; then
            echo "❌ 没有运行中的训练进程"
        else
            read -p "确定要停止训练 (PID: $TRAINING_PID)? (y/n): " -n 1 -r
            echo ""
            if [[ $REPLY =~ ^[Yy]$ ]]; then
                kill -15 $TRAINING_PID
                echo "✅ 已发送停止信号"
                sleep 2
                if ps -p $TRAINING_PID > /dev/null; then
                    echo "⚠️  进程仍在运行，强制停止..."
                    kill -9 $TRAINING_PID
                    echo "✅ 训练已强制停止"
                else
                    echo "✅ 训练已正常停止"
                fi
            fi
        fi
        ;;
    *)
        echo "❌ 无效选项"
        ;;
esac
