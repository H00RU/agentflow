#!/bin/bash

echo "==================================="
echo "  Qwen生成的Workflow代码示例"
echo "==================================="
echo ""

for dir in output/workflows_generated/*/; do
    if [ -f "$dir/graph.py" ]; then
        echo "📁 $(basename $dir)"
        echo "-----------------------------------"
        echo ""
        echo "📝 修改说明:"
        if [ -f "$dir/modification.txt" ]; then
            cat "$dir/modification.txt" | head -10
        fi
        echo ""
        echo "💻 生成的代码:"
        cat "$dir/graph.py" | head -50
        echo ""
        echo "==================================="
        echo ""
    fi
done
