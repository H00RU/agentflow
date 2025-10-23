#!/usr/bin/env python3
"""
训练监控脚本 - 实时追踪Qwen的学习进度和operator选择
"""

import os
import time
import json
from pathlib import Path
from collections import defaultdict

class TrainingMonitor:
    """监控训练进度,特别关注operator选择的演化"""

    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        self.workflow_dir = self.output_dir / "workflows" / "AIME"
        self.operator_history = []
        self.score_history = []

    def scan_workflows(self):
        """扫描所有生成的workflow"""
        if not self.workflow_dir.exists():
            return []

        workflows = []
        for round_dir in sorted(self.workflow_dir.iterdir()):
            if not round_dir.is_dir() or not round_dir.name.startswith("round_"):
                continue

            # 读取modification.txt
            mod_file = round_dir / "modification.txt"
            if mod_file.exists():
                with open(mod_file, 'r') as f:
                    content = f.read()

                # 提取operators
                operators = []
                if "Operators:" in content:
                    ops_line = content.split("Operators:")[1].split("\n")[1]
                    operators = [op.strip() for op in ops_line.split(",")]

                workflows.append({
                    'round': round_dir.name,
                    'operators': operators,
                    'path': str(round_dir)
                })

        return workflows

    def print_operator_evolution(self, workflows):
        """打印operator选择的演化"""
        print("\n" + "="*70)
        print("🔍 OPERATOR选择演化分析")
        print("="*70)

        if not workflows:
            print("暂无workflow数据")
            return

        # 统计operator使用频率
        operator_counts = defaultdict(int)
        operator_by_round = {}

        for wf in workflows:
            round_name = wf['round']
            operators = wf['operators']
            operator_by_round[round_name] = operators

            for op in operators:
                operator_counts[op] += 1

        # 显示每一轮的选择
        print("\n📊 每轮Operator选择:")
        for round_name in sorted(operator_by_round.keys()):
            operators = operator_by_round[round_name]
            print(f"  {round_name:15s}: {', '.join(operators)}")

        # 分析是否有错误的operator
        print("\n⚠️  Operator适配性分析 (AIME数学任务):")
        for op, count in sorted(operator_counts.items(), key=lambda x: -x[1]):
            if op == "CustomCodeGenerate":
                status = "❌ 错误! (代码生成operator不适合数学题)"
            elif op == "Custom":
                status = "✅ 正确 (通用operator适合数学推理)"
            elif op == "ScEnsemble":
                status = "✅ 正确 (提高答案质量)"
            elif op in ["Review", "Revise"]:
                status = "✅ 有用 (改进答案)"
            elif op == "Test":
                status = "⚠️  可选 (测试功能有限)"
            else:
                status = "❓ 未知"

            print(f"  {op:20s}: 使用{count}次  {status}")

        # 检查泛用性问题
        print("\n🎯 泛用性评估:")
        if "CustomCodeGenerate" in operator_counts:
            print("  ⚠️  发现问题: Qwen仍在为数学任务选择代码生成operator")
            print("     建议: 需要更多训练让Qwen学会区分任务类型")
        else:
            print("  ✅ 良好: Qwen没有选择不适配的CustomCodeGenerate")

        if "Custom" in operator_counts:
            print("  ✅ 良好: Qwen正确选择了Custom operator用于数学推理")
        else:
            print("  ⚠️  注意: Qwen没有使用Custom,可能缺少推理能力")

        return operator_counts

    def monitor_loop(self, interval=30):
        """持续监控循环"""
        print("🚀 开始监控训练进度...")
        print(f"📁 监控目录: {self.workflow_dir}")
        print(f"⏱️  刷新间隔: {interval}秒")

        last_workflow_count = 0

        while True:
            workflows = self.scan_workflows()

            if len(workflows) > last_workflow_count:
                print(f"\n🆕 发现新workflow! (总数: {len(workflows)})")
                self.print_operator_evolution(workflows)
                last_workflow_count = len(workflows)

            time.sleep(interval)

def main():
    import sys

    output_dir = "/content/drive/MyDrive/agentflow/outputs/quick_test"
    if len(sys.argv) > 1:
        output_dir = sys.argv[1]

    monitor = TrainingMonitor(output_dir)

    # 先打印一次当前状态
    workflows = monitor.scan_workflows()
    monitor.print_operator_evolution(workflows)

    # 如果指定了持续监控
    if "--watch" in sys.argv:
        monitor.monitor_loop()

if __name__ == "__main__":
    main()
