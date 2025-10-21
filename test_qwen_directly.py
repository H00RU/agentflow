"""
直接测试Qwen模型是否工作
不经过复杂的训练循环
"""
import sys
sys.path.insert(0, '/root/aflow_integration/integration')

from qwen_policy import QwenRLPolicy

# 测试Qwen
policy = QwenRLPolicy(
    model_path="/root/models/Qwen2.5-7B-Instruct",
    device="cpu"  # Worker用CPU
)

# 测试状态
test_state = """
Workflow Round: 2
Operators: ['Custom', 'Programmer']
Current Score: 0.78
Parent Score: 0.75
Dataset: HumanEval
"""

print("=" * 60)
print("测试Qwen Q-value估计:")
print("=" * 60)
q_value = policy.get_q_value(test_state)
print(f"Q-value: {q_value:.4f}")

print("\n" + "=" * 60)
print("测试Qwen动作建议:")
print("=" * 60)
suggestion = policy.suggest_action(test_state)
print(f"建议: {suggestion}")

print("\n" + "=" * 60)
print("✅ Qwen模型工作正常！")
print("=" * 60)
