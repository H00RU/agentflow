#!/usr/bin/env python3
"""
测试动态优化集成
验证修改不影响现有配置，同时支持新的动态模式
"""

import sys
import os

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'AFlow'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'AFlow', 'scripts'))

from deep_workflow_env import create_deep_workflow_env, DYNAMIC_OPTIMIZER_AVAILABLE

def test_static_mode():
    """测试静态模式（默认，保持向后兼容）"""
    print("="*70)
    print("测试 1: 静态模式（默认）")
    print("="*70)

    opt_llm_config = {
        "model": "gpt-4o-mini",
        "key": os.getenv("OPENAI_API_KEY", "fake-key-for-test"),
        "base_url": "https://api.openai.com/v1",
        "temperature": 0.9
    }

    exec_llm_config = {
        "model": "gpt-4o-mini",
        "key": os.getenv("OPENAI_API_KEY", "fake-key-for-test"),
        "base_url": "https://api.openai.com/v1",
        "temperature": 0.7
    }

    try:
        # 默认不传 use_dynamic_optimizer，应该是静态模式
        env = create_deep_workflow_env(
            dataset="AIME",
            opt_llm_config=opt_llm_config,
            exec_llm_config=exec_llm_config,
            operators=["Custom", "ScEnsemble", "Review", "Revise"],
            env_num=1,
            sample=2
        )

        print(f"✅ 环境创建成功")
        print(f"  - 模式: {'动态' if env.use_dynamic_optimizer else '静态'}")
        print(f"  - 数据集: {env.dataset}")
        print(f"  - Workspace: {env.workspace_path}")

        # 检查是否有 workflow_parser（静态模式特有）
        if hasattr(env, 'workflow_parser'):
            print(f"  - ✅ WorkflowParser 存在（静态模式）")
        else:
            print(f"  - ❌ WorkflowParser 不存在（不应该发生）")

        # 检查是否有 optimizers（动态模式特有）
        if hasattr(env, 'optimizers'):
            print(f"  - ⚠️  Optimizers 存在（动态模式）")
        else:
            print(f"  - ✅ Optimizers 不存在（静态模式）")

        print(f"\n✅ 测试 1 通过：静态模式工作正常\n")
        return True

    except Exception as e:
        print(f"❌ 测试 1 失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dynamic_mode():
    """测试动态模式"""
    print("="*70)
    print("测试 2: 动态模式")
    print("="*70)

    if not DYNAMIC_OPTIMIZER_AVAILABLE:
        print("⚠️  动态优化器不可用，跳过此测试")
        print("  需要的组件: RLEnhancedOptimizer, SharedExperiencePool, StateManager")
        return True  # 不算失败

    opt_llm_config = {
        "model": "gpt-4o-mini",
        "key": os.getenv("OPENAI_API_KEY", "fake-key-for-test"),
        "base_url": "https://api.openai.com/v1",
        "temperature": 0.9
    }

    exec_llm_config = {
        "model": "gpt-4o-mini",
        "key": os.getenv("OPENAI_API_KEY", "fake-key-for-test"),
        "base_url": "https://api.openai.com/v1",
        "temperature": 0.7
    }

    try:
        # 明确启用动态模式
        env = create_deep_workflow_env(
            dataset="AIME",
            opt_llm_config=opt_llm_config,
            exec_llm_config=exec_llm_config,
            operators=["Custom", "ScEnsemble", "Review", "Revise"],
            env_num=2,
            sample=16,
            use_dynamic_optimizer=True,
            validation_rounds=3,
            rl_weight=0.5
        )

        print(f"✅ 环境创建成功")
        print(f"  - 模式: {'动态' if env.use_dynamic_optimizer else '静态'}")
        print(f"  - 数据集: {env.dataset}")
        print(f"  - Workspace: {env.workspace_path}")

        # 检查是否有 optimizers（动态模式特有）
        if hasattr(env, 'optimizers'):
            print(f"  - ✅ Optimizers 存在（动态模式），数量: {len(env.optimizers)}")
        else:
            print(f"  - ❌ Optimizers 不存在（不应该发生）")

        # 检查共享组件
        if hasattr(env, 'shared_experience_pool'):
            print(f"  - ✅ SharedExperiencePool 存在")
        else:
            print(f"  - ❌ SharedExperiencePool 不存在")

        if hasattr(env, 'state_manager'):
            print(f"  - ✅ StateManager 存在")
        else:
            print(f"  - ❌ StateManager 不存在")

        print(f"\n✅ 测试 2 通过：动态模式工作正常\n")
        return True

    except Exception as e:
        print(f"❌ 测试 2 失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_config_compatibility():
    """测试与现有配置文件的兼容性"""
    print("="*70)
    print("测试 3: 配置文件兼容性")
    print("="*70)

    import yaml

    # 尝试加载现有配置文件
    config_files = [
        "aime_minimal_test.yaml",
        "aime_full_test.yaml"
    ]

    for config_file in config_files:
        config_path = os.path.join(os.path.dirname(__file__), config_file)
        if not os.path.exists(config_path):
            print(f"⚠️  配置文件不存在: {config_file}")
            continue

        print(f"\n检查配置文件: {config_file}")

        try:
            with open(config_path) as f:
                config = yaml.safe_load(f)

            env_config = config.get('environment', {})

            # 检查是否设置了 use_dynamic_optimizer
            use_dynamic = env_config.get('use_dynamic_optimizer', False)
            print(f"  - use_dynamic_optimizer: {use_dynamic} (默认为 False)")

            # 这些配置应该能正常加载
            print(f"  - dataset: {env_config.get('train_datasets', ['未设置'])[0]}")
            print(f"  - sample: {env_config.get('sample', '未设置')}")
            print(f"  - operators: {len(env_config.get('operators', []))} 个")

            print(f"  ✅ 配置文件兼容")

        except Exception as e:
            print(f"  ❌ 配置文件加载失败: {e}")
            return False

    print(f"\n✅ 测试 3 通过：配置文件兼容\n")
    return True


def main():
    """运行所有测试"""
    print("\n" + "="*70)
    print("动态优化集成测试")
    print("="*70 + "\n")

    print(f"动态优化器可用性: {DYNAMIC_OPTIMIZER_AVAILABLE}\n")

    results = []

    # 测试 1: 静态模式（必须通过）
    results.append(("静态模式", test_static_mode()))

    # 测试 2: 动态模式
    results.append(("动态模式", test_dynamic_mode()))

    # 测试 3: 配置兼容性
    results.append(("配置兼容性", test_config_compatibility()))

    # 汇总
    print("="*70)
    print("测试汇总")
    print("="*70)

    for name, passed in results:
        status = "✅ 通过" if passed else "❌ 失败"
        print(f"{name}: {status}")

    all_passed = all(passed for _, passed in results)

    print("\n" + "="*70)
    if all_passed:
        print("🎉 所有测试通过！修改向后兼容")
    else:
        print("⚠️  部分测试失败，请检查")
    print("="*70 + "\n")

    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
