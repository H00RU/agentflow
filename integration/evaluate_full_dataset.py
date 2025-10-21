#!/usr/bin/env python3
"""
Complete HumanEval Dataset Evaluation Script
完整HumanEval数据集评估脚本

This script evaluates a trained workflow on:
1. All 131 training problems
2. All 33 test problems
3. Complete 164 problems

Usage:
    python3 evaluate_full_dataset.py [--config CONFIG_FILE]
"""

import asyncio
import argparse
import yaml
import sys
import importlib.util
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

# Add paths
sys.path.insert(0, '/content/agentworkflow/AFlow')
sys.path.insert(0, '/root/aflow_verl_integration/integration')

from workflow_evaluator import WorkflowEvaluator


async def load_latest_workflow(config: Dict[str, Any], output_dir: str):
    """Load the latest trained workflow"""
    print("Loading latest workflow...")

    # Find latest workflow directory
    workflow_base = Path(output_dir) / 'workflows_generated' / 'HumanEval'
    if not workflow_base.exists():
        raise FileNotFoundError(f"Workflow directory not found: {workflow_base}")

    # Get all round directories sorted by modification time
    rounds = sorted(
        [d for d in workflow_base.iterdir() if d.is_dir()],
        key=lambda x: x.stat().st_mtime,
        reverse=True
    )

    if not rounds:
        raise FileNotFoundError(f"No workflow rounds found in {workflow_base}")

    latest_round = rounds[0]
    workflow_file = latest_round / 'graph.py'

    if not workflow_file.exists():
        raise FileNotFoundError(f"Workflow file not found: {workflow_file}")

    print(f"  Using workflow: {latest_round.name}")
    print(f"  File: {workflow_file}")

    # Load workflow module
    spec = importlib.util.spec_from_file_location('workflow_module', workflow_file)
    workflow_module = importlib.util.module_from_spec(spec)
    sys.modules['workflow_module'] = workflow_module
    spec.loader.exec_module(workflow_module)

    # Instantiate workflow
    workflow = workflow_module.Workflow(
        name='EvalWorkflow',
        llm_config=config['environment']['exec_llm_config'],
        dataset='HumanEval'
    )

    print("  Workflow loaded successfully\n")
    return workflow, latest_round.name


async def evaluate_full_dataset(config_file: str = 'deep_config_full_scale.yaml'):
    """Evaluate on complete HumanEval dataset"""

    print("="*80)
    print("  Complete HumanEval Dataset Evaluation")
    print("  完整HumanEval数据集评估")
    print("="*80)
    print()

    # Load configuration
    print(f"Loading configuration from: {config_file}")
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    print("  Configuration loaded\n")

    # Initialize evaluator
    print("Initializing WorkflowEvaluator...")
    evaluator = WorkflowEvaluator(
        dataset='HumanEval',
        sample_size=3,
        timeout_per_problem=30
    )

    total_problems = len(evaluator.problems)
    train_size = int(total_problems * 0.8)  # 131
    test_size = total_problems - train_size  # 33

    print(f"  Total problems: {total_problems}")
    print(f"  Training set: {train_size} problems (80%)")
    print(f"  Test set: {test_size} problems (20%)")
    print()

    # Load workflow
    output_dir = config.get('output_dir', './output/full_scale_training')
    workflow, workflow_name = await load_latest_workflow(config, output_dir)

    # Results storage
    results = {
        'timestamp': datetime.now().isoformat(),
        'workflow': workflow_name,
        'config_file': config_file
    }

    # 1. Evaluate on complete training set (131 problems)
    print("="*80)
    print(f"1/3: Evaluating on COMPLETE training set ({train_size} problems)")
    print("="*80)
    start_time = datetime.now()

    train_result = await evaluator.evaluate_workflow(
        workflow,
        num_problems=train_size,
        use_test_set=False,
        random_sample=False  # Evaluate ALL, not random sample
    )

    train_duration = (datetime.now() - start_time).total_seconds()
    print(f"\nCompleted! Duration: {train_duration:.1f}s ({train_duration/60:.1f} min)")
    print(f"Pass@K: {train_result['pass_at_k']:.4f} ({train_result['passed']}/{train_result['total']})")
    print(f"Accuracy: {train_result['pass_at_k']*100:.2f}%")
    print()

    results['training_set'] = {
        'num_problems': train_size,
        'passed': train_result['passed'],
        'total': train_result['total'],
        'pass_at_k': train_result['pass_at_k'],
        'accuracy_pct': train_result['pass_at_k'] * 100,
        'duration_sec': train_duration
    }

    # 2. Evaluate on complete test set (33 problems)
    print("="*80)
    print(f"2/3: Evaluating on COMPLETE test set ({test_size} problems)")
    print("="*80)
    start_time = datetime.now()

    test_result = await evaluator.evaluate_workflow(
        workflow,
        num_problems=test_size,
        use_test_set=True,
        random_sample=False  # Evaluate ALL 33 test problems
    )

    test_duration = (datetime.now() - start_time).total_seconds()
    print(f"\nCompleted! Duration: {test_duration:.1f}s ({test_duration/60:.1f} min)")
    print(f"Pass@K: {test_result['pass_at_k']:.4f} ({test_result['passed']}/{test_result['total']})")
    print(f"Accuracy: {test_result['pass_at_k']*100:.2f}%")
    print()

    results['test_set'] = {
        'num_problems': test_size,
        'passed': test_result['passed'],
        'total': test_result['total'],
        'pass_at_k': test_result['pass_at_k'],
        'accuracy_pct': test_result['pass_at_k'] * 100,
        'duration_sec': test_duration
    }

    # 3. Combined statistics (all 164 problems)
    print("="*80)
    print(f"3/3: Combined statistics (all {total_problems} problems)")
    print("="*80)

    combined_passed = train_result['passed'] + test_result['passed']
    combined_total = train_result['total'] + test_result['total']
    combined_accuracy = combined_passed / combined_total if combined_total > 0 else 0

    print(f"Total passed: {combined_passed}/{combined_total}")
    print(f"Overall accuracy: {combined_accuracy*100:.2f}%")
    print()

    results['combined'] = {
        'num_problems': total_problems,
        'passed': combined_passed,
        'total': combined_total,
        'accuracy': combined_accuracy,
        'accuracy_pct': combined_accuracy * 100
    }

    # Summary
    print("="*80)
    print("  Evaluation Summary")
    print("  评估总结")
    print("="*80)
    print(f"Workflow: {workflow_name}")
    print(f"Timestamp: {results['timestamp']}")
    print()
    print(f"Training set ({train_size} problems):")
    print(f"  Passed: {train_result['passed']}/{train_result['total']}")
    print(f"  Accuracy: {train_result['pass_at_k']*100:.2f}%")
    print(f"  Duration: {train_duration/60:.1f} min")
    print()
    print(f"Test set ({test_size} problems):")
    print(f"  Passed: {test_result['passed']}/{test_result['total']}")
    print(f"  Accuracy: {test_result['pass_at_k']*100:.2f}%")
    print(f"  Duration: {test_duration/60:.1f} min")
    print()
    print(f"Combined (all {total_problems} problems):")
    print(f"  Passed: {combined_passed}/{combined_total}")
    print(f"  Accuracy: {combined_accuracy*100:.2f}%")
    print("="*80)
    print()

    # Save results to file
    timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = f'full_evaluation_{timestamp_str}.txt'

    with open(results_file, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("Complete HumanEval Dataset Evaluation Results\n")
        f.write("完整HumanEval数据集评估结果\n")
        f.write("="*80 + "\n\n")
        f.write(f"Timestamp: {results['timestamp']}\n")
        f.write(f"Workflow: {workflow_name}\n")
        f.write(f"Config: {config_file}\n\n")

        f.write(f"Training Set ({train_size} problems):\n")
        f.write(f"  Passed: {train_result['passed']}/{train_result['total']}\n")
        f.write(f"  Accuracy: {train_result['pass_at_k']*100:.2f}%\n")
        f.write(f"  Duration: {train_duration/60:.1f} min\n\n")

        f.write(f"Test Set ({test_size} problems):\n")
        f.write(f"  Passed: {test_result['passed']}/{test_result['total']}\n")
        f.write(f"  Accuracy: {test_result['pass_at_k']*100:.2f}%\n")
        f.write(f"  Duration: {test_duration/60:.1f} min\n\n")

        f.write(f"Combined (all {total_problems} problems):\n")
        f.write(f"  Passed: {combined_passed}/{combined_total}\n")
        f.write(f"  Accuracy: {combined_accuracy*100:.2f}%\n")

    print(f"Results saved to: {results_file}")
    print()

    return results


def main():
    parser = argparse.ArgumentParser(description='Evaluate on complete HumanEval dataset')
    parser.add_argument(
        '--config',
        type=str,
        default='deep_config_full_scale.yaml',
        help='Configuration file path'
    )

    args = parser.parse_args()

    # Run evaluation
    asyncio.run(evaluate_full_dataset(args.config))


if __name__ == '__main__':
    main()
