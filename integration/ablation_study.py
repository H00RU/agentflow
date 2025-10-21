"""
消融实验 Ablation Study
测试不同配置对准确率的影响
"""

import asyncio
import sys
import os
import json
from typing import Dict, Any, List
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'AFlow'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'AFlow', 'scripts'))

from scripts.logs import logger
from scripts.async_llm import create_llm_instance
import workspace.HumanEval.workflows.template.operator as operator


class AblationStudy:
    """消融实验类"""

    def __init__(self, llm_config, problems_subset):
        """
        Args:
            llm_config: LLM配置
            problems_subset: 测试问题子集（例如前20个）
        """
        self.llm_config = llm_config
        self.problems = problems_subset
        self.llm = create_llm_instance(llm_config)

    async def run_all_ablations(self):
        """运行所有消融实验"""
        print("="*80)
        print("  消融实验 Ablation Study")
        print("="*80)
        print(f"测试问题数: {len(self.problems)}")
        print()

        results = {}

        # 1. Baseline: 单次生成 (n=1)
        print("\n1️⃣  Baseline: 单次生成 (n=1, no ensemble)")
        results['baseline_n1'] = await self.test_config(
            n_generate=1,
            use_ensemble=False
        )

        # 2. 双次生成 + ensemble (n=2)
        print("\n2️⃣  双次生成 + ensemble (n=2)")
        results['ensemble_n2'] = await self.test_config(
            n_generate=2,
            use_ensemble=True
        )

        # 3. 三次生成 + ensemble (n=3) - 当前配置
        print("\n3️⃣  三次生成 + ensemble (n=3) - Current")
        results['ensemble_n3'] = await self.test_config(
            n_generate=3,
            use_ensemble=True
        )

        # 4. 五次生成 + ensemble (n=5)
        print("\n4️⃣  五次生成 + ensemble (n=5)")
        results['ensemble_n5'] = await self.test_config(
            n_generate=5,
            use_ensemble=True
        )

        # 5. 三次生成但不用ensemble (n=3, take first)
        print("\n5️⃣  三次生成不用ensemble (n=3, take first)")
        results['no_ensemble_n3'] = await self.test_config(
            n_generate=3,
            use_ensemble=False
        )

        # 打印总结
        self.print_summary(results)

        # 保存结果
        self.save_results(results)

        return results

    async def test_config(
        self,
        n_generate: int,
        use_ensemble: bool
    ) -> Dict[str, Any]:
        """
        测试特定配置

        Args:
            n_generate: 生成候选solution的数量
            use_ensemble: 是否使用ensemble选择

        Returns:
            测试结果
        """
        start_time = datetime.now()

        code_gen = operator.CustomCodeGenerate(self.llm)
        sc_ensemble = operator.ScEnsemble(self.llm)

        passed = 0
        total = len(self.problems)
        details = []

        for i, (task_id, problem) in enumerate(self.problems.items(), 1):
            print(f"  [{i}/{total}] {task_id}...", end='', flush=True)

            try:
                # 生成n个候选solutions
                solutions = []
                for j in range(n_generate):
                    sol = await code_gen(
                        problem=problem['prompt'],
                        entry_point=problem['entry_point'],
                        instruction=""
                    )
                    solutions.append(sol['response'])

                # 选择solution
                if use_ensemble and n_generate > 1:
                    # 使用ensemble
                    result = await sc_ensemble(
                        solutions=solutions,
                        problem=problem['prompt']
                    )
                    solution = result['response']
                else:
                    # 取第一个
                    solution = solutions[0]

                # 测试solution
                is_passed = self._test_solution(
                    solution,
                    problem.get('test', ''),
                    problem['entry_point']
                )

                if is_passed:
                    passed += 1
                    print(" ✅")
                else:
                    print(" ❌")

                details.append({
                    'task_id': task_id,
                    'passed': is_passed
                })

            except Exception as e:
                print(f" ❌ Error: {e}")
                details.append({
                    'task_id': task_id,
                    'passed': False,
                    'error': str(e)
                })

        duration = (datetime.now() - start_time).total_seconds()
        pass_rate = passed / total

        result = {
            'config': {
                'n_generate': n_generate,
                'use_ensemble': use_ensemble
            },
            'passed': passed,
            'total': total,
            'pass_rate': pass_rate,
            'duration': duration,
            'details': details
        }

        print(f"\n  结果: {passed}/{total} = {pass_rate*100:.2f}%")
        print(f"  耗时: {duration:.1f}秒")

        return result

    def _test_solution(
        self,
        solution: str,
        test_code: str,
        entry_point: str
    ) -> bool:
        """测试solution"""
        if not solution or not test_code:
            return False

        try:
            test_env = {}
            exec(solution, test_env)
            exec(test_code, test_env)
            return True
        except:
            return False

    def print_summary(self, results: Dict[str, Dict]):
        """打印总结"""
        print("\n" + "="*80)
        print("  消融实验总结 Ablation Study Summary")
        print("="*80)
        print()

        # 按准确率排序
        sorted_results = sorted(
            results.items(),
            key=lambda x: x[1]['pass_rate'],
            reverse=True
        )

        print(f"{'配置':<30} {'n生成':<10} {'Ensemble':<12} {'Pass Rate':<15} {'提升':<10}")
        print("-"*80)

        baseline_rate = results['baseline_n1']['pass_rate']

        for name, result in sorted_results:
            config = result['config']
            n_gen = config['n_generate']
            use_ens = "✅" if config['use_ensemble'] else "❌"
            pass_rate = result['pass_rate'] * 100
            improvement = (result['pass_rate'] - baseline_rate) * 100

            # 高亮当前配置
            marker = " 👈 Current" if (n_gen == 3 and config['use_ensemble']) else ""

            print(f"{name:<30} {n_gen:<10} {use_ens:<12} {pass_rate:>6.2f}% ({result['passed']}/{result['total']})  {improvement:>+6.2f}%{marker}")

        print()

        # 关键发现
        print("🔍 关键发现:")
        print(f"  • Baseline (n=1): {baseline_rate*100:.2f}%")
        print(f"  • 当前配置 (n=3+ensemble): {results['ensemble_n3']['pass_rate']*100:.2f}%")
        print(f"  • Ensemble的提升: {(results['ensemble_n3']['pass_rate'] - results['no_ensemble_n3']['pass_rate'])*100:.2f}%")
        print(f"  • 增加采样次数(n=3→n=5)的边际收益: {(results['ensemble_n5']['pass_rate'] - results['ensemble_n3']['pass_rate'])*100:.2f}%")
        print()

    def save_results(self, results: Dict):
        """保存结果"""
        filename = f"ablation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        # 转换为可序列化格式
        serializable = {}
        for key, val in results.items():
            serializable[key] = {
                'config': val['config'],
                'passed': val['passed'],
                'total': val['total'],
                'pass_rate': val['pass_rate'],
                'duration': val['duration']
            }

        with open(filename, 'w') as f:
            json.dump(serializable, f, indent=2)

        print(f"✅ 结果已保存到: {filename}")


async def main():
    """主函数"""
    import yaml

    # 加载配置
    with open('deep_config_full_scale.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # 加载HumanEval问题
    print("加载HumanEval数据集...")
    humaneval_path = '/content/agentworkflow/AFlow/data/HumanEval.jsonl'
    problems = {}
    with open(humaneval_path, 'r') as f:
        for line in f:
            problem = json.loads(line)
            problems[problem['task_id']] = problem

    # 选择前20个问题做快速消融测试
    all_ids = list(problems.keys())
    subset_ids = all_ids[:20]  # 前20个问题
    problems_subset = {pid: problems[pid] for pid in subset_ids}

    print(f"✅ 加载 {len(problems_subset)} 个问题用于消融实验")
    print()

    # 创建消融实验
    ablation = AblationStudy(
        llm_config=config['environment']['exec_llm_config'],
        problems_subset=problems_subset
    )

    # 运行实验
    results = await ablation.run_all_ablations()

    print("\n✅ 消融实验完成!")


if __name__ == '__main__':
    asyncio.run(main())
