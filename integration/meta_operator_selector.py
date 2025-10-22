"""
Meta Operator Selector - 元学习Operator选择器
越练越强的智能operator选择系统

功能：
1. 从历史经验中学习最佳operator组合
2. 根据问题特征自动选择策略
3. 持续优化，越用越准确
4. Colab友好，经验保存到Google Drive

适配路径：/content/drive/MyDrive/agentflow/meta_learning/
"""

import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict
import hashlib
from datetime import datetime


@dataclass
class OperatorExperience:
    """单次operator使用经验"""
    problem_text: str
    problem_features: Dict  # 问题特征（长度、关键词等）
    dataset_type: str
    operators_used: List[str]
    workflow_structure: str  # 简化的workflow描述
    success: bool
    score: float
    execution_time: float
    timestamp: str


class ProblemFeatureExtractor:
    """问题特征提取器"""

    # 任务类型关键词
    TASK_KEYWORDS = {
        'code': ['function', 'class', 'implement', 'code', 'program', 'algorithm',
                 'return', 'def', 'write a', 'complete the'],
        'math': ['calculate', 'solve', 'equation', 'number', 'math', 'arithmetic',
                 'compute', 'sum', 'product', 'divide', 'multiply'],
        'reasoning': ['why', 'explain', 'reason', 'analyze', 'think', 'because',
                      'therefore', 'prove', 'demonstrate'],
        'qa': ['who', 'what', 'where', 'when', 'which', 'how many', 'how much'],
        'geometry': ['triangle', 'circle', 'angle', 'area', 'perimeter', 'volume',
                     'square', 'rectangle', 'polygon'],
        'algebra': ['variable', 'polynomial', 'factor', 'simplify', 'expand',
                    'equation', 'inequality', 'expression']
    }

    # 难度指示词
    DIFFICULTY_INDICATORS = {
        'easy': ['simple', 'basic', 'easy', 'straightforward'],
        'medium': ['moderate', 'standard', 'typical'],
        'hard': ['complex', 'difficult', 'challenging', 'advanced', 'prove']
    }

    def extract(self, problem: str, dataset_type: str = None) -> Dict:
        """
        提取问题特征

        Returns:
            {
                'length': int,
                'has_code_block': bool,
                'has_numbers': bool,
                'task_type_scores': Dict[str, float],
                'difficulty_score': float,
                'dataset_type': str,
                'embedding': List[float]  # 用于神经网络的特征向量
            }
        """
        problem_lower = problem.lower()

        features = {
            'length': len(problem),
            'word_count': len(problem.split()),
            'has_code_block': '```' in problem or 'def ' in problem_lower,
            'has_numbers': any(char.isdigit() for char in problem),
            'has_equations': any(sym in problem for sym in ['=', '+', '-', '*', '/', '^']),
            'dataset_type': dataset_type or 'unknown'
        }

        # 任务类型评分
        task_scores = {}
        for task_type, keywords in self.TASK_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw in problem_lower)
            task_scores[task_type] = score / len(keywords)  # 归一化
        features['task_type_scores'] = task_scores
        features['primary_task_type'] = max(task_scores, key=task_scores.get)

        # 难度评分
        difficulty_score = 0
        for difficulty, indicators in self.DIFFICULTY_INDICATORS.items():
            matches = sum(1 for ind in indicators if ind in problem_lower)
            if difficulty == 'easy':
                difficulty_score -= matches * 0.3
            elif difficulty == 'hard':
                difficulty_score += matches * 0.5
        features['difficulty_score'] = max(0, min(1, (difficulty_score + 1) / 2))

        # 生成embedding（固定长度特征向量）
        features['embedding'] = self._create_embedding(features, task_scores)

        return features

    def _create_embedding(self, features: Dict, task_scores: Dict) -> List[float]:
        """创建固定长度的特征向量（用于神经网络输入）"""
        embedding = [
            features['length'] / 1000.0,  # 归一化长度
            features['word_count'] / 200.0,
            float(features['has_code_block']),
            float(features['has_numbers']),
            float(features['has_equations']),
            features['difficulty_score'],
        ]

        # 添加任务类型得分
        embedding.extend([task_scores[t] for t in sorted(task_scores.keys())])

        return embedding


class OperatorSelectorNetwork(nn.Module):
    """Operator选择神经网络"""

    def __init__(self, input_dim: int, num_operator_combos: int):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_operator_combos)
        )

    def forward(self, x):
        return self.network(x)


class MetaOperatorSelector:
    """
    元学习Operator选择器

    核心思想：
    1. 记录每次operator使用的效果
    2. 从历史经验中学习问题特征 -> 最佳operator的映射
    3. 使用小型神经网络预测新问题应该用哪些operator
    4. 支持持续学习，越用越准确
    """

    # 预定义的operator组合策略
    OPERATOR_STRATEGIES = {
        'code_simple': {
            'operators': ['CustomCodeGenerate'],
            'description': '简单代码生成',
            'best_for': ['HumanEval', 'MBPP']
        },
        'code_with_test': {
            'operators': ['CustomCodeGenerate', 'Test', 'Review'],
            'description': '代码生成+测试+审查',
            'best_for': ['HumanEval', 'MBPP']
        },
        'math_direct': {
            'operators': ['Custom'],
            'description': '直接数学推理',
            'best_for': ['AIME', 'MATH', 'GSM8K']
        },
        'math_ensemble': {
            'operators': ['Custom', 'ScEnsemble'],
            'description': '多次数学推理+集成',
            'best_for': ['AIME', 'MATH']
        },
        'math_code': {
            'operators': ['Custom', 'Programmer', 'ScEnsemble'],
            'description': '数学推理+代码验证+集成',
            'best_for': ['AIME', 'GSM8K']
        },
        'reasoning': {
            'operators': ['Custom', 'Review', 'Revise'],
            'description': '推理+审查+修订',
            'best_for': ['general reasoning']
        },
        'qa_simple': {
            'operators': ['Custom'],
            'description': '简单问答',
            'best_for': ['QA tasks']
        },
        'qa_ensemble': {
            'operators': ['Custom', 'ScEnsemble'],
            'description': '多次问答+集成',
            'best_for': ['complex QA']
        }
    }

    def __init__(self, save_dir: str = None):
        """
        初始化元学习选择器

        Args:
            save_dir: 经验和模型保存目录（默认用Google Drive路径）
        """
        # 自动检测路径
        if save_dir is None:
            if os.path.exists("/content/drive/MyDrive"):
                # Colab环境
                save_dir = "/content/drive/MyDrive/agentflow/meta_learning"
            else:
                # 本地环境
                save_dir = os.path.join(os.path.dirname(__file__), "..", "meta_learning")

        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

        self.feature_extractor = ProblemFeatureExtractor()

        # 经验数据库
        self.experience_db: List[OperatorExperience] = []
        self.strategy_stats = defaultdict(lambda: {'total': 0, 'success': 0, 'avg_score': 0.0})

        # 神经网络选择器
        self.embedding_dim = len(self.feature_extractor.extract("test")['embedding'])
        self.num_strategies = len(self.OPERATOR_STRATEGIES)
        self.selector_net = OperatorSelectorNetwork(self.embedding_dim, self.num_strategies)
        self.optimizer = torch.optim.Adam(self.selector_net.parameters(), lr=0.001)

        # 加载历史经验
        self._load_state()

        print(f"[MetaOperatorSelector] Initialized with {len(self.experience_db)} historical experiences")
        print(f"[MetaOperatorSelector] Save directory: {save_dir}")

    def select_operators(self,
                        problem: str,
                        dataset_type: str = None,
                        use_exploration: bool = True,
                        exploration_rate: float = 0.1) -> Dict:
        """
        为给定问题选择最佳operator组合

        Args:
            problem: 问题文本
            dataset_type: 数据集类型（可选）
            use_exploration: 是否使用探索（训练时建议True，测试时False）
            exploration_rate: 探索率

        Returns:
            {
                'strategy_name': str,
                'operators': List[str],
                'confidence': float,
                'reason': str
            }
        """
        # 提取问题特征
        features = self.feature_extractor.extract(problem, dataset_type)

        # 使用神经网络预测
        embedding = torch.FloatTensor(features['embedding']).unsqueeze(0)

        with torch.no_grad():
            logits = self.selector_net(embedding)
            probs = F.softmax(logits, dim=1).squeeze()

        # 探索 vs 利用
        if use_exploration and np.random.random() < exploration_rate:
            # 探索：随机选择策略（但偏向未充分尝试的）
            strategy_idx = self._exploration_select()
            reason = "exploration"
        else:
            # 利用：选择最高概率的策略
            strategy_idx = torch.argmax(probs).item()
            reason = "exploitation"

        strategy_name = list(self.OPERATOR_STRATEGIES.keys())[strategy_idx]
        strategy = self.OPERATOR_STRATEGIES[strategy_name]

        result = {
            'strategy_name': strategy_name,
            'operators': strategy['operators'],
            'confidence': probs[strategy_idx].item(),
            'reason': reason,
            'description': strategy['description'],
            'problem_features': features
        }

        print(f"[MetaOperatorSelector] Selected '{strategy_name}' for {dataset_type} "
              f"(confidence: {result['confidence']:.3f}, {reason})")

        return result

    def record_experience(self,
                         problem: str,
                         dataset_type: str,
                         strategy_name: str,
                         operators_used: List[str],
                         workflow_structure: str,
                         success: bool,
                         score: float,
                         execution_time: float = 0.0):
        """
        记录一次operator使用经验

        这是元学习的核心：每次执行后都记录结果，用于后续学习
        """
        features = self.feature_extractor.extract(problem, dataset_type)

        experience = OperatorExperience(
            problem_text=problem[:200],  # 只保存前200字符
            problem_features=features,
            dataset_type=dataset_type,
            operators_used=operators_used,
            workflow_structure=workflow_structure,
            success=success,
            score=score,
            execution_time=execution_time,
            timestamp=datetime.now().isoformat()
        )

        self.experience_db.append(experience)

        # 更新策略统计
        stats = self.strategy_stats[strategy_name]
        stats['total'] += 1
        if success:
            stats['success'] += 1
        stats['avg_score'] = (stats['avg_score'] * (stats['total'] - 1) + score) / stats['total']

        print(f"[MetaOperatorSelector] Recorded experience: {strategy_name} "
              f"(score: {score:.3f}, success: {success})")

        # 定期训练和保存
        if len(self.experience_db) % 10 == 0:
            self._train_selector()
            self._save_state()

    def _train_selector(self, num_epochs: int = 50):
        """训练选择器神经网络"""
        if len(self.experience_db) < 20:
            print("[MetaOperatorSelector] Not enough experience to train (need ≥20)")
            return

        print(f"[MetaOperatorSelector] Training selector with {len(self.experience_db)} experiences...")

        # 准备训练数据
        X = []
        y = []
        weights = []

        strategy_list = list(self.OPERATOR_STRATEGIES.keys())

        for exp in self.experience_db:
            X.append(exp.problem_features['embedding'])

            # 找到使用的策略索引
            strategy_idx = None
            for idx, (name, strategy) in enumerate(self.OPERATOR_STRATEGIES.items()):
                if set(strategy['operators']) == set(exp.operators_used):
                    strategy_idx = idx
                    break

            if strategy_idx is None:
                strategy_idx = 0  # 默认第一个

            y.append(strategy_idx)

            # 权重：成功的经验权重更高
            weight = exp.score if exp.success else 0.1
            weights.append(weight)

        X = torch.FloatTensor(X)
        y = torch.LongTensor(y)
        weights = torch.FloatTensor(weights)

        # 训练
        self.selector_net.train()
        for epoch in range(num_epochs):
            self.optimizer.zero_grad()

            logits = self.selector_net(X)
            loss = F.cross_entropy(logits, y, reduction='none')
            loss = (loss * weights).mean()  # 加权损失

            loss.backward()
            self.optimizer.step()

            if (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")

        self.selector_net.eval()
        print("[MetaOperatorSelector] Training completed")

    def _exploration_select(self) -> int:
        """探索式选择：偏向尝试次数少的策略"""
        strategy_names = list(self.OPERATOR_STRATEGIES.keys())

        # 计算每个策略的尝试次数
        attempts = [self.strategy_stats[name]['total'] for name in strategy_names]
        total_attempts = sum(attempts) + len(strategy_names)  # +N避免除零

        # 反向概率：尝试少的概率高
        inv_probs = [(total_attempts - att + 1) for att in attempts]
        probs = np.array(inv_probs) / sum(inv_probs)

        return np.random.choice(len(strategy_names), p=probs)

    def get_statistics(self) -> Dict:
        """获取学习统计信息"""
        stats = {
            'total_experiences': len(self.experience_db),
            'strategy_performance': {}
        }

        for name, s in self.strategy_stats.items():
            if s['total'] > 0:
                stats['strategy_performance'][name] = {
                    'attempts': s['total'],
                    'success_rate': s['success'] / s['total'],
                    'avg_score': s['avg_score']
                }

        # 找出最佳策略
        if self.strategy_stats:
            best_strategy = max(self.strategy_stats.items(),
                              key=lambda x: x[1]['avg_score'] if x[1]['total'] > 0 else 0)
            stats['best_strategy'] = {
                'name': best_strategy[0],
                'avg_score': best_strategy[1]['avg_score']
            }

        return stats

    def _save_state(self):
        """保存状态到磁盘（Google Drive）"""
        try:
            # 保存经验数据库
            exp_path = os.path.join(self.save_dir, 'experience_db.json')
            with open(exp_path, 'w') as f:
                experiences_dict = [asdict(exp) for exp in self.experience_db]
                json.dump(experiences_dict, f, indent=2)

            # 保存策略统计
            stats_path = os.path.join(self.save_dir, 'strategy_stats.json')
            with open(stats_path, 'w') as f:
                json.dump(dict(self.strategy_stats), f, indent=2)

            # 保存神经网络
            model_path = os.path.join(self.save_dir, 'selector_net.pt')
            torch.save({
                'model_state_dict': self.selector_net.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
            }, model_path)

            print(f"[MetaOperatorSelector] State saved to {self.save_dir}")
        except Exception as e:
            print(f"[MetaOperatorSelector] Error saving state: {e}")

    def _load_state(self):
        """从磁盘加载状态"""
        try:
            # 加载经验数据库
            exp_path = os.path.join(self.save_dir, 'experience_db.json')
            if os.path.exists(exp_path):
                with open(exp_path, 'r') as f:
                    experiences_dict = json.load(f)
                    self.experience_db = [
                        OperatorExperience(**exp) for exp in experiences_dict
                    ]
                print(f"[MetaOperatorSelector] Loaded {len(self.experience_db)} experiences")

            # 加载策略统计
            stats_path = os.path.join(self.save_dir, 'strategy_stats.json')
            if os.path.exists(stats_path):
                with open(stats_path, 'r') as f:
                    stats = json.load(f)
                    self.strategy_stats = defaultdict(
                        lambda: {'total': 0, 'success': 0, 'avg_score': 0.0},
                        stats
                    )

            # 加载神经网络
            model_path = os.path.join(self.save_dir, 'selector_net.pt')
            if os.path.exists(model_path):
                checkpoint = torch.load(model_path, map_location='cpu')
                self.selector_net.load_state_dict(checkpoint['model_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.selector_net.eval()
                print("[MetaOperatorSelector] Loaded trained model")
        except Exception as e:
            print(f"[MetaOperatorSelector] Error loading state: {e}")
            print("[MetaOperatorSelector] Starting fresh")


def test_meta_selector():
    """测试元学习选择器"""
    print("="*80)
    print("Testing Meta Operator Selector")
    print("="*80)

    # 创建选择器（自动检测路径）
    selector = MetaOperatorSelector()

    # 测试不同类型的问题
    test_problems = [
        ("Write a function to calculate fibonacci numbers", "HumanEval"),
        ("Solve the equation: 2x + 5 = 13", "AIME"),
        ("What is the area of a circle with radius 5?", "GSM8K"),
        ("Implement a binary search algorithm", "MBPP"),
        ("Prove that the sum of angles in a triangle is 180 degrees", "MATH")
    ]

    print("\n1. Testing operator selection:")
    print("-" * 80)
    for problem, dataset in test_problems:
        result = selector.select_operators(problem, dataset, use_exploration=False)
        print(f"\nProblem: {problem[:60]}...")
        print(f"Dataset: {dataset}")
        print(f"Selected: {result['strategy_name']}")
        print(f"Operators: {result['operators']}")
        print(f"Confidence: {result['confidence']:.3f}")

    # 模拟记录经验
    print("\n2. Simulating experience recording:")
    print("-" * 80)
    for problem, dataset in test_problems:
        result = selector.select_operators(problem, dataset)
        # 模拟随机结果
        success = np.random.random() > 0.3
        score = np.random.random() * 0.5 + 0.3 if success else np.random.random() * 0.3

        selector.record_experience(
            problem=problem,
            dataset_type=dataset,
            strategy_name=result['strategy_name'],
            operators_used=result['operators'],
            workflow_structure="test_structure",
            success=success,
            score=score,
            execution_time=1.0
        )

    # 显示统计
    print("\n3. Statistics:")
    print("-" * 80)
    stats = selector.get_statistics()
    print(f"Total experiences: {stats['total_experiences']}")
    print("\nStrategy performance:")
    for name, perf in stats['strategy_performance'].items():
        print(f"  {name}:")
        print(f"    Attempts: {perf['attempts']}")
        print(f"    Success rate: {perf['success_rate']:.2%}")
        print(f"    Avg score: {perf['avg_score']:.3f}")


if __name__ == "__main__":
    test_meta_selector()
