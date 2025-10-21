# AFlow 和 verl-agent 详细流程分析

## 目录
1. [AFlow 工作流优化流程](#aflow-workflow)
2. [verl-agent RL训练流程](#verl-agent-workflow)
3. [两个系统的对比](#comparison)

---

# 一、AFlow 工作流优化流程 {#aflow-workflow}

## 1.1 总体架构

```
┌─────────────────────────────────────────────────────────────┐
│                    AFlow 优化循环                              │
│                                                               │
│  ┌─────────────┐      ┌─────────────┐      ┌─────────────┐ │
│  │  数据集加载  │ ───> │  初始工作流  │ ───> │  优化循环    │ │
│  └─────────────┘      └─────────────┘      └─────────────┘ │
│                                                   │          │
│                                                   ↓          │
│         ┌──────────────────────────────────────────┐        │
│         │        蒙特卡洛树搜索 (MCTS)               │        │
│         │  ┌──────────────────────────────────┐   │        │
│         │  │ 1. 选择 (Selection)              │   │        │
│         │  │ 2. 扩展 (Expansion)              │   │        │
│         │  │ 3. 评估 (Evaluation)             │   │        │
│         │  │ 4. 更新 (Update)                 │   │        │
│         │  └──────────────────────────────────┘   │        │
│         └──────────────────────────────────────────┘        │
│                           │                                  │
│                           ↓                                  │
│                  ┌─────────────────┐                        │
│                  │   最优工作流     │                        │
│                  └─────────────────┘                        │
└─────────────────────────────────────────────────────────────┘
```

## 1.2 详细执行流程

### 阶段 1: 初始化 (`run.py`)

```python
# run.py:100-137

1. 解析命令行参数
   - dataset: 数据集类型 (HumanEval/MBPP/GSM8K/MATH/HotpotQA/DROP)
   - sample: 采样数量
   - max_rounds: 最大迭代轮数
   - opt_model_name: 优化模型 (claude-3-5-sonnet)
   - exec_model_name: 执行模型 (gpt-4o-mini)

2. 加载LLM配置
   - 创建优化LLM实例 (用于生成新工作流)
   - 创建执行LLM实例 (用于执行工作流)

3. 下载数据集
   download(["datasets"], force_download=args.if_force_download)

4. 初始化优化器
   optimizer = Optimizer(
       dataset=config.dataset,
       question_type=config.question_type,
       opt_llm_config=opt_llm_config,
       exec_llm_config=exec_llm_config,
       operators=config.operators,  # 预定义的操作符
       ...
   )

5. 开始优化
   optimizer.optimize("Graph")
```

### 阶段 2: MCTS 优化循环 (`optimizer.py:71-119`)

```
┌───────────────────────────────────────────────────────────────────┐
│                     MCTS 优化主循环                                 │
│                                                                     │
│  for round in range(max_rounds):                                   │
│      │                                                              │
│      ├─> 1. 选择 (Selection)                                       │
│      │       从历史工作流中选择表现最好的 sample 个                   │
│      │       data_utils.get_top_rounds(self.sample)                │
│      │                                                              │
│      ├─> 2. 扩展 (Expansion)                                       │
│      │       使用优化LLM生成新的工作流变体                           │
│      │       ├─ 加载历史经验                                        │
│      │       ├─ 构建优化提示词                                      │
│      │       └─ LLM 生成新的工作流代码                              │
│      │                                                              │
│      ├─> 3. 评估 (Evaluation)                                      │
│      │       在验证集上评估新工作流                                  │
│      │       evaluation_utils.evaluate_graph(...)                  │
│      │                                                              │
│      └─> 4. 更新 (Update)                                          │
│            更新经验数据和结果                                        │
│            experience_utils.update_experience(...)                 │
│                                                                     │
│  检查收敛条件                                                        │
│  convergence_utils.check_convergence(top_k=3)                     │
└───────────────────────────────────────────────────────────────────┘
```

### 阶段 3: 工作流生成详解 (`optimizer.py:132-198`)

```python
# optimizer.py:132-198

async def _optimize_graph(self):
    # Step 1: 采样父节点
    top_rounds = self.data_utils.get_top_rounds(self.sample)
    sample = self.data_utils.select_round(top_rounds)

    # Step 2: 读取父节点的工作流
    prompt, graph_load = self.graph_utils.read_graph_files(
        sample["round"], graph_path
    )

    # Step 3: 加载历史经验
    processed_experience = self.experience_utils.load_experience()
    experience = self.experience_utils.format_experience(
        processed_experience, sample["round"]
    )

    # Step 4: 构建优化提示
    graph_optimize_prompt = self.graph_utils.create_graph_optimize_prompt(
        experience=experience,
        score=sample["score"],
        graph=graph[0],
        prompt=prompt,
        operator_description=operator_description,
        type=self.type,
        log_data=log_data
    )

    # Step 5: 调用LLM生成新工作流
    while True:
        response = await self.optimize_llm.call_with_format(
            graph_optimize_prompt,
            graph_formatter
        )

        # Step 6: 检查修改是否重复
        check = self.experience_utils.check_modification(
            processed_experience,
            response["modification"],
            sample["round"]
        )

        if check:  # 不重复则接受
            break

    # Step 7: 保存新工作流
    self.graph_utils.write_graph_files(
        directory, response, self.round + 1, self.dataset
    )

    # Step 8: 评估新工作流
    avg_score = await self.evaluation_utils.evaluate_graph(
        self, directory, validation_n, data, initial=False
    )

    return avg_score
```

### 阶段 4: 工作流评估 (`evaluator.py`)

```
┌──────────────────────────────────────────────────────────┐
│                   工作流评估流程                           │
│                                                           │
│  1. 加载数据集                                             │
│     data_path = f"data/datasets/{dataset}_validate.jsonl"│
│                                                           │
│  2. 创建基准测试实例                                       │
│     benchmark = benchmark_class(                          │
│         name=dataset,                                     │
│         file_path=data_path,                             │
│         log_path=path                                    │
│     )                                                    │
│                                                           │
│  3. 配置工作流                                             │
│     configured_graph = await self._configure_graph(      │
│         dataset, graph, params                           │
│     )                                                    │
│                                                           │
│  4. 运行评估                                               │
│     score = await benchmark.run_evaluation(              │
│         configured_graph, va_list                        │
│     )                                                    │
│     ├─ 遍历验证集的每个问题                                │
│     ├─ 使用工作流生成解答                                  │
│     ├─ 评估解答的正确性                                    │
│     └─ 返回平均得分                                       │
│                                                           │
│  5. 返回结果                                               │
│     return (score, avg_cost, total_cost)                 │
└──────────────────────────────────────────────────────────┘
```

### 阶段 5: 操作符系统 (`operators.py`)

AFlow 提供了一系列预定义的操作符，可以组合成工作流：

```python
# operators.py 中定义的主要操作符

1. Custom
   - 自定义指令的通用操作符
   - interface: __call__(input, instruction)

2. AnswerGenerate
   - 生成问题答案
   - interface: __call__(input: str) -> Tuple[str, str]

3. CustomCodeGenerate
   - 生成代码
   - interface: __call__(problem, entry_point, instruction)

4. ScEnsemble (Self-Consistency)
   - 集成多个解决方案，选择最一致的
   - Paper: Self-Consistency (2203.11171)
   - interface: __call__(solutions: List[str], problem: str)

5. Programmer
   - 生成并执行 Python 代码
   - 支持代码执行和错误处理
   - interface: __call__(problem: str, analysis: str)

6. Test
   - 测试代码解决方案
   - interface: __call__(problem, solution, entry_point, test_loop)

7. Review
   - 审查解决方案
   - interface: __call__(problem, solution, mode)

8. Revise
   - 根据反馈修订解决方案
   - interface: __call__(problem, solution, feedback, mode)

9. MdEnsemble (Medprompt Ensemble)
   - 多次投票集成
   - Paper: Can Generalist Foundation Models... (2311.16452)
   - interface: __call__(solutions: List[str], problem: str)
```

工作流示例：

```python
# workspace/GSM8K/workflows/round_1/graph.py

class Workflow:
    def __init__(self, name, llm_config, dataset):
        self.name = name
        self.dataset = dataset

        # 初始化操作符
        exec_llm = create_llm_instance(llm_config)
        self.custom = Custom(exec_llm)
        self.programmer = Programmer(exec_llm)
        self.sc_ensemble = ScEnsemble(exec_llm)

    async def __call__(self, problem):
        # Step 1: 使用自定义操作符生成多个解决方案
        solutions = []
        for i in range(5):
            solution = await self.custom(
                input=problem,
                instruction=custom_prompt
            )
            solutions.append(solution)

        # Step 2: 使用 Programmer 操作符生成代码解决方案
        code_solution = await self.programmer(
            problem=problem,
            analysis=""
        )
        solutions.append(code_solution)

        # Step 3: 使用 ScEnsemble 集成所有解决方案
        final_answer = await self.sc_ensemble(
            solutions=solutions,
            problem=problem
        )

        return final_answer
```

### 阶段 6: 经验系统 (`experience_utils.py`)

```python
# experience_utils.py:12-96

经验数据结构：
{
    "round_1": {
        "score": 0.75,
        "success": {
            "round_3": {
                "modification": "增加 ScEnsemble 操作符",
                "score": 0.82
            }
        },
        "failure": {
            "round_2": {
                "modification": "移除 Review 操作符",
                "score": 0.68
            }
        }
    }
}

经验使用流程：
1. 加载所有轮次的经验 (load_experience)
2. 格式化为提示词 (format_experience)
   - 显示父节点的原始得分
   - 列出所有失败的修改（禁止重复）
   - 列出所有成功的修改（禁止重复）
3. 检查新修改是否重复 (check_modification)
4. 更新经验数据 (update_experience)
```

---

# 二、verl-agent RL训练流程 {#verl-agent-workflow}

## 2.1 总体架构

```
┌────────────────────────────────────────────────────────────────┐
│                    verl-agent 训练架构                           │
│                                                                 │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐│
│  │  环境     │<-->│  智能体   │<-->│  奖励     │<-->│  优化器  ││
│  │ (Envs)   │    │ (Actor)  │    │ (Reward)  │    │ (PPO)    ││
│  └──────────┘    └──────────┘    └──────────┘    └──────────┘│
│       │               │                │                │       │
│       └───────────────┴────────────────┴────────────────┘       │
│                           │                                     │
│                           ↓                                     │
│              ┌──────────────────────────┐                      │
│              │   GiGPO 优势估计          │                      │
│              │  Episode-level + Step-level│                   │
│              └──────────────────────────┘                      │
└────────────────────────────────────────────────────────────────┘
```

## 2.2 详细训练流程

### 阶段 1: 初始化 (`ray_trainer.py:828-900`)

```python
# ray_trainer.py:828-900

def init_workers(self):
    # Step 1: 创建资源池
    self.resource_pool_manager.create_resource_pool()

    # Step 2: 创建 Actor-Rollout Worker
    #   - Actor: 策略网络，用于更新
    #   - Rollout: 推理引擎，用于生成动作
    actor_rollout_cls = RayClassWithInitArgs(
        cls=self.role_worker_mapping[Role.ActorRollout],
        config=self.config.actor_rollout_ref,
        role="actor_rollout",
    )

    # Step 3: 创建 Critic Worker (如果使用 GAE)
    if self.use_critic:
        critic_cls = RayClassWithInitArgs(
            cls=self.role_worker_mapping[Role.Critic],
            config=self.config.critic
        )

    # Step 4: 创建 Reference Policy Worker (如果计算 KL)
    if self.use_reference_policy:
        ref_policy_cls = RayClassWithInitArgs(
            self.role_worker_mapping[Role.RefPolicy],
            config=self.config.actor_rollout_ref,
            role="ref"
        )

    # Step 5: 初始化环境
    envs, val_envs = make_envs(config)
    #   - 创建并行环境
    #   - 配置环境管理器
    #   - 设置 projection function (动作映射)
```

### 阶段 2: 训练主循环 (`ray_trainer.py:1007-1304`)

```
┌────────────────────────────────────────────────────────────────┐
│                        训练主循环                                 │
│                                                                  │
│  for epoch in range(total_epochs):                              │
│      for batch in train_dataloader:                             │
│          │                                                       │
│          ├─> Step 1: 多轮轨迹收集 (rollout_loop)                │
│          │        ├─ 环境重置                                    │
│          │        ├─ 多步交互循环                                │
│          │        │   ├─ 观察预处理                             │
│          │        │   ├─ 动作生成                               │
│          │        │   ├─ 环境步进                               │
│          │        │   └─ 数据收集                               │
│          │        └─ 返回轨迹数据                                │
│          │                                                       │
│          ├─> Step 2: 计算奖励                                    │
│          │        reward_fn(batch)                              │
│          │                                                       │
│          ├─> Step 3: 计算 log probabilities                     │
│          │        ├─ old_log_prob (当前策略)                    │
│          │        └─ ref_log_prob (参考策略，如果需要)          │
│          │                                                       │
│          ├─> Step 4: 计算价值 (如果使用 Critic)                 │
│          │        values = critic_wg.compute_values(batch)      │
│          │                                                       │
│          ├─> Step 5: 计算优势 (Advantage)                       │
│          │        ├─ GAE: 使用价值函数                          │
│          │        ├─ GRPO: Episode-level 分组                   │
│          │        └─ GiGPO: Episode + Step 双层分组             │
│          │                                                       │
│          ├─> Step 6: 更新 Critic (如果使用)                     │
│          │        critic_wg.update_critic(batch)                │
│          │                                                       │
│          ├─> Step 7: 更新 Actor                                 │
│          │        actor_rollout_wg.update_actor(batch)          │
│          │                                                       │
│          └─> Step 8: 验证 (定期)                                │
│                   self._validate()                              │
│                                                                  │
└────────────────────────────────────────────────────────────────┘
```

### 阶段 3: 多轮轨迹收集详解 (`rollout_loop.py:476-531`)

```python
# rollout_loop.py:476-531

def multi_turn_loop(self, gen_batch, actor_rollout_wg, envs, is_train):
    """
    智能体-环境交互主循环
    """

    # Step 1: 重复 batch 以支持 env grouping
    if is_train:
        gen_batch = gen_batch.repeat(
            repeat_times=self.config.env.rollout.n,
            interleave=True
        )

    # Step 2: 选择采样模式
    if self.config.algorithm.filter_groups.enable and is_train:
        # 动态采样 (Dynamic Sampling - DAPO)
        result = self.dynamic_multi_turn_loop(...)
    else:
        # 普通采样
        result = self.vanilla_multi_turn_loop(...)

    # Step 3: 收集轨迹数据
    gen_batch_output = self.gather_rollout_data(
        total_batch_list=total_batch_list,
        episode_rewards=total_episode_rewards,
        episode_lengths=total_episode_lengths,
        success=total_success,
        traj_uid=total_traj_uid,
        tool_callings=totoal_tool_callings,
    )

    return gen_batch_output
```

### 阶段 4: 环境交互详解 (`rollout_loop.py:277-406`)

```python
# rollout_loop.py:277-406

def vanilla_multi_turn_loop(self, gen_batch, actor_rollout_wg, envs):
    """
    标准的多轮交互循环
    """

    batch_size = len(gen_batch.batch)

    # Step 1: 环境重置，获取初始观察
    obs, infos = envs.reset(
        kwargs=gen_batch.non_tensor_batch.pop('env_kwargs', None)
    )
    #   obs = {
    #       'text': List[str],      # 文本观察
    #       'image': np.ndarray,    # 图像观察（如果有）
    #       'anchor': Any           # 锚点状态（用于 GiGPO）
    #   }

    # Step 2: 为每个环境分配唯一的轨迹 ID
    if self.config.env.rollout.n > 0:  # env grouping
        # 每 n 个环境共享同一个 group ID
        uid_batch = []
        for i in range(batch_size):
            if i % self.config.env.rollout.n == 0:
                uid = str(uuid.uuid4())
            uid_batch.append(uid)
    else:
        # 所有环境使用相同的 ID
        uid = str(uuid.uuid4())
        uid_batch = [uid for _ in range(batch_size)]

    traj_uid = [str(uuid.uuid4()) for _ in range(batch_size)]

    # Step 3: 初始化跟踪变量
    is_done = np.zeros(batch_size, dtype=bool)
    total_batch_list = [[] for _ in range(batch_size)]
    episode_lengths = np.zeros(batch_size)
    episode_rewards = np.zeros(batch_size)

    # Step 4: 交互循环
    for _step in range(self.config.env.max_steps):
        active_masks = np.logical_not(is_done)

        # 4.1 预处理观察
        batch = self.preprocess_batch(gen_batch=gen_batch, obs=obs)
        #     - 应用 chat template
        #     - tokenize
        #     - 处理多模态数据（如果有）

        # 4.2 生成动作
        batch_output = actor_rollout_wg.generate_sequences(batch_input)
        #     - 前向传播
        #     - 采样动作
        #     - 计算 log_prob

        # 4.3 解码动作
        text_actions = self.tokenizer.batch_decode(
            batch.batch['responses'],
            skip_special_tokens=True
        )

        # 4.4 环境步进
        next_obs, rewards, dones, infos = envs.step(text_actions)

        # 4.5 更新统计量
        episode_rewards[active_masks] += rewards[active_masks]
        episode_lengths[active_masks] += 1

        # 4.6 存储步骤数据
        batch.non_tensor_batch['rewards'] = rewards
        batch.non_tensor_batch['active_masks'] = active_masks
        batch.non_tensor_batch['uid'] = uid_batch
        batch.non_tensor_batch['traj_uid'] = traj_uid

        batch_list = to_list_of_dict(batch)
        for i in range(batch_size):
            total_batch_list[i].append(batch_list[i])

        # 4.7 更新 done 状态
        is_done = np.logical_or(is_done, dones)
        obs = next_obs

        # 4.8 检查所有环境是否完成
        if is_done.all():
            break

    # Step 5: 计算成功率
    success = envs.success_evaluator(
        total_infos=total_infos,
        total_batch_list=total_batch_list,
        episode_rewards=episode_rewards,
        episode_lengths=episode_lengths,
    )

    return (total_batch_list, episode_rewards, episode_lengths,
            success, traj_uid, tool_callings)
```

### 阶段 5: GiGPO 优势计算 (`core_gigpo.py:138-171`)

```python
# core_gigpo.py:138-171

def compute_gigpo_outcome_advantage(
    token_level_rewards,
    step_rewards,
    response_mask,
    anchor_obs,
    index,
    traj_index,
    step_advantage_w=1.0,
    mode="mean_norm"
):
    """
    GiGPO 两级优势估计
    """

    # Step 1: Episode-level 相对优势（论文公式 3）
    #   - 根据 episode group ID (index) 分组
    #   - 计算每组的平均回报
    #   - 每个轨迹的优势 = 该轨迹回报 - 组平均回报
    episode_advantages = episode_norm_reward(
        token_level_rewards,
        response_mask,
        index,           # episode group ID
        traj_index,      # trajectory ID
        epsilon,
        remove_std
    )

    # Step 2: 锚点状态分组（论文公式 6）
    #   - 根据 anchor_obs（锚点观察）分组
    #   - 相同/相似的观察被分到同一组
    #   - 每组分配唯一的 step_group_uid
    step_group_uids = build_step_group(
        anchor_obs,
        index,
        enable_similarity,      # 是否启用相似度匹配
        similarity_thresh       # 相似度阈值
    )

    # Step 3: Step-level 相对优势（论文公式 7）
    #   - 根据 step_group_uid 分组
    #   - 计算每组的平均 step reward
    #   - 每个步骤的优势 = 该步骤奖励 - 组平均奖励
    step_advantages = step_norm_reward(
        step_rewards,
        response_mask,
        step_group_uids,  # step group ID
        epsilon,
        remove_std
    )

    # Step 4: 联合优势（论文公式 8）
    #   总优势 = episode优势 + λ * step优势
    scores = episode_advantages + step_advantage_w * step_advantages

    return scores, scores
```

**GiGPO 分组机制可视化：**

```
Episode-level 分组（通过 episode group ID）：
┌────────────────────────────────────────────────────────────┐
│ Group 1 (uid_1):                                           │
│   Traj 1: [obs1, act1, r1] -> [obs2, act2, r2] -> Done   │
│   Traj 2: [obs1, act1, r2] -> [obs3, act3, r3] -> Done   │
│   Traj 3: [obs1, act1, r3] -> [obs2, act2, r1] -> Done   │
│                                                            │
│   Episode优势 = 该轨迹总回报 - Group平均总回报             │
└────────────────────────────────────────────────────────────┘

Step-level 分组（通过 anchor observation）：
┌────────────────────────────────────────────────────────────┐
│ Step Group A (anchor_obs="obs1"):                         │
│   Step 1 from Traj 1                                      │
│   Step 1 from Traj 2                                      │
│   Step 1 from Traj 3                                      │
│                                                            │
│ Step Group B (anchor_obs="obs2"):                         │
│   Step 2 from Traj 1                                      │
│   Step 3 from Traj 3                                      │
│                                                            │
│ Step Group C (anchor_obs="obs3"):                         │
│   Step 2 from Traj 2                                      │
│                                                            │
│   Step优势 = 该步骤奖励 - Step Group平均奖励               │
└────────────────────────────────────────────────────────────┘

最终优势 = Episode优势 + λ × Step优势
```

### 阶段 6: 环境管理系统 (`env_manager.py`)

```python
# 以 ALFWorld 为例

class AlfWorldEnvironmentManager(EnvironmentManagerBase):
    def __init__(self, envs, projection_f, config):
        self.memory = SimpleMemory()  # 历史记忆
        super().__init__(envs, projection_f, config)

    def reset(self, kwargs):
        # Step 1: 环境重置
        text_obs, image_obs, infos = self.envs.reset()

        # Step 2: 提取任务描述
        self.extract_task(text_obs)

        # Step 3: 初始化记忆
        self.memory.reset(batch_size=len(text_obs))

        # Step 4: 构建完整观察
        full_text_obs = self.build_text_obs(
            text_obs,
            self.envs.get_admissible_commands,
            init=True
        )

        return {
            'text': full_text_obs,
            'image': image_obs,
            'anchor': text_obs  # 用于 GiGPO
        }, infos

    def step(self, text_actions):
        # Step 1: 动作映射（从文本到环境动作）
        actions, valids = self.projection_f(
            text_actions,
            self.envs.get_admissible_commands
        )

        # Step 2: 环境步进
        text_obs, image_obs, rewards, dones, infos = self.envs.step(actions)

        # Step 3: 存储历史
        self.memory.store({
            'text_obs': self.pre_text_obs,
            'action': actions
        })

        # Step 4: 构建新观察
        full_text_obs = self.build_text_obs(
            text_obs,
            self.envs.get_admissible_commands
        )

        # Step 5: 添加动作有效性信息
        for i, info in enumerate(infos):
            info['is_action_valid'] = valids[i]

        next_observations = {
            'text': full_text_obs,
            'image': image_obs,
            'anchor': text_obs
        }

        return next_observations, rewards, dones, infos

    def build_text_obs(self, text_obs, admissible_actions, init=False):
        """
        构建智能体的观察文本
        """
        postprocess_text_obs = []

        if not init and self.config.env.history_length > 0:
            # 获取最近的历史
            memory_contexts, valid_lens = self.memory.fetch(
                self.config.env.history_length,
                obs_key="text_obs",
                action_key="action"
            )

        for i in range(len(text_obs)):
            # 格式化可执行动作
            reformatted_admissible_actions = "\n ".join(
                f"'{s}'" for s in admissible_actions[i] if s != 'help'
            )

            if init or self.config.env.history_length <= 0:
                # 无历史的观察
                obs = ALFWORLD_TEMPLATE_NO_HIS.format(
                    current_observation=text_obs[i],
                    admissible_actions=reformatted_admissible_actions
                )
            else:
                # 带历史的观察
                obs = ALFWORLD_TEMPLATE.format(
                    task_description=self.tasks[i],
                    step_count=len(self.memory[i]),
                    history_length=valid_lens[i],
                    action_history=memory_contexts[i],
                    current_step=len(self.memory[i]) + 1,
                    current_observation=text_obs[i],
                    admissible_actions=reformatted_admissible_actions
                )

            postprocess_text_obs.append(obs)

        return postprocess_text_obs
```

**提示词模板示例（ALFWorld）：**

```python
# agent_system/environments/prompts/alfworld.py

ALFWORLD_TEMPLATE = """You are an expert autonomous agent operating in the ALFWorld text-based environment.

Your task is to: {task_description}

Prior to this step, you have already taken {step_count} step(s). Below are the most recent {history_length} observations and the corresponding actions you took:

{action_history}

You are now at step {current_step} and your current observation is:
{current_observation}

Your admissible actions are:
{admissible_actions}

Now it's your turn to take one action for the current step.

You should first reason step-by-step about the current situation, then choose one admissible action. This reasoning process MUST be enclosed within <think> </think> tags. Once you've finished your reasoning, you should choose an admissible action and present it within <action> </action> tags.
"""
```

### 阶段 7: 策略更新 (`ray_trainer.py:1248-1252`)

```python
# ray_trainer.py:1248-1252

# 更新 Actor
batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
actor_output = self.actor_rollout_wg.update_actor(batch)

# Actor Worker 内部执行：
# 1. 重新计算 log_probs 和 entropies
# 2. 计算 PPO loss
#    - ratio = exp(new_log_prob - old_log_prob)
#    - clipped_ratio = clip(ratio, 1-eps, 1+eps)
#    - loss = -min(ratio * advantages, clipped_ratio * advantages)
# 3. 添加 KL loss（如果配置）
#    - kl_loss = kl_divergence(policy || ref_policy)
# 4. 反向传播并更新参数
# 5. 返回训练指标
```

---

# 三、两个系统的对比 {#comparison}

## 3.1 核心区别

| 维度 | AFlow | verl-agent |
|------|-------|------------|
| **优化对象** | 工作流结构（代码） | 策略网络参数（权重） |
| **搜索空间** | 离散的工作流图 | 连续的参数空间 |
| **评估方式** | 在验证集上直接评估 | 在环境中交互获得奖励 |
| **学习方式** | 进化搜索（MCTS） | 梯度下降（RL） |
| **样本效率** | 低（每个工作流需要完整评估） | 中（需要多次交互） |
| **可解释性** | 高（生成的是代码） | 低（黑盒神经网络） |
| **泛化能力** | 需要针对每个数据集搜索 | 训练后可直接部署 |

## 3.2 适用场景

**AFlow 适合：**
- 需要可解释的推理流程
- 数据集相对静态
- 有足够的计算资源进行搜索
- 希望自动化 Prompt Engineering

**verl-agent 适合：**
- 需要实时交互的任务
- 长时程决策问题
- 需要快速部署的应用
- 希望端到端优化智能体

## 3.3 数据流对比

### AFlow 数据流：
```
问题 -> 工作流 -> 解答 -> 评分 -> 更新经验 -> 生成新工作流
```

### verl-agent 数据流：
```
观察 -> 策略 -> 动作 -> 环境 -> 奖励 -> 计算优势 -> 更新策略
```

---

# 四、关键代码索引

## AFlow 关键文件

```
run.py:100-137                      # 主入口
optimizer.py:71-119                 # MCTS 主循环
optimizer.py:132-198                # 工作流生成
evaluator.py:38-54                  # 工作流评估
operators.py:1-411                  # 操作符定义
graph_utils.py:55-95                # 工作流管理
experience_utils.py:12-96           # 经验系统
```

## verl-agent 关键文件

```
ray_trainer.py:1007-1304            # 训练主循环
rollout_loop.py:277-406             # 环境交互
rollout_loop.py:476-531             # 多轮循环
core_gigpo.py:138-171               # GiGPO 优势计算
core_gigpo.py:243-331               # Step-level 分组
env_manager.py:133-243              # ALFWorld 环境管理
env_manager.py:385-518              # WebShop 环境管理
```

---

# 五、配置示例

## AFlow 配置示例

```bash
python run.py \\
    --dataset MATH \\
    --sample 4 \\
    --max_rounds 20 \\
    --opt_model_name claude-3-5-sonnet-20241022 \\
    --exec_model_name gpt-4o-mini \\
    --check_convergence True \\
    --validation_rounds 5
```

## verl-agent (GiGPO) 配置示例

```bash
python -m verl.trainer.main_ppo \\
    algorithm.adv_estimator=gigpo \\
    algorithm.gamma=0.95 \\
    algorithm.gigpo.step_advantage_w=1.0 \\
    algorithm.gigpo.mode=mean_std_norm \\
    env.env_name=alfworld/AlfredTWEnv \\
    env.max_steps=50 \\
    env.rollout.n=8 \\
    data.train_batch_size=16 \\
    actor_rollout_ref.model.path=Qwen/Qwen2.5-1.5B-Instruct \\
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \\
    trainer.total_epochs=150
```

---

# 总结

1. **AFlow** 通过蒙特卡洛树搜索在离散的工作流空间中寻找最优推理流程，适合需要可解释性和自动化 Prompt Engineering 的场景。

2. **verl-agent** 通过强化学习在连续的参数空间中优化智能体策略，特别是 GiGPO 算法通过两级分组机制（Episode + Step）实现了细粒度的信用分配，适合长时程、多轮交互的任务。

3. 两个系统代表了 Agent 优化的两个方向：**结构搜索** vs **参数优化**，可以根据具体需求选择合适的方法。
