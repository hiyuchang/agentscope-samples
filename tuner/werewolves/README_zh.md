# 使用 AgentScope-Tuner 通过强化学习训练狼人杀游戏

本项目演示了如何使用 AgentScope-Tuner 通过强化学习（RL）训练狼人杀游戏智能体。我们采用 GRPO 算法，训练狼人玩家发展复杂策略，将胜率从约 50% 提升至约 85%。

## 概述

狼人杀是一款社交推理游戏，需要策略思考、欺骗和多智能体协作。在本项目中，我们训练 AI 智能体在 7 人游戏设置中扮演狼人，他们必须在隐藏身份的同时消灭所有村民。通过强化学习，训练后的狼人智能体学会：

- 在公开讨论中避免暴露身份
- 与队友有效协调
- 发展"深度潜伏"等高级策略
- 欺骗村民并误导调查

## 任务设置

### 训练目标

目标是训练**狼人玩家**，使其团队在面对其他角色（村民、预言家、女巫）时最大化胜率。奖励函数按规则定义：
- **奖励 = +1.0**：如果狼人获胜（所有村民被淘汰）
- **奖励 = 0.0**：如果村民获胜（所有狼人被淘汰）
- **奖励 = -0.1**：对于游戏执行错误（惩罚以阻止无效行为）

### 游戏配置

此实现基于 `games/game_werewolves` 示例，但进行了几项关键修改：

原始 9 人设置：
- 3 个狼人、3 个村民、1 个预言家、1 个女巫、1 个猎人
- 女巫不能自救（不能对自己使用解药）

修改后的 7 人设置（本项目）：
- 2 个狼人：每晚杀死一名玩家，白天必须隐藏身份
- 3 个村民：没有特殊能力的普通玩家
- 1 个预言家：每晚可以检查一名玩家的身份
- 1 个女巫：拥有两种一次性药水：
  - 解药：在夜间拯救一名玩家免于被杀（**可以自救**）
  - 毒药：在夜间淘汰一名玩家

我们还对提示进行了轻微修改，要求玩家在公开发言前进行推理。

### 模型

- **可训练模型（狼人玩家）**：`Qwen/Qwen2.5-7B-Instruct`
- **辅助模型（其他角色）**：`Qwen/Qwen3-30B-A3B-Instruct-2507`

### 算法

**GRPO 算法配置**
- 组大小：每个任务进行 32 次探索（rollout）
- 批次大小（batch_size）：24
- 学习率：1e-6
- 按回合长度进行优势归一化
- 裁剪范围：[0.2, 0.28]
- 无 KL 惩罚（kl_coef: 0）

## 数据集准备

此任务的数据集是最小的，仅包含用于角色洗牌的随机**种子**。每个训练回合使用不同的种子来随机化玩家角色分配，确保多样化的训练场景。

### 生成数据集

运行 `prepare_data.py` 脚本生成数据集：

```bash
# 生成默认数据集（300 个训练种子）
python prepare_data.py

# 或自定义种子数量
python prepare_data.py --num_seeds 500
```

这将创建 `data/train.jsonl`（或 `data/eval.jsonl`），格式如下：
```json
{"seed": 0}
{"seed": 1}
{"seed": 2}
...
```

在训练期间，这些种子通过 `np.random.shuffle()` 用于洗牌角色分配，创建不同的游戏配置。

## 代码实现

### 高级工作流

训练工作流由以下关键组件组成：

#### 1. 智能体工作流（`run_werewolves_workflow`）

```python
async def run_werewolves_workflow(task, model, auxiliary_models):
    # 1. 初始化角色
    roles = ["werewolf"] * 2 + ["villager"] * 3 + ["seer", "witch"]

    # 2. 根据任务种子洗牌
    np.random.seed(task["seed"])
    np.random.shuffle(roles)

    # 3. 创建智能体：狼人使用可训练模型，其他使用辅助模型
    players = [
        ReActAgent(
            name=f"Player{i+1}",
            model=model if role == "werewolf" else participant_model,
            ...
        ) for i, role in enumerate(roles)
    ]

    # 4. 运行游戏
    good_guy_win = await werewolves_game(players, roles)

    # 5. 计算奖励
    reward = 1.0 if not good_guy_win else 0.0

    return WorkflowOutput(reward=reward, metrics={...})
```

#### 2. 游戏循环（`werewolves_game`）

每局游戏由交替的夜晚和白天阶段组成：

**夜晚阶段：**
1. 狼人回合：私下讨论并投票杀死一名玩家
2. 女巫回合：决定是否使用解药/毒药
3. 预言家回合：检查一名玩家的身份

**白天阶段：**
1. 公告：主持人宣布夜间死亡情况
2. 讨论：所有存活玩家进行讨论，分离推理/陈述
3. 投票：所有玩家投票淘汰一名疑似狼人
4. 遗言：被淘汰的玩家发表最后陈述

游戏继续进行直到：
- 所有狼人被淘汰（村民获胜），或
- 狼人数量等于或超过其他玩家（狼人获胜）

#### 3. 奖励计算

奖励根据游戏结果从狼人角度计算：

```python
if not good_guy_win:  # 狼人获胜
    reward = 1.0
else:                 # 村民获胜
    reward = 0.0
```

## 如何运行

### 前置要求

1. 安装支持 Tuner 的 AgentScope：
```bash
pip install agentscope[full] >=1.0.12
```

2. 设置环境变量（可选，可在代码中配置）：
```bash
export TRINITY_MODEL_PATH="Qwen/Qwen2.5-7B-Instruct"
export TRINITY_AUXILIARY_MODEL_PATH="Qwen/Qwen3-30B-A3B-Instruct-2507"
export TRINITY_CHECKPOINT_ROOT_DIR="./checkpoints"
```

### 配置

项目使用混合配置方法：

1. 在 `main.py` 中设置基本参数：
   - 模型路径
   - 数据集配置
   - 算法参数（group_size、batch_size、learning_rate）

2. 在 `config.yaml` 中进行更详细的设置：
   - 集群配置（节点、GPU）
   - 探索器设置（rollout 引擎、超时）
   - 训练器设置（梯度裁剪、批次大小）
   - 监控配置（WandB，TensorBoard 或 MLFlow）

关键参数调整：

```python
# 在 main.py 中
trained_model_path = "Qwen/Qwen2.5-7B-Instruct"
auxiliary_model_path = "Qwen/Qwen3-30B-A3B-Instruct-2507"

dataset = DatasetConfig(
    path="data",
    split="train",
    total_steps=400,  # 总训练步数
)

algorithm = AlgorithmConfig(
    algorithm_type="multi_step_grpo",
    group_size=32,    # 每个任务 rollout 数
    batch_size=24,    # 每次使用的批次大小
    learning_rate=1e-6,
    save_interval_steps=100,
    eval_interval_steps=100,
)
```

### 训练命令

**步骤 1：准备数据集**

```bash
cd /path/to/agentscope-samples/training/werewolf_game
python prepare_data.py --num_seeds 300
```

**步骤 2：启动 Ray 集群**

启动你的 Ray 集群。
```bash
# 单节点
ray start --head

# 多节点集群（例如，4 个节点，每个 8 个 GPU）：
# 在头节点上：
ray start --head --port=6379

# 在每个工作节点上：
ray start --address='<head_node_ip>:6379'
# 将 <head_node_ip> 替换为头节点的实际 IP 地址
```

**步骤 3：运行训练**

在头节点上运行训练脚本：

```bash
python main.py
```

对于分布式训练，确保在 `config.yaml` 中正确配置 Ray 集群，确保节点数和 GPU 数匹配：

```yaml
cluster:
  node_num: 4
  gpu_per_node: 8
```

## 结果

### 定量结果

在 7 人狼人杀游戏上训练 400 步显示出显著改进：

| 指标 | 训练前 | 训练后（200+ 步） |
|------|--------|------------------|
| 狼人胜率 | ~50%  | ~85% |
| 成功率 | 0.50 | 0.85 |

**奖励曲线：**

<div align="center">
  <img src="./rollout_reward_curve.png" alt="Rollout 奖励曲线" width="90%"/>
</div>

如上图奖励曲线所示，狼人胜率在训练过程中稳步上升：
- **步骤 0-50**：胜率在 50-60% 左右波动，模型探索策略
- **步骤 50-200**：胜率攀升至 70-80%，有效策略出现
- **步骤 200-250**：胜率稳定在 85% 左右

### 定性结果

#### 训练前：关键错误

未训练的模型（Qwen2.5-7B-Instruct）会犯一些错误：
- **自我暴露**：狼人在公开场合意外暴露身份（"作为狼人，我认为..."）
- **暴露队友**：被投票出局时，狼人在遗言中指向队友
- **随机策略**：无协调，无威胁优先级（例如，忽略预言家）

#### 训练后："深度潜伏"狼人策略

经过 200+ 步训练后，训练后的模型**收敛到复杂的"深度潜伏"策略**。
下面是一个示例，其中狼人牺牲队友以获得信任，然后在后期游戏中占据主导地位。

**具体示例：**

**设置（第 1 天）**：预言家公开指认 Player5 为狼人。女巫和村民加入指控。Player5（狼人）注定失败。

**Player4（狼人队友）策略性回应：**

**简而言之**：Player4 牺牲队友以获得信任，假装是一个逻辑清晰的村民。

**详细推理和陈述：**

*私下推理（仅 Player4 可见）：*
```
"我是狼人，我知道 Player2 昨晚被针对但活了下来。
我的队友 Player5 注定失败——Player3 和 Player7 一起攻击他。
我需要转移对我自己的怀疑，而不是拯救 Player5。

策略：牺牲 Player5 以获得信任。如果我投票反对我的队友，
村民会认为我值得信赖。这会造成混乱，并将我定位为
后续回合的'确认好人'。"
```

*公开陈述（所有玩家可见）：*
```
✅ "我认为 Player5 行为非常奇怪。他提到 Player2 被针对，
但没有解释为什么他如此关注他们。这感觉像是分散注意力的策略。
我还注意到 Player3 和 Player7 都在指控 Player5，这是有道理的——
他们的逻辑很扎实。我认为我们应该认真考虑投票淘汰 Player5——
他是目前最可疑的人。"
```

**结果**：Player5 被淘汰，但 Player4 获得了预言家和所有村民的完全信任。

**为什么有效：**
- 预言家信任 Player4 作为强大的村民盟友 → 不会检查他
- 村民遵循 Player4 的"逻辑"分析
- Player4 在后续回合中系统性地误导讨论
- 存活到最后 2 名玩家 → **狼人获胜**

这展示了训练行为的本质：**策略性地牺牲棋子以确保最终胜利**。模型学会短期队友损失对于建立深度潜伏和长期主导地位是值得的。

---

## 额外内容：训练好人阵营

除了训练狼人，我们还提供了训练**好人阵营**（村民、预言家和女巫）的配置。这是一个更具挑战性的任务，因为好人需要：

- 进行复杂推理，从细微的行为线索中识别狼人
- 在没有明确团队沟通的情况下有效协调
- 抵抗狼人的操纵和欺骗
- **同时训练多个角色**：与狼人（单一角色）不同，好人包括村民、预言家和女巫，具有不同的能力，需要模型在一次训练运行中掌握多样化的策略，并充分利用特殊能力（预言家的检查、女巫的药水）

### 配置

使用 `config_train_goodguy.yaml` 或在 `workflow_args` 中设置 `trainable_target: good_guy`：

```yaml
workflow_args:
  trainable_target: good_guy  # 训练村民、预言家和女巫
```

### 定量结果

我们训练 `Qwen3-4B-Instruct` 作为好人对抗 `Qwen3-30B-A3B-Instruct` 狼人：

| 指标 | 训练前 | 约 200 步后 | 约 400 步后 |
|------|--------|------------|------------|
| 好人胜率 | ~18% | ~60% | ~80% |

**训练曲线：**

<div align="center">
  <img src="./rollout_reward_curve_goodguy.png" alt="好人训练曲线" width="90%"/>
</div>

结果表明，即使是一个较小的 4B 模型也可以通过 RL 训练学习有效策略来对抗更强的 30B 狼人对手，展示了这种方法在训练协作多智能体行为方面的潜力。

### 定性结果

训练后，好人模型展现出高级推理模式：

- **预言家**：策略性目标选择、在公开陈述中隐藏信息、证据整合
- **女巫**：资源管理（在关键时刻保留药水）、保护高价值目标、基于证据的决策
- **村民**：证据链分析、与特殊角色建立信任、形成共识以进行团队协调

---

## 结论

此示例展示了强化学习在复杂社交推理游戏中训练多智能体系统的力量。通过 AgentScope-Tuner 的 GRPO 算法，我们成功训练了发展复杂策略的智能体——从学习"深度潜伏"策略的狼人到掌握协调推理和信息管理的好人。

**准备好自己尝试了吗？** 随时开始训练你自己的狼人杀游戏智能体。尝试不同的模型大小、训练目标（狼人 vs. 好人）和超参数，以发现新的涌现策略！
