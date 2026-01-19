# 使用 AgentScope-Tuner 训练 FrozenLake Agent

## 摘要

本示例展示如何使用 AgentScope-Tuner 配合 [Trinity-RFT](https://github.com/agentscope-ai/Trinity-RFT) 对 [Frozen Lake](https://gymnasium.farama.org/environments/toy_text/frozen_lake/) 任务进行强化微调。智能体需要在冰湖网格中从起点走到终点，避开坑洞，并在有限步数内完成任务。

## 任务设定

### 智能体目标
智能体要在冰湖网格上从起点 (S) 抵达终点 (G)，同时：
- 规划路径经过冰面 (F) 到达终点
- 避开会结束回合且奖励为 0 的坑洞 (H)
- 在限定步数内完成任务

### 智能体类型
智能体实现为 **ReActAgent**，它的行为包括：
- 观察当前冰湖网格状态
- 推理下一步最优动作
- 执行动作（上、下、左、右）在环境中移动
- 在多步交互中维护内部状态

### 环境
环境基于 Gymnasium 的 FrozenLake，并提供：
- **网格导航**：随机生成 2x2 至 6x6 的地图
- **格子类型**：
  - `S`：起点
  - `F`：冰面（可通行）
  - `H`：坑洞（奖励 0，结束回合）
  - `G`：终点（奖励 +1.0，结束回合）
- **动作空间**：离散动作（上、下、左、右）
- **奖励设计**：
  - 到达终点 +1.0
  - 掉入坑洞或未在最大步数内到达终点为 0.0
- **观测**：返回当前玩家位置的文本网格表示

智能体不使用外部工具，直接通过以下接口与环境交互：
- `env.reset(task)`：根据任务参数初始化环境
- `env.step(action)`：执行动作，返回观测、奖励和结束标志
- `env.render()`：返回当前状态的文本表示

## 数据集准备

数据集包含用于生成 FrozenLake 环境的任务参数，每个样本包含：
- `seed`：随机种子，保证地图可复现
- `size`：网格大小（在 2 和 `map_max_size` 之间随机，如 4x4、6x6）
- `p`：格子为冰面的概率（0.6 到 0.85 之间随机），其余为坑洞
- `index`：样本索引
- `uid`：由 seed、size、p 组合而成的唯一 ID

运行数据准备脚本生成训练集与测试集：

```bash
python get_frozenlake_data.py --map_max_size 6 --train_size 10000 --test_size 100
```

生成的目录结构示例：
```
/path/to/frozenlake_dataset/
    ├── train.parquet  # 10000 条训练样本
    └── test.parquet   # 100 条测试样本
```

样本示例：
```json
{"seed": 12345, "size": 5, "p": 0.75, "index": 0, "uid": "12345_5_0.75"}
```

**注意**：脚本会过滤无解的地图，确保在最大步数 (`env_max_steps=8`) 内存在从起点到终点的可行路径。

## 代码实现

本节提供代码实现的高级概览。详细实现请参考源代码。

### 高级概览
实现由三部分组成：
1. **Agent** (`FrozenLakeAgent`)：继承 `ReActAgent`，负责多步交互
2. **环境** (`FrozenLakeEnv`)：封装 Gymnasium FrozenLake
3. **工作流** (`run_frozen_lake`)：组织智能体与环境的交互流程

### 工作流
`run_frozen_lake` 实现多步交互流程：

```python
async def run_frozen_lake(
    task: Dict,
    model: ChatModelBase,
    auxiliary_models: Dict[str, ChatModelBase],
) -> WorkflowOutput:
    # ...

    # 创建智能体和环境
    agent = FrozenLakeAgent(model=model, ...)
    env = FrozenLakeEnv(...)
    observation, _ = env.reset(task)
    rewards = []
    # ...

    # 智能体-环境交互循环
    for _ in range(max_steps):
        response = await agent.reply(msg=Msg("user", agent.get_prompt(observation), role="user"))
        action = agent.get_action(response)
        observation, reward, done, _ = env.step(action)
        rewards.append(reward)
        if done:
            break

    # ...
    final_reward = sum(rewards)
    final_response = Msg("assistant", response_content, role="assistant")

    return WorkflowOutput(
        reward=final_reward,
        response=final_response,
        metrics={"env_steps": float(step_count), "env_done": float(done)},
    )
```

**关键特性：**
- 多步交互：单次 episode 内多次动作，不是单轮 QA
- 状态跟踪：记录当前步、上次动作与观测
- 错误处理：无效动作或异常会被捕获并处理

### 奖励函数
无需额外 judge，奖励由环境直接给出：
- 1.0：到达终点
- 0.0：掉入坑洞或超步数未达终点

工作流返回：
- `reward`：累计奖励
- `response`：包含观测、总奖励、步数、终止原因的最终回复
- `metrics`：`env_steps`（步数）、`env_done`（是否结束）

### 实现细节

环境 (`FrozenLakeEnv`) 封装了 Gymnasium 的 FrozenLake，提供：
- `reset(task)`: 使用任务参数初始化环境
- `step(action)`: 执行动作并返回 (observation, reward, done, info)
- `render()`: 返回当前状态的文本表示

智能体 (`FrozenLakeAgent`) 继承 `ReActAgent`，提供：
- `reply(msg)`: 回复消息并返回动作（继承自 AgentScope）
- `get_prompt(observation)`: 从当前观测生成提示
- `get_action(response)`: 解析模型响应以提取动作（Up/Down/Left/Right）
- `update_state(action, observation)`: 在每步后更新内部状态

详细实现请参考 [frozenlake_env.py](./frozenlake_env.py) 和 [frozenlake_agent.py](./frozenlake_agent.py)。

### 步骤 4：使用 `tune` 训练工作流

```python
from agentscope.tuner import tune, DatasetConfig

if __name__ == "__main__":
    config_path = os.path.join(
        os.path.dirname(__file__),
        "config.yaml",
    )
    dataset = DatasetConfig(
        path="/path/to/frozenlake_dataset",
        name="default",
        split="train",
    )
    tune(
        workflow_func=run_frozen_lake,
        train_dataset=dataset,
        config_path=config_path,
    )
```

训练配置请参考 [config.yaml](./config.yaml)。完整配置详情请参考 [Trinity-RFT 配置指南](https://agentscope-ai.github.io/Trinity-RFT/en/main/tutorial/trinity_configs.html)。

---

## 运行方法

### 依赖
- 至少 2 张 NVIDIA GPU，CUDA 版本 ≥ 12.8
- 按 [Trinity-RFT 安装指南](https://agentscope-ai.github.io/Trinity-RFT/en/main/tutorial/trinity_installation.html) 从源码安装
- 安装 gymnasium 冰湖环境：
  ```bash
  pip install gymnasium[toy_text]
  ```
- 下载模型权重（示例）：
  ```bash
  huggingface-cli download Qwen/Qwen2.5-3B-Instruct
  ```

### 步骤 1：准备数据集
```bash
python get_frozenlake_data.py --map_max_size 6 --train_size 10000 --test_size 100
```
将 `main.py` 中的数据集路径改为你的生成目录。

### 步骤 2：配置训练

关键配置可在代码中设置，包括：

**算法配置** (`AlgorithmConfig`)：
- `algorithm_type`: `multi_step_grpo`（用于多步任务的组相对策略优化）
- `group_size`: 每批次的策略更新组大小（默认 16）
- `batch_size`: 批大小（默认 32）
- `learning_rate`: 学习率（默认 1e-6）

**模型配置** (`TunerModelConfig`)：
- `model_path`: 基础模型路径（如 `Qwen/Qwen2.5-3B-Instruct`）
- `max_model_len`: 最大上下文长度（默认 25600）
- `max_tokens`: 响应最大生成长度（默认 2048）
- `inference_engine_num`: 推理引擎数量（默认 6，表示用 6 个 GPU 进行推理）

**数据集配置** (`DatasetConfig`)：
- `path`: 数据集路径（默认 `/path/to/frozenlake`）
- `split`: 数据集分片（默认 `train`）

可根据硬件资源和训练需求调整这些参数。其他参数可在 [config.yaml](./config.yaml) 中指定。

### 步骤 3：设置 Ray 集群

设置 [Ray](https://github.com/ray-project/ray) 集群：
```bash
ray start --head
# 对于多节点设置，在工作节点上运行以下命令
# ray start --address=<master_address>
```

### 步骤 4：运行训练脚本
```bash
python main.py
```
训练将开始，可通过日志监控进度。检查点将每 `trainer.save_interval` 步保存一次。

## 实验结果

### 训练奖励曲线

训练过程中的奖励曲线显示智能体的学习进度：

<div align="center">
  <img src="./critic_rewards_mean.png" alt="reward" width="90%"/>
</div>

训练奖励通常随着智能体学习更有效地导航冰湖而随训练轮次增加。

### 智能体输出示例

智能体输出示例如下：
```
From the current observation, let's analyze the situation. The player (P) is at: (4, 0), and the goal (G) is at: (2, 3). There is also a hole (O) at (4, 4). Given this, I can move towards the goal without worrying about slippery tiles right now.

The shortest path from P to G involves moving left (4 steps) followed by moving down (1 step), since going directly would bypass the hole or move us further from the goal. Let's move left first.

Let's take the action ```Left```.
```
