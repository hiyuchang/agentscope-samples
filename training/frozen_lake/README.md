# Training FrozenLake Agent with RL using AgentScope-Tuner

## Summary

This example demonstrates how to use AgentScope-Tuner to implement reinforcement fine-tuning for the [Frozen Lake](https://gymnasium.farama.org/environments/toy_text/frozen_lake/) task using [Trinity-RFT](https://github.com/modelscope/Trinity-RFT). The agent learns to navigate a frozen lake grid from a starting position to a goal while avoiding holes through multi-step interactions with the environment.

## Task Setting

### Agent Goal
The agent's objective is to navigate from the starting position (S) to the goal position (G) on a frozen lake grid without falling into holes (H). The agent must:
- Plan a path through frozen tiles (F) to reach the goal
- Avoid holes that terminate the episode with zero reward
- Complete the task within a limited number of steps

### Agent Type
The agent is implemented as a **ReActAgent** (Reasoning and Acting Agent) that:
- Observes the current state of the frozen lake grid
- Reasons about the best action to take
- Executes actions (Up, Down, Left, Right) to move through the environment
- Maintains internal state across multiple steps in an episode

### Environment
The environment is based on Gymnasium's FrozenLake environment, wrapped to provide:
- **Grid-based navigation**: Randomly generated maps with configurable size (2x2 to 6x6)
- **Tile types**:
  - `S`: Start position
  - `F`: Frozen tile (safe to walk on)
  - `H`: Hole (terminates episode with reward 0)
  - `G`: Goal (terminates episode with reward +1.0)
- **Action space**: Discrete actions (Up, Down, Left, Right)
- **Reward structure**:
  - +1.0 for reaching the goal
  - 0.0 for falling into a hole or failing to reach the goal
- **Observations**: Text-based grid representation showing current player position

The agent does not use external tools. It interacts directly with the environment through:
- `env.reset(task)`: Initialize environment with task parameters
- `env.step(action)`: Execute action and receive observation, reward, and done flag
- `env.render()`: Get text representation of current state


## Dataset Preparation

The dataset contains task parameters for generating FrozenLake environments. Each sample specifies:
- `seed`: Random seed for reproducible map generation
- `size`: Grid size (randomly sampled from 2 to `map_max_size`, e.g., 4x4, 6x6)
- `p`: Probability that a tile is frozen (vs. being a hole), randomly sampled from 0.6 to 0.85
- `index`: Sample index
- `uid`: Unique identifier combining seed, size, and p

Run the data preparation script to generate training and test datasets:

```bash
python get_frozenlake_data.py --map_max_size 6 --train_size 10000 --test_size 100
```

This will create parquet files in the specified directory:

```
/path/to/frozenlake_dataset/
    ├── train.parquet  # 10000 training samples
    └── test.parquet   # 100 test samples
```

Each sample looks like:

```json
{"seed": 12345, "size": 5, "p": 0.75, "index": 0, "uid": "12345_5_0.75"}
```

**Note**: The data preparation script ensures that all generated maps have a valid path from start to goal within the maximum allowed steps (`env_max_steps=8`), filtering out unsolvable tasks.

## Code Implementation

This section provides a high-level overview of the code implementation. For detailed implementation, please refer to the source code.

### High-level Overview

The implementation consists of three main components:

1. **Agent** (`FrozenLakeAgent`): Extends `ReActAgent` to handle multi-step navigation
2. **Environment** (`FrozenLakeEnv`): Wraps Gymnasium's FrozenLake environment
3. **Workflow** (`run_frozen_lake`): Orchestrates the agent-environment interaction loop

### Agent Workflow

The workflow function `run_frozen_lake` implements the agent-environment interaction loop:

```python
async def run_frozen_lake(
    task: Dict,
    model: TrinityChatModel,
    auxiliary_models: Dict[str, TrinityChatModel],
) -> WorkflowOutput:
    # ...

    # Create agent and environment
    agent = FrozenLakeAgent(model=model, ...)
    env = FrozenLakeEnv(...)
    observation, _ = env.reset(task)
    rewards = []
    # ...

    # Agent-environment interaction loop
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
        metrics={
            "env_steps": float(step_count),
            "env_done": float(done),
        },
    )

```

**Key characteristics:**
- Multi-step interaction: The agent takes multiple actions in a single episode, unlike single-turn QA tasks
- State tracking: The agent maintains internal state (current step, last action, last observation) across steps
- Error handling: Invalid actions or agent errors are caught and handled gracefully

### Reward Function

No separate judge function is needed. The reward comes directly from the environment:
- 1.0: Agent successfully reaches the goal (G)
- 0.0: Agent falls into a hole (H) or fails to reach the goal within the maximum steps

The reward is computed as the sum of step rewards throughout the episode. The workflow returns:
- `reward`: Final cumulative reward
- `response`: Final response message containing observation, total reward, steps taken, and termination reason
- `metrics`: Additional metrics including `env_steps` (number of steps taken) and `env_done` (whether episode completed)

### Implementation Details

The environment (`FrozenLakeEnv`) wraps Gymnasium's FrozenLake and provides:
- `reset(task)`: Initialize the environment with task parameters
- `step(action)`: Execute an action and return (observation, reward, done, info)
- `render()`: Return a text representation of the current state

The agent (`FrozenLakeAgent`) extends `ReActAgent` and provides:
- `reply(msg)`: Reply to a message and return an action (inherited from AgentScope)
- `get_prompt(observation)`: Generate a prompt from the current observation
- `get_action(response)`: Parse the model's response to extract an action (Up/Down/Left/Right)
- `update_state(action, observation)`: Update internal state after each step

See [frozenlake_env.py](./frozenlake_env.py) and [frozenlake_agent.py](./frozenlake_agent.py) for implementation details.

### Step 4: Use `tune` to train the workflow

```python
from agentscope.tuner import tune, Dataset

if __name__ == "__main__":
    config_path = os.path.join(
        os.path.dirname(__file__),
        "config.yaml",
    )
    dataset = Dataset(
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

See [config.yaml](./config.yaml) for the training configuration. For full configuration details, see [Trinity-RFT Configuration Guide](https://modelscope.github.io/Trinity-RFT/en/main/tutorial/trinity_configs.html).

---

## How to Run

### Prerequisites

- At least 2 NVIDIA GPUs with CUDA 12.8 or newer
- Follow the Trinity-RFT [installation guide](https://modelscope.github.io/Trinity-RFT/en/main/tutorial/trinity_installation.html) to install the latest version from source code
- Install gymnasium for the FrozenLake environment:

  ```bash
  pip install gymnasium[toy_text]
  ```

- Download the model checkpoint (example):

  ```bash
  huggingface-cli download Qwen/Qwen2.5-3B-Instruct
  ```

### Step 1: Prepare the Dataset

```bash
python get_frozenlake_data.py --map_max_size 6 --train_size 10000 --test_size 100
```

Update the dataset path in `main.py` to point to your generated dataset directory.

### Step 2: Configure the Training

Key configuration can be identified in the code, including:

**Algorithm Configuration** (`algorithm`):
- `algorithm_type`: `multi_step_grpo` (Group Relative Policy Optimization for multi-step tasks)
- `group_size`: Number of policy update iterations per batch (default: 16)
- `batch_size`: Batch size for training (default: 32)
- `learning_rate`: Learning rate (default: 1e-6)

**Model Configuration** (`model`):
- `model_path`: Path to the base model (e.g., `Qwen/Qwen2.5-3B-Instruct`)
- `max_model_len`: Maximum model context length (default: 25600)
- `max_tokens`: Maximum tokens for response generation (default: 2048)
- `inference_engine_num`: Number of inference engines (default: 6)

**Dataset Configuration** (`dataset`):
- `path`: Path to the dataset (default: `/path/to/frozenlake`)
- `split`: Split of the dataset (default: `train`)

Adjust these parameters based on your hardware resources and training requirements. Other parameters can be spetified in  [config.yaml](./config.yaml).


### Step 3: Set Up Ray Cluster

Set up a [Ray](https://github.com/ray-project/ray) cluster:

```bash
ray start --head
# for multi-node setup, run the following command on worker nodes
# ray start --address=<master_address>
```

### Step 4: Run the Training Script

```bash
python main.py
```

The training will start and you can monitor the progress through the logs. Checkpoints will be saved once every `trainer.save_interval` steps.

## Experimental Results

### Training Reward Curve

The reward curve during training shows the agent's learning progress:

![reward](./critic_rewards_mean.png)

The training reward typically increases over epochs as the agent learns to navigate the frozen lake more effectively.

### Example Agent Output

An example of agent output is given below:
```
From the current observation, let's analyze the situation. The player (P) is at: (4, 0), and the goal (G) is at: (2, 3). There is also a hole (O) at (4, 4). Given this, I can move towards the goal without worrying about slippery tiles right now.

The shortest path from P to G involves moving left (4 steps) followed by moving down (1 step), since going directly would bypass the hole or move us further from the goal. Let's move left first.

Let's take the action ```Left```.
```