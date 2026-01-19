# 使用数据增强策略训练数学智能体

本示例演示了如何使用 **AgentScope-Tuner** 训练数学问题求解智能体。我们将重点利用**以数据为中心**的功能，例如 `difficulty_based` 任务选择器，以提高数据利用率和训练效率。

## 任务设置

我们使用基础的 [math-agent 示例](https://github.com/agentscope-ai/agentscope-samples/blob/main/tuner/math_agent/main.py) 作为基线。智能体是 **`ReActAgent`**，通过逐步推理解决数学推理问题。

如果任务太容易或太难，训练可能会效率低下。本示例演示如何使用**任务选择器**基于**数据反馈**动态选择任务，专注于"具有挑战性"的样本以最大化训练效率。这些以数据为中心的技术是通用的，可适应其他智能体工作流。

## 数据集准备

为启用基于难度的采样，训练数据必须包含难度特征（如 LLM 的通过率）。

1.  **基础数据集**：您可以使用任何标准的数学问题数据集。一个很好的例子是 [LLM360/guru-RL-92k](https://huggingface.co/datasets/LLM360/guru-RL-92k) 中的数学数据，它预先标注了来自不同 LLM 的通过率，作为直接的难度特征。
2.  **构建您自己的特征**：如果您使用自己的数据集，可以通过预先运行几个不同能力的模型并记录它们的通过率来生成这些特征。这可以在 [**Trinity-RFT**](https://github.com/agentscope-ai/Trinity-RFT/pull/440) 框架内完成。
3.  **数据格式**：最终数据集应为 HuggingFace 格式。在此示例中，数据将根据[工作流](https://github.com/agentscope-ai/agentscope-samples/blob/main/tuner/math_agent/main.py)转换为 *GSM8K 格式*。除了任务内容外，它还必须包含您定义的难度特征列（例如 `qwen2.5_7b_pass_rate`）。
4.  **示例数据准备**：我们为此示例提供了一个脚本。只需执行 `python prepare_data.py` 即可生成所需的数据集。

## 代码实现

本示例采用 [math-agent 示例](https://github.com/agentscope-ai/agentscope-samples/blob/main/tuner/math_agent/main.py) 的 `run_react_agent` 和 `gsm8k_judge` 作为 `workflow_func` 和 `judge_func`，说明可以在不改变核心智能体逻辑的情况下应用训练策略。

### 以数据为中心功能的设计

利用 **Trinity-RFT** 强大的数据处理能力，**AgentScope-Tuner** 为任务选择和经验处理等高级操作提供了接口。

#### 任务选择器

`Task Selector` 决定如何从数据集中选择样本。它可以直接在 YAML 配置文件中配置。

- **内置选择器**：
  - `sequential`：按固定顺序选择样本。
  - `shuffle`：在每个 epoch 开始时打乱数据集。
  - `random`：为每个批次随机选择样本（有放回）。
  - `offline_easy2hard`：按预定义特征对样本进行排序，用于课程学习。
  - `difficulty_based`（自定义）：基于任务难度的自适应采样器。

> 有关 `Task Selector` 的更多详细信息，包括如何基于反馈信号实现自定义选择器，请参阅 **Trinity-RFT** 的 **[Selector 开发指南](https://agentscope-ai.github.io/Trinity-RFT/zh/main/tutorial/develop_selector.html)**。

#### 数据处理器

`Data Processor` 允许在训练期间实时处理**任务**（task）和**经验**（experience），支持计算反馈指标、数据增强或过滤等操作。

例如，`difficulty_based` 选择器需要一个 `pass_rate_calculator` 操作符来计算智能体对每个任务的成功率。然后使用此反馈来调整采样策略。

> 有关 `Data Processor` 的更多详细信息，请参阅 **Trinity-RFT** 的 **[Operator 开发指南](https://agentscope-ai.github.io/Trinity-RFT/zh/main/tutorial/develop_operator.html)**。


### 配置实验

为了保持清晰和简洁，我们建议在 YAML 配置文件中定义所有数据特定参数，包括数据集路径和任务选择器。

我们提供两个配置文件，用于比较基线 `random` 选择器与 `difficulty_based` 选择器。

**实验 1：使用随机选择器的基线（`config_random.yaml`）**

在 `config_random.yaml` 中，我们在 `buffer.explorer_input.taskset` 下配置用于随机采样的 `task_selector`。

```yaml
# 在 config_random.yaml 中
buffer:
  # ...
  explorer_input:
    taskset: # 训练数据
      path: "path/to/your/augmented/math_data"
      split: "train"
      task_selector:
          selector_type: random # 任务选择策略
```

**实验 2：使用基于难度选择器的高级训练（`config_difficulty.yaml`）**

在 `config_difficulty.yaml` 中，我们将 `task_selector` 切换为 `difficulty_based` 并提供其特定参数。请注意，此配置还启用了反馈所需的 `pass_rate_calculator`。

```yaml
# 在 config_difficulty.yaml 中

# 启用计算器为选择器提供反馈
data_processor:
  experience_pipeline:
    operators:
      - name: pass_rate_calculator

buffer:
  # ...
  explorer_input:
    taskset: # 训练数据
      path: "path/to/your/augmented/math_data"
      split: "train"
      task_selector:
        selector_type: difficulty_based # 任务选择策略
        feature_keys: [ "qwen2.5_7b_pass_rate", "qwen3_30b_pass_rate" ]
        kwargs: # 选择算法的超参数
          m: 8
          # ...
```

> 本示例中的 `difficulty_based` 选择器是 ***BOTS*** 算法的实现。有关其内部工作原理的详细信息，请参阅 [***BOTS 论文***](https://arxiv.org/abs/2510.26374) 及其 [***教程***](https://github.com/agentscope-ai/Trinity-RFT/blob/main/examples/bots/README.md)。

## 如何运行

### 步骤 1：前置要求

确保您已按照[指南](https://github.com/agentscope-ai/agentscope-samples/blob/main/tuner/math_agent/README_zh.md#how-to-run)安装了 **AgentScope** 和 **Trinity-RFT**。

### 步骤 2：准备数据集

运行数据准备脚本。确保之后更新 `config_random.yaml` 和 `config_difficulty.yaml` 中的数据集路径。

```bash
python prepare_data.py
```

### 步骤 3：启动 Ray 集群

对于分布式训练，启动 Ray 集群。

```bash
# 单节点
ray start --head
```

### 步骤 4：运行训练

您现在可以运行基线或基于难度的训练实验。

- **使用随机选择器运行基线实验：**

```bash
python main.py --config config_random.yaml
```

- **使用基于难度的选择器运行实验：**
```bash
python main.py --config config_difficulty.yaml
```

## 实验结果

以下结果比较了 `difficulty-based` 选择策略（红线，bots）与标准 `random` 选择策略（黑线，random）的性能。

<div align="center">
  <img src="./training_result.jpg" alt="训练结果图" width="90%"/>
</div>

### 训练奖励曲线

左侧图表显示了训练期间的 rollout 准确率。可以看出，随机策略采样的任务对模型来说似乎很困难，准确率保持在 0.2 以下。相比之下，使用难度选择器会产生更高的平均准确率，表明智能体正在处理更多可以成功解决的任务。

### 在 AIME-24 上的评估

为了比较，我们在 AIME-24 基准上评估了两种选择策略。右侧图表显示，基于难度的方法在性能上表现出更好的上升趋势。
