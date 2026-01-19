# 使用 AgentScope-Tuner 训练 Learn2Ask

本指南演示了如何使用来自 [Grounded in Reality: Learning and Deploying Proactive LLM from Offline Logs](https://arxiv.org/abs/2510.25441) 的 **Learn2Ask** 方法训练主动式 LLM。

---

## 任务设置

在此示例中，给定用户的主诉，医疗助手智能体主动提出有针对性的问题，以收集足够的症状信息，从而全面评估用户的健康状况。查询过程应该高效：智能体必须优化问题质量，并在收集的信息足以进行后续临床评估或决策时立即终止访谈。
这里我们使用 `ReActAgent` 来完成此任务，不需要工具。

---

## 硬件要求

- **使用 GPU 训练**：至少需要 **8 个 H20 GPU**（或同等配置）。
- **不使用 GPU 训练**：您可以使用 **[Tinker](https://thinkingmachines.ai/tinker/)**，无需任何 GPU。

> 💡 所有代码和配置文件位于：
> `tuner/learn_to_ask/`

关键文件：
- 工作流和训练：`tuner/learn_to_ask/main.py`
- 提示词：`tuner/learn_to_ask/prompt.py`
- 训练配置：`tuner/learn_to_ask/config.yaml`
- 数据准备脚本：`tuner/learn_to_ask/data_prepare/`

---

## 数据集准备

> [!NOTE]
> 在此示例中，我们直接使用开源数据集进行训练。然而，在实践中，您通常需要先收集已部署智能体与用户之间的交互日志。在过滤这些原始日志以整理高质量数据集后，您可以遵循相同的流程，使用 AgentScope-Tuner 增强智能体的主动能力。祝调优愉快！

### 1.1 下载数据集
下载 **[RealMedConv](https://huggingface.co/datasets/datajuicer/RealMedConv)** 数据集（`.jsonl` 格式）。
您可以使用以下 Python 脚本下载数据集：

```python
from huggingface_hub import snapshot_download

# 下载到本地目录，例如 `./tuner/learn_to_ask/data`
local_dir = "./tuner/learn_to_ask/data"
snapshot_download(
    repo_id="datajuicer/RealMedConv",
    repo_type="dataset",
    local_dir=local_dir,
)
```

`train_origin.jsonl`（或 `test_origin.jsonl`）中的每一行代表一个完整的医患对话日志，如下所示：

```json
{
  "session_id": 35310,
  "diagn": "Upper Respiratory Tract Infection",
  "messages": [
    {"role": "user", "content": "Sore throat, phlegm, red eyes, cough, hoarse voice"},
    {"role": "user", "content": "I took Amoxicillin"},
    ...
    {"role": "assistant", "content": "<med_search>"}
  ]
}
```

### 1.2 预处理数据
您需要将原始对话日志转换为训练样本。这涉及两个步骤：

#### 🔹 步骤 A：分割对话并提取标签
将每个对话分割为**context–future pairs**，并从后续内容中提取真实症状信息（`info_truth`）。

```bash
python tuner/learn_to_ask/data_prepare/1_info_extract_pipeline.py \
  --input_file /path/to/RealMedConv/train.jsonl \
  --output_file tuner/learn_to_ask/data_raw/train_processed.jsonl \
  --model_path Qwen/Qwen2.5-32B-Instruct
```

#### 🔹 步骤 B：构建最终训练数据集
将处理后的样本转换为用于训练/测试的最终格式。

```bash
python tuner/learn_to_ask/data_prepare/2_build_dataset.py \
  --input_file tuner/learn_to_ask/data_raw/train_processed.jsonl \
  --output_file tuner/learn_to_ask/data/train.jsonl
```

---

### 工作原理：Context–Future pairsSegmentation

对于对话中的每一轮，我们创建一个样本，包含：
- `messages`：到该点为止的**已观察对话历史**（context）。
- `remaining_chat`：该点之后发生的**所有内容**（future）。
- 唯一 ID：`cid = {session_id}_{turn_index}`

示例输出：
```json
{
  "cid": "35310_7",
  "session_id": "35310",
  "diagn": "Upper Respiratory Tract Infection",
  "messages": [ ... up to turn 7 ... ],
  "remaining_chat": [ ... all future messages ... ]
}
```

### 提取真实标签

从 `remaining_chat` 中，我们自动推导出两个关键标签：
- `decision_truth`：助手应该继续提问（`"continue"`）还是停止（`"stop"`）？
- `info_truth`：后续提到的结构化症状列表（用于在训练期间计算奖励信号）。

示例：
```json
{
  "decision_truth": "continue",
  "info_truth": "Symptom: sore throat, Symptom quality: thick discharge, Symptom quality: yellowish discharge, ..."
}
```

这些标签在训练期间为奖励函数 $R_a$（动作准确性）和 $R_s$（症状覆盖率）提供支持。

---

## 代码实现

### 智能体工作流

工作流函数 `run_react_agent` 实现了 `ReActAgent` 的工作方式。

```python
async def run_react_agent(
    task: Dict,
    model: OpenAIChatModel,
    auxiliary_models: Dict[str, OpenAIChatModel],
) -> WorkflowOutput:
    assert (
        len(auxiliary_models) == 1
    ), "Please provide only one `auxiliary_models` for `learn_to_ask`."

    import importlib

    spec = importlib.util.spec_from_file_location(
        "prompt",
        os.path.join(os.path.dirname(__file__), "prompt.py"),
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if TRAIN_MODE == "Ra":
        sys_prompt = module.rollout_prompt_med_Ra
    else:
        sys_prompt = module.rollout_prompt_med

    agent = ReActAgent(
        name="react_agent",
        sys_prompt=sys_prompt,
        model=model,
        formatter=OpenAIChatFormatter(),
        toolkit=None,
        memory=InMemoryMemory(),
        max_iters=1,
    )
    messages = format_messages(task["messages"])
    response = await agent.reply(
        [
            Msg(name=x["role"], content=x["content"], role=x["role"])
            for x in messages
        ],
    )
    return WorkflowOutput(
        response=response,
    )
```

### 评判函数

评判函数 `learn2ask_judge` 使用 LLM-as-a-Judge 实现奖励计算：

```python
async def learn2ask_judge(
    task: Dict,
    response: Msg,
    auxiliary_models: Dict[str, OpenAIChatModel],
) -> JudgeOutput:
    response_text = response.get_text_content()
    action_truth = task.get("decision_truth", "continue")
    action_response = "stop" if "<stop />" in response_text else "continue"
    
    # 计算动作准确性分数
    action_score = 1.0 if action_truth == action_response else 0.0
    
    # 计算格式和内容分数
    if action_score == 1.0 and action_truth == "continue":
        # 使用 LLM-as-a-Judge 评估问题质量
        score_dict = await llm_reward(task, response_text, auxiliary_models)
        format_score = float(score_dict.get("format_score", 0.0))
        content_score = float(score_dict.get("content_score", 0.0))
    elif action_score == 1.0:  # stop 动作
        content_score, format_score = 1.0, (1.0 if response_text == "<stop />" else 0.0)
    else:
        format_score = content_score = 0.0
    
    # 根据训练模式组合最终奖励
    if TRAIN_MODE == "Ra+Rs":  # 默认：动作 + 症状奖励
        final_reward = action_score * (1 + 2 * content_score) + format_score
    elif TRAIN_MODE == "Ra":  # 仅动作奖励
        final_reward = 2 * content_score + format_score
    else:  # 仅症状奖励
        final_reward = action_score * 3 + format_score
    
    return JudgeOutput(reward=final_reward, metrics={"reward": final_reward})
```

此奖励函数考虑：
- 动作准确性：`action_score`
- 问题质量（症状覆盖率）：`content_score`
- 格式分数：`format_score`

有关实现细节，请参阅 [main.py](./main.py)。

---

## 配置和训练模型

### 选项 A：编辑 Python 脚本（简单）
打开 `tuner/learn_to_ask/main.py` 并调整设置：

```python
if __name__ == "__main__":
    train_mode = "Ra+Rs"     # 同时使用动作和症状奖励
    fusion_mode = "default"  # 如何组合奖励
    dataset = DatasetConfig(path="tuner/learn_to_ask/data", split="train")

    tuner_model = OpenAIChatModel(
        model_path="Qwen/Qwen2.5-7B-Instruct",
        max_model_len=8192,
        tensor_parallel_size=1,  # 根据您的 GPU 设置调整
        ...
    )

    auxiliary_models = {
        AUXILIARY_MODEL_NAME: OpenAIChatModel(
            model_path="Qwen/Qwen2.5-32B-Instruct",  # 用于评估的更大模型
            tensor_parallel_size=2,
            ...
        )
    }

    algorithm = AlgorithmConfig(
        algorithm_type="grpo",
        learning_rate=5e-7,
        batch_size=64,
    )

    tune(...)  # 开始训练
```

### 选项 B：使用 YAML 配置（高级）
编辑 `tuner/learn_to_ask/config.yaml` 以获得更多控制。

#### 🌐 没有 GPU？使用 Tinker！
如果您没有 GPU，可以通过设置启用 **Tinker 后端**：

```yaml
model:
  tinker:
    enable: true  # ← 将此设置为 true
```

此外，请确保更新 `tuner/learn_to_ask/main.py` 中的 `model_path`，使其指向与 Tinker 兼容的模型。

> 🔗 了解更多关于 Tinker 后端： [Tinker 后端文档](https://agentscope-ai.github.io/Trinity-RFT/zh/main/tutorial/example_tinker_backend.html)

### 启动训练
```bash
python tuner/learn_to_ask/main.py
```

---

## 评估

使用**rollout 和评估流程**：
1. 在测试集上生成响应。
2. 使用强大的评估模型（`Qwen2.5-32B-Instruct`）对它们进行评分。

运行评估：
```bash
python tuner/learn_to_ask/data_prepare/3_rollout_then_evaluate.py \
  --eval_model_path path/to/your/trained/model \
  --grader_model_path Qwen/Qwen2.5-32B-Instruct \
  --test_file_path tuner/learn_to_ask/data/test.jsonl \
  --rollout_file_path path/to/rollout.jsonl \
  --eval_file_path path/to/output.jsonl
```

> ⚠️ **注意**：您的训练模型必须首先转换为 **Hugging Face 格式**。
> 请参阅：[转换 FSDP 检查点指南](https://agentscope-ai.github.io/Trinity-RFT/zh/main/tutorial/faq.html)

---

## 实验结果

我们比较了三种方法：
- **基础模型**：`Qwen2.5-7B-Instruct`（无微调）
- **Trinity**：直接响应生成
- **AgentScope-Tuner (Learn2Ask)**：使用 ReAct 智能体进行主动提问

| 指标                               | 基础模型 | Trinity | AgentScope-Tuner (Learn2Ask) |
|------------------------------------|---------:|--------:|--------------------:|
| 平均继续内容                        |    0.436 |   0.496 |               0.509 |
| 胜率（继续内容）                    |    0.122 |   0.246 |               0.224 |
| 平均继续决策准确性                  |    0.963 |   0.909 |               0.922 |
| 平均停止决策准确性                  |    0.024 |   0.927 |               0.909 |
| **总决策准确性**                    |    0.467 |   0.918 |               0.915 |
| **响应格式质量**                    |    0.376 |   0.713 |               0.882 |
| **总奖励**                          |    1.281 |   3.078 |               3.237 |

<div align="center">
  <img src="./learn2ask.png" alt="训练曲线" width="90%"/>
</div>

> ✅ **关键洞察**：Learn2Ask (AgentScope-Tuner) 通过教模型**何时提问和提问什么**，实现了最高的整体性能——使其真正具有主动性。

### 具体示例

下面显示了智能体在评估数据上的行为示例：

```json
{
  "cid": "206_1",
  "session_id": 206,
  "diagn": "Common cold",
  "messages": [
    {
      "role": "assistant",
      "content": "Hello, currently there are many people consulting, please wait patiently."
    },
    {
      "role": "user",
      "content": "Hello, this morning around 10 o'clock, I suddenly felt a bit feverish and had a heavy head. I took my temperature and it was 38.5°C. I took one tablet of paracetamol at 1 o'clock, which helped a lot. By 6 or 7 in the evening, I felt cold and feverish again, so I took another tablet of paracetamol. I'm not sure if it's a side effect of the medication, but I feel a bit nauseous and my stomach keeps producing gas. Now, I feel cold again, have a stuffy nose with clear runny nose, an itchy throat with coughing, a heavy head, and can't sleep."
    }
  ],
  "decision_truth": "continue",
  "info_truth": "symptom: needle-like sensation, symptom location: ear, symptom nature: sharp pain, symptom severity: occurred twice",
  "rollouts": [
    "Do you also experience body aches along with your symptoms?  \nA) Yes B) No C) Not Sure",
    "Do you experience these symptoms every time after taking paracetamol? (A) Yes, definitely. (B) Sometimes. (C) No, never before.",
    "Do you have a history of allergies? A) Yes B) No C) Not Sure"
  ],
  "grades": [
    {
      "think": "The doctor's last message asks about body aches using multiple choice options rather than a single direct open-ended question, which deviates from the typical questioning style indicated by the grading policy. However, since it is framed in the form of \"Do you...\", it counts as one question despite being presented with choices. Thus, it satisfies the criteria for 'contains exactly one question'. For the content score, the query does not pertain directly to the details listed in the reference information such as the specific symptom ('needle-like' sharp pain in the ear that has occurred twice), thereby making it highly relevant yet indirectly related due to its focus on comprehensive assessment of flu-related discomforts without addressing specifics noted in the patient's primary concern documented earlier.",
      "action_score": 1,
      "format_score": 1.0,
      "content_score": 0.5
    },
    {
      "think": "The doctor's last message includes just one multiple-choice question regarding whether the patient experiences those mentioned symptoms each time they take paracetamol. This does relate highly to understanding possible drug-related symptoms; however, none of them aligns perfectly with \"needle-like\" sensations occurring specifically in ears according to the reference information given.",
      "action_score": 1,
      "format_score": 1.0,
      "content_score": 0.5
    },
    {
      "think": "The doctor's last statement does contain just one question pertaining to allergy history, which is highly relevant when trying to diagnose symptoms such as those described by the patient (fever, nausea). However, none of these concerns specifically relate back to the reference information detailing \"needle-like sensation\", \"sharp pain\" related to the ears occurring twice. Therefore, while highly pertinent medically, they do not pertain to the exact points outlined in the Ref Info section about the patient experience according to that specific prompt context.",
      "action_score": 1,
      "format_score": 1.0,
      "content_score": 0.5
    }
  ]
}
```

---

## 📚 引用

如果您使用此代码或框架，请引用我们的工作：

```bibtex
@misc{learn2ask,
      title={Grounded in Reality: Learning and Deploying Proactive LLM from Offline Logs},
      author={Fei Wei and Daoyuan Chen and Ce Wang and Yilun Huang and Yushuo Chen and Xuchen Pan and Yaliang Li and Bolin Ding},
      year={2025},
      eprint={2510.25441},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2510.25441}
}
```
