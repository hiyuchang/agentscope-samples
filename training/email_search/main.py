# -*- coding: utf-8 -*-
"""Example of training an Email Search agent with Trinity-RFT."""
import os
from typing import Dict
from _email_search_agent import EmailSearchAgent
from _utils import (
    AnswerModel,
    FinalRubric,
    QueryModel,
)
from agentscope import logger
from agentscope.formatter import OpenAIChatFormatter
from agentscope.message import Msg
from agentscope.tuner import (
    TunerChatModel,
    Dataset,
    JudgeOutput,
    WorkflowOutput,
    Algorithm,
    tune,
)


SYSTEM_PROMPT = """You are an email search agent. You are given a user query
and a list of tools you can use to search the user's email. Use the tools to
search the user's emails and find the answer to the user's query. You may take
up to {max_turns} turns to find the answer, so if your first seach doesn't
find the answer, you can try with different keywords.

Always describe what you see and plan your next steps clearly. When taking
actions, explain what you're doing and why. When the answer to the task is
found, call `generate_response` to finish the process. Only call
`generate_response` when answer is found. You should not respond any next steps
in `generate_response`. Complete all steps and then call `generate_response`.

User's email address is {inbox_address}

Today's date is {query_date}

"""


async def run_email_search_agent(
    task: Dict,
    model: TunerChatModel,
    auxiliary_models: Dict[str, TunerChatModel],
) -> WorkflowOutput:  # noqa: PLR0915
    """A workflow function using the Email Search agent to solve tasks.

    Args:
        task (Dict): The task to be solved.
            Should contain fields from QueryModel.
        model (TrinityChatModel): The language model to use.

    Returns:
        WorkflowOutput: The output containing the agent's response.
    """
    assert len(auxiliary_models) > 0, "LLM-as-a-Judge is required"

    # Parse task data
    query = QueryModel.model_validate(task)
    question = task.get("question", task.get("task_desc", ""))

    # Get workflow arguments with defaults
    workflow_args = task.get("workflow_args", {})
    max_turns = int(workflow_args.get("max_turns", 10))

    # Format system prompt
    system_prompt = SYSTEM_PROMPT.format(
        max_turns=max_turns,
        inbox_address=query.inbox_address,
        query_date=query.query_date,
    )

    # Create EmailSearchAgent
    agent = EmailSearchAgent(
        name="email_search_agent",
        sys_prompt=system_prompt,
        model=model,
        formatter=OpenAIChatFormatter(),
        max_iters=max_turns,
    )

    # Reset agent state for a new rollout
    await agent.reset()

    # Run the agent with structured output
    response = await agent.reply(
        msg=Msg("user", question, role="user"),
        structured_model=AnswerModel,
    )

    # Extract answer and sources from response metadata
    answer_and_sources = response.metadata or {}
    if not answer_and_sources:
        # Fallback: try to parse from content
        answer_and_sources = {
            "answer": response.get_text_content() or "",
            "sources": [],
        }

    # Store agent state for judge function
    # We'll pass this through the response metadata
    response_metadata = {
        "answer_and_sources": answer_and_sources,
        "query": query.model_dump(),
        "message_id_list": agent.message_id_list,
        "ever_read_message_ids": agent.ever_read_message_ids,
        # Estimate actual_turns from memory length
        "actual_turns": (
            max(1, (len(agent.memory.content) - 1) // 2)
            if len(agent.memory.content) > 1
            else 1
        ),
    }

    # Update response metadata
    if response.metadata is None:
        response.metadata = {}
    response.metadata.update(response_metadata)

    return WorkflowOutput(
        response=response,
    )


def _calculate_partial_rewards(rubric: FinalRubric) -> float:
    """Calculate partial rewards based on rubric."""
    partial_rewards = 0.0
    partial_rewards += 0.1 if rubric.ever_found_right_email else 0
    partial_rewards += 0.1 if rubric.ever_read_right_email else 0
    partial_rewards += 0.1 if rubric.sources_correct else 0
    return partial_rewards


def _calculate_correct_answer_reward(
    rubric: FinalRubric,
    max_turns: int,
) -> float:
    """Calculate reward for correct answers."""
    reward = 1.0
    reward += 0.3 if rubric.sources_correct else 0
    reward += 0.1 / rubric.num_sources if rubric.num_sources > 0 else 0
    reward += 0.1 * (1 - rubric.num_turns / max_turns)
    return reward


def _initialize_rubric(
    answer: str,
    sources: list[str],
    actual_turns: int,
    query: QueryModel,
    message_id_list: list[str],
    ever_read_message_ids: list[str],
) -> FinalRubric:
    """Initialize and populate rubric with basic information."""
    rubric = FinalRubric()
    rubric.attempted_answer = answer is not None and answer != ""
    rubric.returned_i_dont_know = answer == "I don't know"
    rubric.num_sources = len(sources)
    rubric.num_turns = actual_turns

    if len(query.message_ids) > 0:
        rubric.ever_found_right_email = query.message_ids[0] in message_id_list
        rubric.ever_read_right_email = (
            query.message_ids[0] in ever_read_message_ids
        )
        rubric.sources_correct = query.message_ids[0] in sources
    return rubric


async def email_search_judge(
    task: Dict,
    response: Msg,
    auxiliary_models: Dict[str, TunerChatModel],
) -> JudgeOutput:
    """A judge function to calculate reward based on agent's response.

    Args:
        task (Dict): The task information for the corresponding workflow.
        response (Msg): The response generated by the corresponding workflow.
        auxiliary_models (Dict[str, TunerChatModel]):
            A dictionary of additional chat models available for LLM-as-a-Judge
            usage. The keys are model names, and the values are the
            corresponding TunerChatModel instances.

    Returns:
        JudgeOutput: The reward value assigned by the judge function.
    """
    # Extract metadata from response
    metadata = response.metadata or {}
    answer_and_sources = metadata.get("answer_and_sources", {})
    query_dict = metadata.get("query", {})
    message_id_list = metadata.get("message_id_list", [])
    ever_read_message_ids = metadata.get("ever_read_message_ids", [])
    actual_turns = metadata.get("actual_turns", 0)

    # Parse query model
    if not query_dict:
        query_dict = task
    query = QueryModel.model_validate(query_dict)

    # Get arguments
    workflow_args = task.get("workflow_args", {})
    max_turns = int(workflow_args.get("max_turns", 10))

    # Extract answer and sources
    try:
        answer = answer_and_sources.get("answer", None)
        sources = answer_and_sources.get("sources", [])
    except Exception:
        result = {"accuracy": 0.0, "format": -1.0}
        return JudgeOutput(
            reward=sum(result.values()),
            metrics=result,
        )

    if answer is None:
        result = {"accuracy": 0.0, "format": -1.0}
        return JudgeOutput(
            reward=sum(result.values()),
            metrics=result,
        )

    # Initialize rubric
    rubric = _initialize_rubric(
        answer,
        sources,
        actual_turns,
        query,
        message_id_list,
        ever_read_message_ids,
    )

    # Judge correctness using LLM-as-a-Judge
    try:
        judge_model = (
            auxiliary_models.get("judge") or list(auxiliary_models.values())[0]
            if auxiliary_models
            else None
        )
        judge_response = await judge_correctness(
            answer,
            query,
            judge_model,
        )
        rubric.answer_correct = judge_response
    except Exception as e:
        logger.error("Error judging correctness: %s", e)
        rubric.answer_correct = False

    # Calculate rewards
    partial_rewards = _calculate_partial_rewards(rubric)

    if rubric.attempted_answer and not rubric.answer_correct:
        result = {"accuracy": -1.0, "format": partial_rewards}
    elif rubric.returned_i_dont_know or rubric.ran_out_of_turns:
        result = {"accuracy": 0.0, "format": partial_rewards}
    elif rubric.answer_correct:
        reward = _calculate_correct_answer_reward(rubric, max_turns)
        result = {"accuracy": 1.0, "format": reward}
    else:
        result = {"accuracy": 0.0, "format": 0.0}

    metrics = result.copy()
    metrics.update({"actual_turns": actual_turns})

    return JudgeOutput(
        reward=sum(result.values()),
        metrics=metrics,
    )


# LLM-as-a-judge


async def judge_correctness(
    answer: str,
    query: QueryModel,
    judge: TunerChatModel,
) -> bool:
    """Use an LLM to decide whether *answer* matches *query.answer*.

    Returns a boolean *accept* flag used for scoring.
    """

    system_prompt = """You are given a question, the reference answer
(labelled **Reference answer**), and an answer generated by an AI assistant
(labelled **AI answer**).

Follow these steps to decide whether the AI answer should be accepted:
1. Identify EXACTLY what information the **question** is asking for
   (e.g. who, what, when, where, why, how, quantity, etc.).
2. From the **Reference answer**, extract ONLY the facts that are required
   to directly satisfy the information need identified in step 1. Treat all
   other facts as non-essential context.
3. Verify that every essential fact from step 2 appears in the **AI answer**
   with the same meaning. Differences in wording, order, or additional
   non-conflicting details are allowed.
4. If any essential fact is missing or contradicted in the **AI answer**,
   then *accept* must be **false**. Otherwise *accept* must be **true**.

Important: Do NOT penalise the **AI answer** for omitting non-essential
facts that appear in the **Reference answer**. The answer should only be
rejected for errors or omissions in the information explicitly requested by
the question.

Return your judgement **accept** from **true** and **false**. Do not return
any other text or formatting.
"""
    prompt = (
        f"Question: {query.question}\n"
        f"Reference answer: {query.answer}\n"
        f"AI answer: {answer}"
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]
    chat_response = await judge(messages)

    # Extract text content from ChatResponse
    result_parts = []
    for block in chat_response.content:
        if isinstance(block, dict) and block.get("type") == "text":
            result_parts.append(str(block.get("text", "")))
    result = "".join(result_parts)
    logger.info("LLM judge response: %s", result)

    return "true" in result.lower()


# End of LLM-as-a-judge


if __name__ == "__main__":
    config_path = os.path.join(
        os.path.dirname(__file__),
        "config.yaml",
    )
    dataset = Dataset(
        path="/path/to/enron_emails_dataset",
        split="train",
    )
    tuner_model = TunerChatModel(
        model_path="Qwen/Qwen3-4B-Instruct-2507",
        max_model_len=20480,
        max_tokens=4096,
        inference_engine_num=4,
        reasoning_parser=None,
    )
    aux_models = {
        "judge": TunerChatModel(
            model_path="Qwen/Qwen3-30B-A3B-Instruct-2507",
            max_model_len=2500,
            max_tokens=2048,
            inference_engine_num=1,
            tensor_parallel_size=2,
            tool_call_parser=None,
            reasoning_parser=None,
        ),
    }
    algorithm = Algorithm(
        algorithm_type="multi_step_grpo",
        group_size=8,
        batch_size=64,
        learning_rate=1e-6,
    )
    tune(
        workflow_func=run_email_search_agent,
        judge_func=email_search_judge,
        train_dataset=dataset,
        model=tuner_model,
        auxiliary_models=aux_models,
        algorithm=algorithm,
        config_path=config_path,
    )
