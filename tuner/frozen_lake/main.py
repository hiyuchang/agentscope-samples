# -*- coding: utf-8 -*-
"""Example of training a FrozenLake agent with Trinity-RFT."""
import os
from typing import Dict
from _frozenlake_agent import FrozenLakeAgent
from _frozenlake_env import FrozenLakeEnv
from agentscope.message import Msg
from agentscope.tuner import (
    tune,
    WorkflowOutput,
    Dataset,
    TunerChatModel,
    Algorithm,
)


async def run_frozen_lake(
    task: Dict,
    model: TunerChatModel,
    auxiliary_models: Dict[str, TunerChatModel],
) -> WorkflowOutput:
    """A workflow function using the FrozenLake agent to solve tasks.

    Args:
        task (Dict): The task to be solved, containing environment parameters
            like size, p, seed, is_slippery, etc.
        model (TunerChatModel): The language model to use.

    Returns:
        WorkflowOutput: The workflow output containing the reward, response and
            metrics.
    """

    assert len(auxiliary_models) == 0, "No auxiliary models are needed"

    # Extract workflow arguments from task or use defaults
    workflow_args = task.get("workflow_args", {})
    if not workflow_args:
        workflow_args = task

    env_max_steps = workflow_args.get("env_max_steps", 8)
    agent_max_steps = workflow_args.get("agent_max_steps", 10)
    is_slippery = workflow_args.get("is_slippery", False)
    desc = workflow_args.get("desc", None)

    # Extract task-specific arguments (for environment generation)
    size = task.get("size", 8)
    p = task.get("p", 0.8)
    seed = task.get("seed", 42)

    # Initialize agent and environment
    agent = FrozenLakeAgent(model=model, max_steps=agent_max_steps)
    env = FrozenLakeEnv(
        max_steps=env_max_steps,
        desc=desc,
        is_slippery=is_slippery,
        size=size,
        p=p,
        seed=seed,
    )

    # Reset environment with task parameters
    observation, _ = env.reset(task)
    observation_str = str(observation)
    rewards = []
    step_count = 0
    done = False
    terminate_reason = None

    # Run agent-environment interaction loop
    for _ in range(agent_max_steps):
        step_count += 1
        try:
            # get prompt
            prompt = agent.get_prompt(observation_str)

            response = await agent.reply(msg=Msg("user", prompt, role="user"))

            # record action and observation
            action = agent.get_action(response)
            agent.update_state(action=action, observation=observation_str)

        except Exception as e:
            terminate_reason = f"agent_error: {str(e)}"
            break

        # environment step
        observation, reward, done, _ = env.step(action)
        observation_str = str(observation)
        rewards.append(reward)

        if done:
            terminate_reason = "success" if env.success() else "hole"
            break

    if terminate_reason is None:
        terminate_reason = "max_steps_reached"

    final_reward = sum(rewards)
    final_observation = observation_str

    # Create response message with environment information
    response_content = (
        f"Final observation:\n{final_observation}\n"
        f"Total reward: {final_reward}\n"
        f"Steps taken: {step_count}\n"
        f"Terminate reason: {terminate_reason}"
    )

    final_response = Msg("assistant", response_content, role="assistant")

    return WorkflowOutput(
        reward=final_reward,
        response=final_response,
        metrics={
            "env_steps": float(step_count),
            "env_done": float(done),
        },
    )


if __name__ == "__main__":
    dataset = Dataset(
        path="/path/to/frozenlake",
        split="train",
    )
    tuner_model = TunerChatModel(
        model_path="Qwen/Qwen2.5-3B-Instruct",
        max_model_len=25600,
        max_tokens=2048,
        inference_engine_num=6,
        reasoning_parser=None,
    )
    algorithm = Algorithm(
        algorithm_type="multi_step_grpo",
        group_size=16,
        batch_size=32,
        learning_rate=1e-6,
    )
    config_path = os.path.join(
        os.path.dirname(__file__),
        "config.yaml",
    )  # define some default parameters
    tune(
        workflow_func=run_frozen_lake,
        model=tuner_model,
        train_dataset=dataset,
        algorithm=algorithm,
        config_path=config_path,
    )
