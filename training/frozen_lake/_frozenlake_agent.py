# -*- coding: utf-8 -*-
"""Adapted from Trinity-RFT"""
import re
from _utils import SYSTEM_PROMPT, FrozenLakeAction  # pylint: disable=E0611
from agentscope.agent import ReActAgent
from agentscope.formatter import OpenAIChatFormatter
from agentscope.message import Msg
from agentscope.model import OpenAIChatModel


INVALID_ACTION = "still"
VALID_ACTIONS = {
    "left": 1,
    "down": 2,
    "right": 3,
    "up": 4,
}


class FrozenLakeAgent(ReActAgent):
    """Agent for FrozenLake environment."""

    def __init__(self, model: OpenAIChatModel, max_steps: int = 20):
        super().__init__(
            name="frozenlake_agent",
            model=model,
            sys_prompt=SYSTEM_PROMPT,
            formatter=OpenAIChatFormatter(),
            max_iters=1,
        )
        self.response_structure = FrozenLakeAction
        self.current_step = 0
        self.last_action = None
        self.last_observation = None
        self.max_steps = max_steps

    def get_prompt(self, observation: str) -> str:
        """Get prompt for the agent based on current observation."""
        prompt = (
            f"Current Observation ({self.current_step}): \n"
            + observation
            + "\n"
            + (
                "You have not achieved the goal, P has not reached G yet. "
                "Please give the next action."
            )
        )
        if self.current_step > 0 and self.last_action is not None:
            if self.last_observation == observation:
                prompt += (
                    "\nYour last response is invalid. "
                    "Your position didn't change at all. "
                    "You may need to recheck your thinking process, "
                    "action outputted, and the format of response. "
                    "Remember, you should only output the NEXT ACTION "
                    "at each iteration in the ``` ```. "
                    "For example, if you want to move up, "
                    "you should output ```Up```."
                )

        if (
            self.max_steps is not None
            and self.max_steps - self.current_step > 0
        ):
            remaining = self.max_steps - self.current_step
            prompt += (
                f"\nThe maximum number of steps remaining is {remaining}."
            )

        return prompt

    def get_action(self, msg: Msg) -> str:
        """Extract action from agent response message."""
        response: str = (
            msg.content
            if isinstance(msg.content, str)
            else msg.content[0].get("text")
        )
        action = INVALID_ACTION

        matches = re.findall(r"```(.*?)```", response, re.DOTALL)

        if matches:
            last_match_content = matches[-1].strip()
            action = last_match_content.lower()
            if action not in VALID_ACTIONS:
                action = INVALID_ACTION

        return action

    def update_state(self, action: str, observation: str) -> None:
        """Update agent state with action and observation."""
        self.last_action = action
        self.last_observation = observation
        self.current_step += 1

    async def reset(self) -> None:
        """Reset agent state for a new episode."""
        self.current_step = 0
        self.last_action = None
        self.last_observation = None
        await self.memory.clear()
