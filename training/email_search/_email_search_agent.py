# -*- coding: utf-8 -*-
"""Adapted from Trinity-RFT"""
import json
import traceback
from dataclasses import asdict
from datetime import datetime, timedelta
from typing import Any
from _utils import (  # pylint: disable=E0611
    read_email_tool,
    search_emails_tool,
)
from agentscope import logger
from agentscope.agent import ReActAgent
from agentscope.message import TextBlock
from agentscope.tool import Toolkit, ToolResponse


def pre_reasoning_hook(_self: Any, _kwargs: Any) -> dict[str, Any] | None:
    """Pre-reasoning hook to remove tool_choice from kwargs."""
    _kwargs.pop("tool_choice", None)
    return _kwargs


class EmailSearchAgent(ReActAgent):
    """
    A customized ReAct agent with pre-defined tools for
    email search and reading.
    Ref: https://github.com/OpenPipe/ART
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.message_id_list = (
            []
        )  # List to store message IDs found during search
        self.ever_read_message_ids = (
            []
        )  # List to store message IDs that have been read
        toolkit = Toolkit()
        toolkit.register_tool_function(self.search_emails)
        toolkit.register_tool_function(self.read_email)
        super().__init__(*args, toolkit=toolkit, **kwargs)

        self.register_instance_hook(
            "pre_reasoning",
            "tool_choice_hook",
            pre_reasoning_hook,
        )

    async def reset(self) -> None:
        """Reset agent state for a new rollout/episode."""
        self.message_id_list.clear()
        self.ever_read_message_ids.clear()
        await self.memory.clear()

    def search_emails(
        self,
        inbox_address: str,
        query_date: str,
        keywords: list[str],
        **_kwargs: Any,
    ) -> ToolResponse:
        """
        Search the user's email inbox for emails that match the given keywords.

        Args:
            inbox_address: The user's email address.
            query_date: The date of the query in 'YYYY-MM-DD' format.
            keywords: Keywords to search for in the user's email inbox.

        Returns:
            ToolResponse:
                A ToolResponse object containing a list of TextBlock objects
                in the `content` field. On success, the text field of the
                TextBlock contains a JSON string representing a list of email
                summaries (e.g., message_id, snippet) matching the search
                criteria. Each email summary is converted to a dictionary via
                `asdict`. On failure, the text indicates an error message.
        """

        try:
            next_day = (
                datetime.strptime(query_date, "%Y-%m-%d") + timedelta(days=1)
            ).strftime(
                "%Y-%m-%d",
            )
            res = search_emails_tool(
                inbox=inbox_address,
                sent_before=next_day,
                keywords=keywords,
            )

            self.message_id_list.extend([r.message_id for r in res])

            return ToolResponse(
                content=[
                    TextBlock(
                        type="text",
                        text=json.dumps([asdict(r) for r in res]),
                    ),
                ],
            )
        except Exception as e:
            logger.info(
                "Error in search_emails: %s, traceback: %s",
                e,
                traceback.format_exc(),
            )
            return ToolResponse(
                content=[
                    TextBlock(
                        type="text",
                        text=(
                            f"Error: Failed to search emails.\n"
                            f"Error message: {e}"
                        ),
                    ),
                ],
            )

    def read_email(self, message_id: str, **_kwargs: Any) -> ToolResponse:
        """
        Read the content of an email from the user's email inbox.
        Returns the email content.

        Args:
            message_id (str): The unique identifier of the email to read.

        Returns:
            ToolResponse:
                A ToolResponse object containing the email content or an
                error message if the email is not found.
        """

        try:
            email_content = read_email_tool(message_id)

            self.ever_read_message_ids.append(message_id)

            if email_content is None:
                return ToolResponse(
                    content=[
                        TextBlock(
                            type="text",
                            text=(
                                f"Error: Email (message_id = {message_id}) "
                                f"not found."
                            ),
                        ),
                    ],
                )
            return ToolResponse(
                content=[
                    TextBlock(
                        type="text",
                        text=json.dumps(email_content.model_dump()),
                    ),
                ],
            )
        except Exception as e:
            logger.info(
                "Error in read_email: %s, traceback: %s",
                e,
                traceback.format_exc(),
            )
            return ToolResponse(
                content=[
                    TextBlock(
                        type="text",
                        text=(
                            f"Error: Failed to read email.\n"
                            f"Error message: {e}"
                        ),
                    ),
                ],
            )
