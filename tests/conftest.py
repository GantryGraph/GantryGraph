from __future__ import annotations

import pytest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage


@pytest.fixture
def mock_llm_one_shot() -> FakeMessagesListChatModel:
    """LLM that emits one tool call then declares completion."""
    return FakeMessagesListChatModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "shell_run",
                        "args": {"command": "echo hi"},
                        "id": "call_1",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(content="Task completed."),
        ]
    )


@pytest.fixture
def mock_llm_done_immediately() -> FakeMessagesListChatModel:
    """LLM that declares completion without using any tools."""
    return FakeMessagesListChatModel(
        responses=[AIMessage(content="I can complete this without tools.")]
    )


@pytest.fixture
def mock_llm_error_then_done() -> FakeMessagesListChatModel:
    """LLM that attempts a tool, gets an error, then recovers and finishes."""
    return FakeMessagesListChatModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "nonexistent_tool",
                        "args": {},
                        "id": "call_err",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(content="I see the tool failed. Task is complete anyway."),
        ]
    )
