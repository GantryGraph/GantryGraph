"""E2E: full GantryEngine loop with mock LLM + REAL tools on the demo-app.

This is the most important stress test: it simulates exactly what a real
LLM agent would do — call shell_run to get test output, call file_read to
see the source, call file_write to fix bugs, call shell_run again to verify.

Uses FakeMessagesListChatModel with pre-scripted responses so the test is
deterministic and doesn't require an API key.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pytest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage

from gantrygraph import GantryEngine
from gantrygraph.actions.filesystem import FileSystemTools
from gantrygraph.actions.shell import ShellTool
from gantrygraph.core.events import GantryEvent

_PY = sys.executable
DEMO_APP = Path(__file__).parent.parent.parent / "demo-app"

FIXED_UTILS = '''\
"""Utility functions — all bugs fixed."""


def safe_divide(a: float, b: float) -> float:
    """Divide a by b. Returns 0.0 if b is zero."""
    if b == 0.0:
        return 0.0
    return a / b


def is_palindrome(text: str) -> bool:
    """Return True if text reads the same forwards and backwards (case-insensitive)."""
    lower = text.lower()
    return lower == lower[::-1]


def word_count(text: str) -> int:
    """Count words in a string, ignoring extra whitespace."""
    words = text.split()
    return len([w for w in words if w])


def celsius_to_fahrenheit(c: float) -> float:
    """Convert Celsius to Fahrenheit."""
    return c * 9 / 5 + 32


def fahrenheit_to_celsius(f: float) -> float:
    """Convert Fahrenheit to Celsius."""
    return (f - 32) * 5 / 9
'''


@pytest.fixture
def sandbox(tmp_path: Path) -> Path:
    dest = tmp_path / "app"
    shutil.copytree(DEMO_APP, dest)
    return dest


def _scripted_llm(sandbox: Path) -> FakeMessagesListChatModel:
    """Mock LLM that reproduces a realistic agent decision sequence."""
    return FakeMessagesListChatModel(
        responses=[
            # Step 1: run tests to see what's broken
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "shell_run",
                        "args": {"command": f"{_PY} -m pytest test_utils.py -v --tb=short"},
                        "id": "c1",
                        "type": "tool_call",
                    }
                ],
            ),
            # Step 2: read the source file
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "file_read",
                        "args": {"path": "utils.py"},
                        "id": "c2",
                        "type": "tool_call",
                    }
                ],
            ),
            # Step 3: write the fully fixed source
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "file_write",
                        "args": {"path": "utils.py", "content": FIXED_UTILS},
                        "id": "c3",
                        "type": "tool_call",
                    }
                ],
            ),
            # Step 4: verify all tests pass
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "shell_run",
                        "args": {"command": f"{_PY} -m pytest test_utils.py -v"},
                        "id": "c4",
                        "type": "tool_call",
                    }
                ],
            ),
            # Step 5: declare completion
            AIMessage(
                content=(
                    "All 3 bugs have been fixed:\n"
                    "1. safe_divide: added zero-guard\n"
                    "2. is_palindrome: added .lower() for case-insensitive comparison\n"
                    "3. word_count: changed split(' ') to split() for whitespace handling\n"
                    "All 14 tests now pass."
                )
            ),
        ]
    )


# ── Core engine loop test ─────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_engine_loop_detects_and_fixes_bugs(sandbox: Path) -> None:
    """Full loop: detect failing tests → read code → fix → verify → done."""
    events: list[GantryEvent] = []

    agent = GantryEngine(
        llm=_scripted_llm(sandbox),
        tools=[
            FileSystemTools(workspace=sandbox),
            ShellTool(workspace=sandbox, allowed_commands=[_PY], timeout=30.0),
        ],
        on_event=lambda e: events.append(e),
        max_steps=10,
    )

    result = await agent.arun("Find and fix all failing tests in this Python project.")

    # The final answer should mention the fixes
    assert "fixed" in result.lower() or "pass" in result.lower()

    # After the agent ran, the source file should actually be fixed
    fixed_content = (sandbox / "utils.py").read_text()
    assert "b == 0.0" in fixed_content
    assert "text.lower()" in fixed_content
    assert "text.split()" in fixed_content

    # The tests should actually pass now
    import subprocess

    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "test_utils.py", "-v"],
        cwd=str(sandbox),
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert "14 passed" in proc.stdout, f"Tests still failing:\n{proc.stdout}"


@pytest.mark.asyncio
async def test_engine_emits_correct_event_sequence(sandbox: Path) -> None:
    """Events emitted should follow: observe → think → act × N → done."""
    events: list[GantryEvent] = []

    agent = GantryEngine(
        llm=_scripted_llm(sandbox),
        tools=[
            FileSystemTools(workspace=sandbox),
            ShellTool(workspace=sandbox, allowed_commands=[_PY], timeout=30.0),
        ],
        on_event=lambda e: events.append(e),
        max_steps=10,
    )
    await agent.arun("Fix the bugs.")

    event_types = [e.event_type for e in events]
    # Must start with observe → think
    assert event_types[0] == "observe"
    assert event_types[1] == "think"
    # Must end with the last think (final answer)
    assert "observe" in event_types
    assert "think" in event_types
    assert "act" in event_types
    # 4 tool calls → 4 act events
    act_count = event_types.count("act")
    assert act_count == 4


@pytest.mark.asyncio
async def test_engine_loop_self_corrects_on_bad_command(sandbox: Path) -> None:
    """If a tool returns an error, the engine continues (self-correction)."""
    llm = FakeMessagesListChatModel(
        responses=[
            # First: run a nonexistent command
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "shell_run",
                        "args": {"command": "nonexistent_tool_xyz"},
                        "id": "c1",
                        "type": "tool_call",
                    }
                ],
            ),
            # Then: recover and run a valid command
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "shell_run",
                        "args": {"command": f"{_PY} -m pytest test_utils.py --tb=no -q"},
                        "id": "c2",
                        "type": "tool_call",
                    }
                ],
            ),
            # Done
            AIMessage(content="Recovered from error and ran tests successfully."),
        ]
    )

    agent = GantryEngine(
        llm=llm,
        tools=[ShellTool(workspace=sandbox, timeout=30.0)],
        max_steps=5,
    )
    result = await agent.arun("Run the tests.")
    # Should complete without crashing
    assert isinstance(result, str)
    assert len(result) > 0


@pytest.mark.asyncio
async def test_engine_guardrail_blocks_rm_rf(sandbox: Path) -> None:
    """GuardrailPolicy prevents destructive shell commands."""
    denied_tools: list[str] = []

    def strict_approval(tool_name: str, args: dict) -> bool:
        if tool_name == "shell_run":
            cmd = args.get("command", "")
            if "rm" in cmd or "del " in cmd:
                denied_tools.append(cmd)
                return False
        return True

    llm = FakeMessagesListChatModel(
        responses=[
            # Try a destructive command
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "shell_run",
                        "args": {"command": "rm -rf utils.py"},
                        "id": "c1",
                        "type": "tool_call",
                    }
                ],
            ),
            # Then a safe command
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "shell_run",
                        "args": {"command": f"{_PY} -m pytest test_utils.py --tb=no -q"},
                        "id": "c2",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(content="Completed safely."),
        ]
    )

    agent = GantryEngine(
        llm=llm,
        tools=[ShellTool(workspace=sandbox, timeout=30.0)],
        approval_callback=strict_approval,
        max_steps=5,
    )
    await agent.arun("Do some file operations.")

    # The destructive command was denied
    assert len(denied_tools) == 1
    assert "rm -rf" in denied_tools[0]
    # The source file still exists
    assert (sandbox / "utils.py").exists()


@pytest.mark.asyncio
async def test_engine_respects_max_steps_in_real_scenario(sandbox: Path) -> None:
    """Agent that loops indefinitely is capped at max_steps."""
    # LLM that always wants to run one more test (never declares done)
    responses = [
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "shell_run",
                    "args": {
                        "command": f"{_PY} -m pytest test_utils.py -q --tb=no -k test_safe_divide_{i}",
                    },
                    "id": f"c{i}",
                    "type": "tool_call",
                }
            ],
        )
        for i in range(20)
    ]

    agent = GantryEngine(
        llm=FakeMessagesListChatModel(responses=responses),
        tools=[ShellTool(workspace=sandbox, allowed_commands=[_PY], timeout=30.0)],
        max_steps=3,
    )
    result = await agent.arun("Keep running tests forever.")
    # Should return SOMETHING after being capped
    assert isinstance(result, str)


# ── Performance sanity checks ─────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_engine_completes_in_reasonable_time(sandbox: Path) -> None:
    """Full 4-tool-call loop should complete in under 10 seconds."""
    import time

    start = time.monotonic()
    agent = GantryEngine(
        llm=_scripted_llm(sandbox),
        tools=[
            FileSystemTools(workspace=sandbox),
            ShellTool(workspace=sandbox, allowed_commands=[_PY], timeout=30.0),
        ],
        max_steps=10,
    )
    await agent.arun("Fix the bugs.")
    elapsed = time.monotonic() - start
    assert elapsed < 10.0, f"Engine took {elapsed:.1f}s — too slow for 4 tool calls"
