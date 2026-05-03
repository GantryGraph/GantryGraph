"""Unit tests for gantrygraph.presets and gantrygraph.config."""

from __future__ import annotations

import os
from typing import Any

import pytest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage


def _done_llm() -> FakeMessagesListChatModel:
    return FakeMessagesListChatModel(responses=[AIMessage(content="Task complete.")])


# ── Import-guard audit ────────────────────────────────────────────────────────


def test_import_claw_never_fails() -> None:
    """`import gantrygraph` must succeed with only core deps installed."""
    import gantrygraph  # noqa: F401 — just test it doesn't raise

    assert gantrygraph.GantryEngine is not None
    assert gantrygraph.GantryConfig is not None
    assert gantrygraph.InMemoryStore is not None


def test_import_claw_actions_never_fails() -> None:
    import gantrygraph.actions  # noqa: F401


def test_import_claw_perception_never_fails() -> None:
    import gantrygraph.perception  # noqa: F401


def test_import_claw_cloud_never_fails() -> None:
    """cloud __init__ must import without fastapi being used."""
    import gantrygraph.cloud  # noqa: F401
    from gantrygraph.cloud import RunRequest, RunResponse

    assert RunRequest is not None
    assert RunResponse is not None


def test_import_gantry_memory_never_fails() -> None:
    import gantrygraph.memory  # noqa: F401
    from gantrygraph.memory import InMemoryStore

    assert InMemoryStore is not None


def test_import_claw_telemetry_never_fails() -> None:
    import gantrygraph.telemetry  # noqa: F401


def test_import_claw_presets_never_fails() -> None:
    import gantrygraph.presets  # noqa: F401
    from gantrygraph.presets import code_agent

    assert callable(code_agent)


def test_desktop_agent_raises_without_extra() -> None:
    """desktop_agent must give a clear ImportError when pyautogui is absent."""
    import sys

    pag_backup = sys.modules.pop("pyautogui", None)
    # Mark as unimportable
    sys.modules["pyautogui"] = None  # type: ignore[assignment]
    try:
        import importlib

        import gantrygraph.actions.mouse_keyboard as mk

        importlib.reload(mk)
        if not mk._HAS_PYAUTOGUI:
            with pytest.raises(ImportError, match="desktop"):
                from gantrygraph.presets import desktop_agent

                desktop_agent(llm=_done_llm())
    finally:
        if pag_backup is not None:
            sys.modules["pyautogui"] = pag_backup
        else:
            sys.modules.pop("pyautogui", None)
        try:
            importlib.reload(mk)
        except Exception:
            pass


def test_browser_agent_raises_without_extra() -> None:
    import sys

    pw_backup = sys.modules.pop("playwright", None)
    sys.modules["playwright"] = None  # type: ignore[assignment]
    try:
        import importlib

        import gantrygraph.actions.browser as bmod

        importlib.reload(bmod)
        if not bmod._HAS_PLAYWRIGHT:
            with pytest.raises(ImportError, match="browser"):
                from gantrygraph.presets import browser_agent

                browser_agent(llm=_done_llm())
    finally:
        if pw_backup is not None:
            sys.modules["playwright"] = pw_backup
        else:
            sys.modules.pop("playwright", None)
        importlib.reload(bmod)


# ── code_agent ──────────────────────────────────────────────────────────────────


def test_code_agent_creates_engine(tmp_path: Any) -> None:
    from gantrygraph import GantryEngine
    from gantrygraph.presets import code_agent

    agent = code_agent(llm=_done_llm(), workspace=str(tmp_path))
    assert isinstance(agent, GantryEngine)


def test_code_agent_has_filesystem_and_shell_tools(tmp_path: Any) -> None:
    from gantrygraph.presets import code_agent

    agent = code_agent(llm=_done_llm(), workspace=str(tmp_path))
    names = {t.name for t in agent._collect_tools()}
    assert "file_read" in names
    assert "shell_run" in names


def test_code_agent_attaches_memory(tmp_path: Any) -> None:
    from gantrygraph.memory import InMemoryStore
    from gantrygraph.presets import code_agent

    agent = code_agent(llm=_done_llm(), workspace=str(tmp_path), with_memory=True)
    assert isinstance(agent._memory, InMemoryStore)


def test_code_agent_without_memory(tmp_path: Any) -> None:
    from gantrygraph.presets import code_agent

    agent = code_agent(llm=_done_llm(), workspace=str(tmp_path), with_memory=False)
    assert agent._memory is None


@pytest.mark.asyncio
async def test_code_agent_runs_to_completion(tmp_path: Any) -> None:
    from gantrygraph.presets import code_agent

    agent = code_agent(llm=_done_llm(), workspace=str(tmp_path), max_steps=5)
    result = await agent.arun("Count the files in the workspace")
    assert isinstance(result, str)
    assert len(result) > 0


def test_code_agent_forwards_kwargs(tmp_path: Any) -> None:
    from gantrygraph.presets import code_agent

    events: list = []
    agent = code_agent(
        llm=_done_llm(),
        workspace=str(tmp_path),
        on_event=lambda e: events.append(e),
    )
    assert agent._event_cb is not None


# ── api_agent ───────────────────────────────────────────────────────────────


def test_api_agent_creates_engine(tmp_path: Any) -> None:
    from gantrygraph import GantryEngine
    from gantrygraph.presets import api_agent

    agent = api_agent(llm=_done_llm(), workspace=str(tmp_path))
    assert isinstance(agent, GantryEngine)


def test_api_agent_with_suspension_sets_checkpointer(tmp_path: Any) -> None:
    from gantrygraph.presets import api_agent

    agent = api_agent(llm=_done_llm(), workspace=str(tmp_path), enable_suspension=True)
    assert agent._checkpointer is not None
    assert agent._use_interrupt is True


# ── mcp_agent ─────────────────────────────────────────────────────────────────


def test_mcp_agent_requires_at_least_one_server() -> None:
    from gantrygraph.presets import mcp_agent

    with pytest.raises(ValueError, match="at least one"):
        mcp_agent(llm=_done_llm())


def test_mcp_agent_creates_mcp_clients() -> None:
    from gantrygraph.mcp import MCPClient
    from gantrygraph.presets import mcp_agent

    agent = mcp_agent(_done_llm(), "echo hello")
    assert any(isinstance(t, MCPClient) for t in agent._raw_tools)


# ── GantryConfig ────────────────────────────────────────────────────────────────


def test_config_default_values() -> None:
    from gantrygraph import GantryConfig

    cfg = GantryConfig()
    assert cfg.max_steps == 50
    assert cfg.memory == "none"
    assert cfg.perception == "none"
    assert cfg.enable_suspension is False


def test_config_build_minimal() -> None:
    from gantrygraph import GantryConfig, GantryEngine

    cfg = GantryConfig(max_steps=10)
    agent = cfg.build(llm=_done_llm())
    assert isinstance(agent, GantryEngine)
    assert agent._max_steps == 10


def test_config_build_with_workspace(tmp_path: Any) -> None:
    from gantrygraph import GantryConfig

    cfg = GantryConfig(workspace=str(tmp_path))
    agent = cfg.build(llm=_done_llm())
    names = {t.name for t in agent._collect_tools()}
    assert "file_read" in names
    assert "shell_run" in names


def test_config_build_with_in_memory(tmp_path: Any) -> None:
    from gantrygraph import GantryConfig
    from gantrygraph.memory import InMemoryStore

    cfg = GantryConfig(workspace=str(tmp_path), memory="in_memory")
    agent = cfg.build(llm=_done_llm())
    assert isinstance(agent._memory, InMemoryStore)


def test_config_build_with_guardrail(tmp_path: Any) -> None:
    from gantrygraph import GantryConfig

    cfg = GantryConfig(
        workspace=str(tmp_path),
        guardrail_requires_approval=["shell_run", "file_delete"],
    )
    agent = cfg.build(llm=_done_llm())
    assert agent._guardrail is not None
    assert "shell_run" in agent._guardrail.requires_approval


def test_config_build_with_suspension(tmp_path: Any) -> None:
    from gantrygraph import GantryConfig

    cfg = GantryConfig(enable_suspension=True)
    agent = cfg.build(llm=_done_llm())
    assert agent._use_interrupt is True
    assert agent._checkpointer is not None


def test_config_build_extra_tools(tmp_path: Any) -> None:
    from gantrygraph import GantryConfig
    from gantrygraph.actions.shell import ShellTools

    extra = ShellTools(workspace=str(tmp_path))
    cfg = GantryConfig()
    agent = cfg.build(llm=_done_llm(), extra_tools=[extra])
    # extra tools are appended
    assert extra in agent._raw_tools


def test_config_from_env_reads_variables(tmp_path: Any) -> None:
    from gantrygraph import GantryConfig

    env = {
        "GANTRY_MAX_STEPS": "7",
        "GANTRY_WORKSPACE": str(tmp_path),
        "GANTRY_MEMORY": "in_memory",
        "GANTRY_ENABLE_SUSPENSION": "true",
        "GANTRY_GUARDRAIL_REQUIRES_APPROVAL": "shell_run, file_delete",
    }
    old = {k: os.environ.pop(k, None) for k in env}
    os.environ.update(env)
    try:
        cfg = GantryConfig.from_env()
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    assert cfg.max_steps == 7
    assert cfg.workspace == str(tmp_path)
    assert cfg.memory == "in_memory"
    assert cfg.enable_suspension is True
    assert "shell_run" in cfg.guardrail_requires_approval
    assert "file_delete" in cfg.guardrail_requires_approval


def test_config_from_env_custom_prefix() -> None:
    from gantrygraph import GantryConfig

    os.environ["MYAPP_MAX_STEPS"] = "99"
    try:
        cfg = GantryConfig.from_env(prefix="MYAPP_")
    finally:
        del os.environ["MYAPP_MAX_STEPS"]

    assert cfg.max_steps == 99


def test_config_from_yaml(tmp_path: Any) -> None:
    pytest.importorskip("yaml", reason="pyyaml not installed")
    from gantrygraph import GantryConfig

    yaml_path = tmp_path / "agent.yaml"
    yaml_path.write_text(
        "max_steps: 15\n"
        "memory: in_memory\n"
        f"workspace: {tmp_path}\n"
        "guardrail_requires_approval:\n"
        "  - shell_run\n"
    )
    cfg = GantryConfig.from_yaml(str(yaml_path))
    assert cfg.max_steps == 15
    assert cfg.memory == "in_memory"
    assert "shell_run" in cfg.guardrail_requires_approval


def test_config_build_with_max_wall_seconds() -> None:
    from gantrygraph import GantryConfig
    from gantrygraph.security.policies import BudgetPolicy

    cfg = GantryConfig(max_steps=20, max_wall_seconds=60.0)
    agent = cfg.build(llm=_done_llm())
    assert agent._budget is not None
    assert isinstance(agent._budget, BudgetPolicy)
    assert agent._budget.max_wall_seconds == 60.0
    assert agent._budget.max_steps == 20


def test_config_build_no_budget_when_no_wall_seconds() -> None:
    from gantrygraph import GantryConfig

    cfg = GantryConfig(max_steps=20)
    agent = cfg.build(llm=_done_llm())
    assert agent._budget is None


def test_config_from_env_desktop_max_resolution() -> None:
    from gantrygraph import GantryConfig

    os.environ["GANTRY_DESKTOP_MAX_RESOLUTION"] = "800,600"
    try:
        cfg = GantryConfig.from_env()
    finally:
        del os.environ["GANTRY_DESKTOP_MAX_RESOLUTION"]

    assert cfg.desktop_max_resolution == (800, 600)


def test_config_from_env_desktop_max_resolution_space_separated() -> None:
    from gantrygraph import GantryConfig

    os.environ["GANTRY_DESKTOP_MAX_RESOLUTION"] = "1920 1080"
    try:
        cfg = GantryConfig.from_env()
    finally:
        del os.environ["GANTRY_DESKTOP_MAX_RESOLUTION"]

    assert cfg.desktop_max_resolution == (1920, 1080)


def test_config_from_env_max_wall_seconds() -> None:
    from gantrygraph import GantryConfig

    os.environ["GANTRY_MAX_WALL_SECONDS"] = "120.5"
    try:
        cfg = GantryConfig.from_env()
    finally:
        del os.environ["GANTRY_MAX_WALL_SECONDS"]

    assert cfg.max_wall_seconds == 120.5


@pytest.mark.asyncio
async def test_config_build_and_run_e2e(tmp_path: Any) -> None:
    """Full round-trip: build from config, run agent, check result."""
    from gantrygraph import GantryConfig

    cfg = GantryConfig(
        workspace=str(tmp_path),
        memory="in_memory",
        max_steps=5,
    )
    agent = cfg.build(llm=_done_llm())
    result = await agent.arun("List the workspace contents")
    assert isinstance(result, str)
    assert len(result) > 0
    # Memory should have been stored
    assert len(agent._memory) == 1  # type: ignore[arg-type]
