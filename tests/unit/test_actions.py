"""Unit tests for gantrygraph/actions — FileSystemTools and ShellTool."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

from gantrygraph.actions.filesystem import FileSystemTools
from gantrygraph.actions.shell import ShellTools

# ── FileSystemTools ──────────────────────────────────────────────────────────

@pytest.fixture
def tmp_workspace(tmp_path: Path) -> Path:
    (tmp_path / "subdir").mkdir()
    (tmp_path / "hello.txt").write_text("hello world", encoding="utf-8")
    return tmp_path


def test_filesystem_tools_returns_four_tools(tmp_workspace: Path) -> None:
    tools = FileSystemTools(workspace=tmp_workspace).get_tools()
    assert len(tools) == 4
    names = {t.name for t in tools}
    assert names == {"file_read", "file_write", "file_list", "file_delete"}


@pytest.mark.asyncio
async def test_file_read_existing(tmp_workspace: Path) -> None:
    tools = {t.name: t for t in FileSystemTools(workspace=tmp_workspace).get_tools()}
    result = await tools["file_read"].ainvoke({"path": "hello.txt"})
    assert result == "hello world"


@pytest.mark.asyncio
async def test_file_read_missing(tmp_workspace: Path) -> None:
    tools = {t.name: t for t in FileSystemTools(workspace=tmp_workspace).get_tools()}
    result = await tools["file_read"].ainvoke({"path": "nonexistent.txt"})
    assert "does not exist" in result


@pytest.mark.asyncio
async def test_file_write_creates_file(tmp_workspace: Path) -> None:
    tools = {t.name: t for t in FileSystemTools(workspace=tmp_workspace).get_tools()}
    result = await tools["file_write"].ainvoke({"path": "new.txt", "content": "hi"})
    assert "Written" in result
    assert (tmp_workspace / "new.txt").read_text() == "hi"


@pytest.mark.asyncio
async def test_file_write_creates_subdirs(tmp_workspace: Path) -> None:
    tools = {t.name: t for t in FileSystemTools(workspace=tmp_workspace).get_tools()}
    await tools["file_write"].ainvoke({"path": "a/b/c.txt", "content": "deep"})
    assert (tmp_workspace / "a" / "b" / "c.txt").exists()


@pytest.mark.asyncio
async def test_file_list_root(tmp_workspace: Path) -> None:
    tools = {t.name: t for t in FileSystemTools(workspace=tmp_workspace).get_tools()}
    result = await tools["file_list"].ainvoke({"path": "."})
    assert "hello.txt" in result


@pytest.mark.asyncio
async def test_file_delete(tmp_workspace: Path) -> None:
    tools = {t.name: t for t in FileSystemTools(workspace=tmp_workspace).get_tools()}
    result = await tools["file_delete"].ainvoke({"path": "hello.txt"})
    assert "Deleted" in result
    assert not (tmp_workspace / "hello.txt").exists()


@pytest.mark.asyncio
async def test_file_read_path_traversal_blocked(tmp_workspace: Path) -> None:
    tools = {t.name: t for t in FileSystemTools(workspace=tmp_workspace).get_tools()}
    with pytest.raises(PermissionError, match="escapes the workspace"):
        await tools["file_read"].ainvoke({"path": "../../etc/passwd"})


@pytest.mark.asyncio
async def test_file_write_path_traversal_blocked(tmp_workspace: Path) -> None:
    tools = {t.name: t for t in FileSystemTools(workspace=tmp_workspace).get_tools()}
    with pytest.raises(PermissionError):
        await tools["file_write"].ainvoke({"path": "../evil.txt", "content": "bad"})


# ── ShellTools ────────────────────────────────────────────────────────────────

def test_shell_tool_returns_one_tool() -> None:
    tools = ShellTools().get_tools()
    assert len(tools) == 1
    assert tools[0].name == "shell_run"


@pytest.mark.asyncio
async def test_shell_tool_echo() -> None:
    tool = ShellTools().get_tools()[0]
    result = await tool.ainvoke({"command": "echo hello"})
    assert "hello" in result


@pytest.mark.asyncio
async def test_shell_tool_allowlist_blocks_command() -> None:
    tool = ShellTools(allowed_commands=["ls"]).get_tools()[0]
    result = await tool.ainvoke({"command": "echo not_allowed"})
    assert "not in the allowed list" in result


@pytest.mark.asyncio
async def test_shell_tool_allowlist_permits_command() -> None:
    tool = ShellTools(allowed_commands=["echo"]).get_tools()[0]
    result = await tool.ainvoke({"command": "echo permitted"})
    assert "permitted" in result


@pytest.mark.asyncio
async def test_shell_tool_nonzero_exit_includes_code() -> None:
    tool = ShellTools().get_tools()[0]
    result = await tool.ainvoke({"command": "sh -c 'exit 42'"})
    assert "42" in result


@pytest.mark.asyncio
async def test_shell_tool_timeout() -> None:
    tool = ShellTools(timeout=0.1).get_tools()[0]
    result = await tool.ainvoke({"command": "sleep 5"})
    assert "timed out" in result


@pytest.mark.asyncio
async def test_shell_tool_workspace_cwd(tmp_path: Path) -> None:
    (tmp_path / "marker.txt").write_text("found", encoding="utf-8")
    tool = ShellTools(workspace=tmp_path).get_tools()[0]
    result = await tool.ainvoke({"command": "ls"})
    assert "marker.txt" in result


@pytest.mark.asyncio
async def test_shell_tool_empty_command() -> None:
    tool = ShellTools().get_tools()[0]
    result = await tool.ainvoke({"command": ""})
    assert "empty" in result.lower()


# ── MouseKeyboardTools import guard ──────────────────────────────────────────

def test_mouse_keyboard_raises_without_extra() -> None:
    """If pyautogui is not installed, ImportError with helpful message."""
    import importlib
    import unittest.mock

    with unittest.mock.patch.dict(sys.modules, {"pyautogui": None}):
        # Re-import the module to trigger the guard
        import gantrygraph.actions.mouse_keyboard as mkm
        importlib.reload(mkm)
        assert not mkm._HAS_PYAUTOGUI
        with pytest.raises(ImportError, match="desktop"):
            mkm.MouseKeyboardTools()


# ── BrowserTools import guard ─────────────────────────────────────────────────

def test_browser_tools_raises_without_extra() -> None:
    """If playwright is not installed, ImportError with helpful message."""
    import importlib
    import unittest.mock

    with unittest.mock.patch.dict(sys.modules, {"playwright": None, "playwright.async_api": None}):
        import gantrygraph.actions.browser as bt
        importlib.reload(bt)
        assert not bt._HAS_PLAYWRIGHT
        with pytest.raises(ImportError, match="browser"):
            bt.BrowserTools()
