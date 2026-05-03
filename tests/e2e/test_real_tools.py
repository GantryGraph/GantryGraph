"""E2E tests — real filesystem + real subprocess, no LLM needed.

These tests verify that the tool layer works correctly in a realistic
scenario: reading source files, running pytest, detecting failures,
writing fixes, re-running to confirm.  They don't require an API key.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pytest

from gantrygraph.actions.filesystem import FileSystemTools
from gantrygraph.actions.shell import ShellTool

_PY = sys.executable

DEMO_APP = Path(__file__).parent.parent.parent / "demo-app"


@pytest.fixture
def sandbox(tmp_path: Path) -> Path:
    """Copy demo-app into a temp sandbox so tests don't mutate the original."""
    dest = tmp_path / "app"
    shutil.copytree(DEMO_APP, dest)
    return dest


# ── Real filesystem operations ────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_file_read_source_code(sandbox: Path) -> None:
    tools = {t.name: t for t in FileSystemTools(workspace=sandbox).get_tools()}
    content = await tools["file_read"].ainvoke({"path": "utils.py"})
    assert "safe_divide" in content
    assert "is_palindrome" in content
    assert "BUG" in content


@pytest.mark.asyncio
async def test_file_list_demo_app(sandbox: Path) -> None:
    tools = {t.name: t for t in FileSystemTools(workspace=sandbox).get_tools()}
    listing = await tools["file_list"].ainvoke({"path": "."})
    assert "utils.py" in listing
    assert "test_utils.py" in listing


# ── Real pytest execution ─────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_shell_detects_failing_tests(sandbox: Path) -> None:
    """ShellTool can run pytest and see failing test names in output."""
    tool = ShellTool(workspace=sandbox).get_tools()[0]
    result = await tool.ainvoke({"command": f"{_PY} -m pytest test_utils.py -v --tb=short"})
    assert "FAILED" in result
    assert "test_safe_divide_by_zero" in result
    assert "test_is_palindrome_case_insensitive" in result
    assert "test_word_count_tabs" in result
    assert "3 failed" in result


@pytest.mark.asyncio
async def test_shell_passes_after_fixing_safe_divide(sandbox: Path) -> None:
    """Write a fix for safe_divide and confirm that specific test now passes."""
    fs_tools = {t.name: t for t in FileSystemTools(workspace=sandbox).get_tools()}
    shell = ShellTool(workspace=sandbox).get_tools()[0]

    # Read original
    original = await fs_tools["file_read"].ainvoke({"path": "utils.py"})
    assert "return a / b" in original

    # Write fix
    fixed = original.replace(
        "    # BUG 1: missing zero-guard — raises ZeroDivisionError instead of returning 0.0\n"
        "    return a / b",
        "    if b == 0.0:\n        return 0.0\n    return a / b",
    )
    await fs_tools["file_write"].ainvoke({"path": "utils.py", "content": fixed})

    # Verify
    result = await shell.ainvoke(
        {"command": f"{_PY} -m pytest test_utils.py::test_safe_divide_by_zero -v"}
    )
    assert "PASSED" in result
    assert "1 passed" in result


@pytest.mark.asyncio
async def test_full_tool_pipeline_fixes_all_bugs(sandbox: Path) -> None:
    """Simulate what the agent does: read → fix → verify, for all 3 bugs."""
    fs_tools = {t.name: t for t in FileSystemTools(workspace=sandbox).get_tools()}
    shell = ShellTool(workspace=sandbox).get_tools()[0]

    # Step 1: run tests, confirm 3 failures
    initial = await shell.ainvoke({"command": f"{_PY} -m pytest test_utils.py -v --tb=line"})
    assert "3 failed" in initial

    # Step 2: read source
    source = await fs_tools["file_read"].ainvoke({"path": "utils.py"})

    # Step 3: apply all three fixes programmatically
    fixed = source
    # Fix 1: safe_divide
    fixed = fixed.replace(
        "    # BUG 1: missing zero-guard — raises ZeroDivisionError instead of returning 0.0\n"
        "    return a / b",
        "    if b == 0.0:\n        return 0.0\n    return a / b",
    )
    # Fix 2: is_palindrome
    fixed = fixed.replace(
        '    # BUG 2: case-sensitive comparison — "Racecar" wrongly returns False\n'
        "    return text == text[::-1]",
        "    lower = text.lower()\n    return lower == lower[::-1]",
    )
    # Fix 3: word_count
    fixed = fixed.replace(
        '    # BUG 3: split(" ") fails on tabs and multiple consecutive spaces\n'
        '    words = text.split(" ")',
        "    words = text.split()",
    )

    # Step 4: write fixed file
    await fs_tools["file_write"].ainvoke({"path": "utils.py", "content": fixed})

    # Step 5: verify all tests pass
    final = await shell.ainvoke({"command": f"{_PY} -m pytest test_utils.py -v"})
    assert "FAILED" not in final
    assert "14 passed" in final


# ── Error self-correction simulation ─────────────────────────────────────────


@pytest.mark.asyncio
async def test_shell_captures_syntax_error_gracefully(sandbox: Path) -> None:
    """If the agent writes broken Python, ShellTool captures the error output."""
    fs_tools = {t.name: t for t in FileSystemTools(workspace=sandbox).get_tools()}
    shell = ShellTool(workspace=sandbox).get_tools()[0]

    # Write intentionally broken Python
    broken = "def broken(\n    pass\n"
    await fs_tools["file_write"].ainvoke({"path": "broken.py", "content": broken})

    result = await shell.ainvoke({"command": f"{_PY} broken.py"})
    # Should NOT crash the test — error is returned as a string
    assert isinstance(result, str)
    assert len(result) > 0
    # Python syntax errors print to stderr, captured by ShellTool
    assert "SyntaxError" in result or "Error" in result or "error" in result


@pytest.mark.asyncio
async def test_shell_nonexistent_command_returns_error_string(sandbox: Path) -> None:
    """Calling a nonexistent binary returns an error string, not an exception."""
    shell = ShellTool(workspace=sandbox).get_tools()[0]
    result = await shell.ainvoke({"command": "nonexistent_binary_xyz_123"})
    assert isinstance(result, str)
    assert "No such file" in result or "not found" in result or "Exit code" in result


@pytest.mark.asyncio
async def test_path_traversal_blocked_even_in_e2e(sandbox: Path) -> None:
    """Security: filesystem tool cannot escape sandbox even with real paths."""
    fs_tools = {t.name: t for t in FileSystemTools(workspace=sandbox).get_tools()}
    with pytest.raises(PermissionError):
        await fs_tools["file_read"].ainvoke({"path": "../../README.md"})


# ── Token efficiency check ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_screenshot_compression_reduces_size() -> None:
    """DesktopScreen resizes large images to fit max_resolution budget."""
    import io

    from PIL import Image

    from gantrygraph.perception.desktop import _resize_preserving_aspect

    # Simulate a 4K screenshot (expensive in tokens)
    big_img = Image.new("RGB", (3840, 2160))
    resized = _resize_preserving_aspect(big_img, (1280, 720))
    assert resized.width <= 1280
    assert resized.height <= 720

    # Compare PNG sizes
    def png_size(img: Image.Image) -> int:
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return len(buf.getvalue())

    big_size = png_size(big_img)
    small_size = png_size(resized)
    ratio = small_size / big_size
    # Resized image should be at least 5× smaller
    assert ratio < 0.2, f"Expected >5× compression, got {1 / ratio:.1f}× (ratio={ratio:.2f})"
