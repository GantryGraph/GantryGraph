"""Shell command execution tool with timeout and optional allowlist."""
from __future__ import annotations

import asyncio
import shlex
from pathlib import Path

from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, Field

from gantrygraph.core.base_action import BaseAction


class ShellTools(BaseAction):
    """Execute shell commands via ``asyncio.create_subprocess_exec``.

    Security controls:
    - *allowed_commands*: if set, only listed executables are permitted.
    - *workspace*: if set, the subprocess cwd is locked to that path.
    - *timeout*: hard wall-clock limit in seconds (default 30 s).

    Example::

        tools = ShellTool(
            workspace="/tmp/project",
            allowed_commands=["git", "ls", "cat"],
            timeout=15.0,
        )
    """

    def __init__(
        self,
        workspace: str | Path | None = None,
        allowed_commands: list[str] | None = None,
        timeout: float = 30.0,
    ) -> None:
        self._workspace = Path(workspace).resolve() if workspace else None
        self._allowed = set(allowed_commands) if allowed_commands else None
        self._timeout = timeout

    def get_tools(self) -> list[BaseTool]:
        return [self._shell_tool()]

    def _shell_tool(self) -> BaseTool:
        allowed_commands = self._allowed
        workspace = self._workspace
        timeout = self._timeout

        class _Args(BaseModel):
            command: str = Field(
                description=(
                    "Shell command to execute."
                    " Use absolute paths or workspace-relative paths."
                )
            )

        async def _run(command: str) -> str:
            parts = shlex.split(command)
            if not parts:
                return "Error: empty command."

            if allowed_commands is not None and parts[0] not in allowed_commands:
                return (
                    f"Error: command '{parts[0]}' is not in the allowed list. "
                    f"Allowed: {sorted(allowed_commands)}"
                )

            try:
                proc = await asyncio.create_subprocess_exec(
                    *parts,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=str(workspace) if workspace else None,
                )
            except (FileNotFoundError, OSError) as exc:
                return f"Error: command not found — {exc}"
            try:
                stdout_b, stderr_b = await asyncio.wait_for(
                    proc.communicate(), timeout=timeout
                )
            except TimeoutError:
                proc.kill()
                await proc.communicate()
                return f"Error: command timed out after {timeout}s."

            stdout = stdout_b.decode("utf-8", errors="replace")
            stderr = stderr_b.decode("utf-8", errors="replace")
            output = (stdout + stderr).strip()
            exit_code = proc.returncode

            if exit_code != 0:
                return f"Exit code {exit_code}:\n{output}"
            return output or "(no output)"

        return StructuredTool.from_function(
            coroutine=_run,
            name="shell_run",
            description=(
                "Run a shell command and return its stdout+stderr. "
                "Use for file operations, git commands, running scripts, etc."
            ),
            args_schema=_Args,
        )


ShellTool = ShellTools  # backward compat alias
