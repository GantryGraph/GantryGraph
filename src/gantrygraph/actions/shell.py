"""Shell command execution tool with timeout and optional allowlist."""

from __future__ import annotations

import asyncio
import logging
import shlex
from pathlib import Path
from typing import TYPE_CHECKING

from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, Field

from gantrygraph.core.base_action import BaseAction

if TYPE_CHECKING:
    from gantrygraph.security.policies import ShellDenylist

logger = logging.getLogger(__name__)


class ShellTools(BaseAction):
    """Execute shell commands via ``asyncio.create_subprocess_exec``.

    Security controls:
    - *denylist*: regex-based filter applied before the OS sees the command.
      Defaults to ``ShellDenylist.default()`` (blocks catastrophic commands).
      Pass ``ShellDenylist.permissive()`` to disable, or build a custom list.
    - *allowed_commands*: if set, only listed executables are permitted.
    - *workspace*: if set, the subprocess cwd is locked to that path.
    - *timeout*: hard wall-clock limit in seconds (default 30 s).

    Example::

        from gantrygraph.security import ShellDenylist

        tools = ShellTools(
            workspace="/tmp/project",
            allowed_commands=["git", "ls", "cat"],
            denylist=ShellDenylist.strict(),
            timeout=15.0,
        )
    """

    def __init__(
        self,
        workspace: str | Path | None = None,
        allowed_commands: list[str] | None = None,
        timeout: float = 30.0,
        denylist: ShellDenylist | None = None,
    ) -> None:
        from gantrygraph.security.policies import ShellDenylist as _ShellDenylist

        self._workspace = Path(workspace).resolve() if workspace else None
        self._allowed = set(allowed_commands) if allowed_commands else None
        self._timeout = timeout
        self._denylist = denylist if denylist is not None else _ShellDenylist.default()

    def get_tools(self) -> list[BaseTool]:
        return [self._shell_tool()]

    def _shell_tool(self) -> BaseTool:
        allowed_commands = self._allowed
        workspace = self._workspace
        timeout = self._timeout
        denylist = self._denylist

        class _Args(BaseModel):
            command: str = Field(
                description=(
                    "Shell command to execute. Use absolute paths or workspace-relative paths."
                )
            )

        async def _run(command: str) -> str:
            parts = shlex.split(command)
            if not parts:
                return "Error: empty command."

            # ── Denylist check ───────────────────────────────────────────────
            matched = denylist.check(command)
            if matched:
                if denylist.on_match == "block":
                    return (
                        f"Error: command blocked by security policy "
                        f"(matched pattern: {matched!r}). "
                        "This command is not permitted for safety reasons."
                    )
                else:
                    logger.warning("Denylist warning — pattern %r matched: %s", matched, command)

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
                stdout_b, stderr_b = await asyncio.wait_for(proc.communicate(), timeout=timeout)
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
