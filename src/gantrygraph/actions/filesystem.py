"""Filesystem tools with workspace-boundary enforcement."""
from __future__ import annotations

import asyncio
from pathlib import Path

from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, Field

from gantrygraph._utils import safe_path
from gantrygraph.core.base_action import BaseAction


class FileSystemTools(BaseAction):
    """Read/write/list tools locked to a specific workspace directory.

    All paths are resolved relative to *workspace* and any attempt to
    escape via ``../`` or absolute paths raises ``PermissionError``.

    Example::

        tools = FileSystemTools(workspace="/home/user/project")
        agent = GantryEngine(..., tools=[tools])
    """

    def __init__(self, workspace: str | Path) -> None:
        self._workspace = Path(workspace).resolve()

    def get_tools(self) -> list[BaseTool]:
        return [
            self._read_tool(),
            self._write_tool(),
            self._list_tool(),
            self._delete_tool(),
        ]

    # ── Tool implementations ──────────────────────────────────────────────────

    def _read_tool(self) -> BaseTool:
        workspace = self._workspace

        class _Args(BaseModel):
            path: str = Field(description="Path relative to the workspace root.")

        async def _read(path: str) -> str:
            resolved = safe_path(workspace, path)
            if not resolved.exists():
                return f"Error: file '{path}' does not exist."
            if not resolved.is_file():
                return f"Error: '{path}' is a directory, not a file."
            content = await asyncio.get_event_loop().run_in_executor(
                None, resolved.read_text, "utf-8"
            )
            return content

        return StructuredTool.from_function(
            coroutine=_read,
            name="file_read",
            description="Read the contents of a file inside the workspace.",
            args_schema=_Args,
        )

    def _write_tool(self) -> BaseTool:
        workspace = self._workspace

        class _Args(BaseModel):
            path: str = Field(description="Path relative to the workspace root.")
            content: str = Field(description="Text content to write.")

        async def _write(path: str, content: str) -> str:
            resolved = safe_path(workspace, path)
            resolved.parent.mkdir(parents=True, exist_ok=True)
            await asyncio.get_event_loop().run_in_executor(
                None, resolved.write_text, content, "utf-8"
            )
            return f"Written {len(content)} characters to '{path}'."

        return StructuredTool.from_function(
            coroutine=_write,
            name="file_write",
            description="Write text content to a file inside the workspace.",
            args_schema=_Args,
        )

    def _list_tool(self) -> BaseTool:
        workspace = self._workspace

        class _Args(BaseModel):
            path: str = Field(default=".", description="Directory path relative to workspace.")

        async def _list(path: str = ".") -> str:
            resolved = safe_path(workspace, path)
            if not resolved.exists():
                return f"Error: path '{path}' does not exist."
            if not resolved.is_dir():
                return f"Error: '{path}' is a file, not a directory."
            entries = await asyncio.get_event_loop().run_in_executor(
                None, lambda: sorted(str(p.relative_to(workspace)) for p in resolved.iterdir())
            )
            return "\n".join(entries) if entries else "(empty directory)"

        return StructuredTool.from_function(
            coroutine=_list,
            name="file_list",
            description="List files and directories at a path inside the workspace.",
            args_schema=_Args,
        )

    def _delete_tool(self) -> BaseTool:
        workspace = self._workspace

        class _Args(BaseModel):
            path: str = Field(description="Path relative to workspace to delete.")

        async def _delete(path: str) -> str:
            resolved = safe_path(workspace, path)
            if not resolved.exists():
                return f"Error: '{path}' does not exist."
            if resolved.is_dir():
                import shutil
                await asyncio.get_event_loop().run_in_executor(
                    None, shutil.rmtree, resolved
                )
                return f"Deleted directory '{path}'."
            await asyncio.get_event_loop().run_in_executor(None, resolved.unlink)
            return f"Deleted file '{path}'."

        return StructuredTool.from_function(
            coroutine=_delete,
            name="file_delete",
            description="Delete a file or directory inside the workspace.",
            args_schema=_Args,
        )
