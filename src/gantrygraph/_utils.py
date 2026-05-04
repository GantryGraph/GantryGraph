from __future__ import annotations

import asyncio
import concurrent.futures
import inspect
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any, TypeVar

T = TypeVar("T")


async def ensure_awaitable(fn: Callable[..., Any], *args: Any) -> Any:
    """Call fn(*args); if the result is a coroutine, await it.

    This lets GantryEngine accept both ``def`` and ``async def`` callbacks
    for ``approval_callback`` and ``on_event`` without the caller having
    to think about it.
    """
    result = fn(*args)
    if inspect.isawaitable(result):
        return await result
    return result


def _run_sync(coro: Awaitable[T]) -> T:
    """Run a coroutine synchronously with Jupyter-compatible fallback.

    If there is already a running event loop (e.g. inside a Jupyter notebook),
    the coroutine is dispatched to a fresh loop in a worker thread to avoid
    the "cannot run nested event loop" error.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)  # type: ignore[arg-type]

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        future: concurrent.futures.Future[T] = pool.submit(asyncio.run, coro)  # type: ignore[arg-type]
        return future.result()


def safe_path(workspace: Path, user_path: str) -> Path:
    """Resolve *user_path* relative to *workspace*, raising on traversal.

    Prevents path traversal attacks in FileSystemTools and ShellTool by
    ensuring the resolved path stays inside the declared workspace root.
    """
    resolved = (workspace / user_path).resolve()
    try:
        resolved.relative_to(workspace.resolve())
    except ValueError:
        raise PermissionError(
            f"Path '{user_path}' escapes the workspace boundary '{workspace}'."
        ) from None
    return resolved


def safe_path_multi(allowed_paths: list[Path] | None, user_path: str) -> Path:
    """Resolve *user_path*, ensuring it falls within at least one allowed directory.

    If *allowed_paths* is ``None`` the path is resolved without restriction
    (unrestricted / full-access mode).

    For relative paths, resolution is attempted against each allowed path in
    order; the first match wins.  For absolute paths the full list is checked.
    """
    if allowed_paths is None:
        return Path(user_path).resolve()

    p = Path(user_path)
    resolved = (allowed_paths[0] / user_path).resolve() if not p.is_absolute() else p.resolve()

    for base in allowed_paths:
        try:
            resolved.relative_to(base.resolve())
            return resolved
        except ValueError:
            continue

    raise PermissionError(
        f"Path '{user_path}' escapes the workspace boundary. "
        f"Allowed directories: {[str(b) for b in allowed_paths]}"
    ) from None
