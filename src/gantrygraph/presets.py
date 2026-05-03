"""Ready-to-use GantryEngine factory functions for the most common scenarios.

Pick the preset that matches your use case, pass your LLM, and go.
Every preset uses lazy imports so only the extras you have installed
are ever touched.

Quick start::

    from gantrygraph.presets import code_agent
    from langchain_anthropic import ChatAnthropic

    agent = code_agent(
        llm=ChatAnthropic(model="claude-sonnet-4-6"),
        workspace="/my/project",
    )
    agent.run("Find and fix all failing tests")

Available presets:

+-------------------+------------------------------+------------------------+
| Preset            | Use case                     | Required extras        |
+===================+==============================+========================+
| ``code_agent``    | code review, test-fixing     | none (core only)       |
+-------------------+------------------------------+------------------------+
| ``desktop_agent`` | screenshot + mouse/keyboard  | ``[desktop]``          |
+-------------------+------------------------------+------------------------+
| ``browser_agent`` | web scraping, form filling   | ``[browser]``          |
+-------------------+------------------------------+------------------------+
| ``mcp_agent``     | MCP tool servers             | none (core only)       |
+-------------------+------------------------------+------------------------+
| ``api_agent``     | REST + SSE cloud deployment  | ``[cloud]``            |
+-------------------+------------------------------+------------------------+
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from gantrygraph.engine.engine import GantryEngine


def code_agent(
    llm: Any,
    workspace: str = ".",
    *,
    shell_allowed_commands: list[str] | None = None,
    shell_timeout: float = 30.0,
    with_memory: bool = True,
    max_steps: int = 30,
    **kwargs: Any,
) -> GantryEngine:
    """Agent for code review, debugging, and test-fixing tasks.

    Bundles ``FileSystemTools`` + ``ShellTools`` + ``InMemoryStore`` memory.
    No optional extras required.

    Args:
        llm:                    Any LangChain ``BaseChatModel``.
        workspace:              Working directory for file and shell tools.
        shell_allowed_commands: Allowlist of executables.  ``None`` = allow all.
        shell_timeout:          Hard wall-clock limit per shell command (seconds).
        with_memory:            Inject ``InMemoryStore`` for within-session recall.
        max_steps:              Hard step cap.
        **kwargs:               Forwarded to ``GantryEngine`` (e.g. ``on_event``).

    Example::

        agent = code_agent(llm, workspace="/app", shell_allowed_commands=["python", "git"])
        agent.run("Run the test suite and fix any failures")
    """
    from gantrygraph.actions.filesystem import FileSystemTools
    from gantrygraph.actions.shell import ShellTools
    from gantrygraph.engine.engine import GantryEngine
    from gantrygraph.memory.in_memory import InMemoryStore

    tools: list[Any] = [
        FileSystemTools(workspace=workspace),
        ShellTools(
            workspace=workspace,
            allowed_commands=shell_allowed_commands,
            timeout=shell_timeout,
        ),
    ]
    return GantryEngine(
        llm=llm,
        tools=tools,
        memory=InMemoryStore() if with_memory else None,
        max_steps=max_steps,
        **kwargs,
    )


def desktop_agent(
    llm: Any,
    *,
    max_resolution: tuple[int, int] = (1280, 720),
    with_memory: bool = False,
    max_steps: int = 50,
    **kwargs: Any,
) -> GantryEngine:
    """Agent for desktop automation — screenshot perception + mouse/keyboard.

    Requires the ``[desktop]`` extra::

        pip install 'gantrygraph[desktop]'

    Args:
        llm:            Any LangChain ``BaseChatModel``.
        max_resolution: Cap the screenshot resolution to reduce token cost.
        with_memory:    Add ``InMemoryVector`` for within-session recall.
        max_steps:      Hard step cap.
        **kwargs:       Forwarded to ``GantryEngine``.
    """
    from gantrygraph.engine.engine import GantryEngine
    from gantrygraph.memory.in_memory import InMemoryStore
    from gantrygraph.perception.desktop import DesktopScreen

    try:
        from gantrygraph.actions.mouse_keyboard import MouseKeyboardTools
    except ImportError as exc:
        raise ImportError(
            "desktop_agent requires the [desktop] extra: "
            "pip install 'gantrygraph[desktop]'"
        ) from exc

    return GantryEngine(
        llm=llm,
        perception=DesktopScreen(max_resolution=max_resolution),
        tools=[MouseKeyboardTools()],
        memory=InMemoryStore() if with_memory else None,
        max_steps=max_steps,
        **kwargs,
    )


def browser_agent(
    llm: Any,
    *,
    start_url: str | None = None,
    headless: bool = True,
    with_memory: bool = False,
    max_steps: int = 30,
    **kwargs: Any,
) -> GantryEngine:
    """Agent for web browsing, scraping, and form-filling tasks.

    Requires the ``[browser]`` extra::

        pip install 'gantrygraph[browser]'
        playwright install chromium

    Args:
        llm:        Any LangChain ``BaseChatModel``.
        start_url:  Optional starting URL.  When provided, ``WebPage`` perception
                    is added and **shares the same browser** as ``BrowserTools``,
                    so the agent automatically sees the current page at each loop
                    iteration.  Without ``start_url``, the agent has no automatic
                    perception and must call ``browser_get_text`` explicitly.
        headless:   Run the browser in headless mode (default ``True``).
        with_memory: Add ``InMemoryVector`` for within-session recall.
        max_steps:  Hard step cap.
        **kwargs:   Forwarded to ``GantryEngine``.

    Example::

        # With perception (recommended for most use cases)
        agent = browser_agent(llm, start_url="https://example.com")

        # Without perception (agent reads the page explicitly via tools)
        agent = browser_agent(llm)
    """
    from gantrygraph.engine.engine import GantryEngine
    from gantrygraph.memory.in_memory import InMemoryStore

    try:
        from gantrygraph.actions.browser import BrowserTools
    except ImportError as exc:
        raise ImportError(
            "browser_agent requires the [browser] extra: "
            "pip install 'gantrygraph[browser]' && playwright install chromium"
        ) from exc

    if start_url is not None:
        # Share a single browser between perception and actions
        from gantrygraph.perception.web import WebPage

        web = WebPage(url=start_url, headless=headless)
        return GantryEngine(
            llm=llm,
            perception=web,
            tools=[BrowserTools(headless=headless, web_page=web)],
            memory=InMemoryStore() if with_memory else None,
            max_steps=max_steps,
            **kwargs,
        )

    return GantryEngine(
        llm=llm,
        tools=[BrowserTools(headless=headless)],
        memory=InMemoryStore() if with_memory else None,
        max_steps=max_steps,
        **kwargs,
    )


def mcp_agent(
    llm: Any,
    *server_commands: str,
    with_memory: bool = True,
    max_steps: int = 50,
    **kwargs: Any,
) -> GantryEngine:
    """Agent that connects to one or more MCP tool servers.

    No optional extras required — ``mcp`` is a core dependency.

    Args:
        llm:             Any LangChain ``BaseChatModel``.
        *server_commands: Shell commands to launch MCP servers.
        with_memory:     Add ``InMemoryVector`` for within-session recall.
        max_steps:       Hard step cap.
        **kwargs:        Forwarded to ``GantryEngine``.

    Example::

        agent = mcp_agent(
            llm,
            "npx -y @modelcontextprotocol/server-filesystem /tmp",
            "npx -y @modelcontextprotocol/server-github",
        )
        # MCP subprocesses start automatically inside arun()
        result = agent.run("List the open PRs and summarise them")
    """
    from gantrygraph.engine.engine import GantryEngine
    from gantrygraph.mcp.client import MCPClient
    from gantrygraph.memory.in_memory import InMemoryStore

    if not server_commands:
        raise ValueError("mcp_agent requires at least one server_command.")

    connectors = [MCPClient(cmd) for cmd in server_commands]
    return GantryEngine(
        llm=llm,
        tools=list(connectors),
        memory=InMemoryStore() if with_memory else None,
        max_steps=max_steps,
        **kwargs,
    )


def api_agent(
    llm: Any,
    workspace: str = ".",
    *,
    enable_suspension: bool = False,
    with_memory: bool = True,
    max_steps: int = 30,
    **kwargs: Any,
) -> GantryEngine:
    """Agent preset optimised for cloud / container deployment.

    Bundles ``FileSystemTools`` + ``ShellTools``.
    When ``enable_suspension=True``, uses ``MemorySaver`` so the agent can be
    frozen and resumed across HTTP requests (see ``POST /resume/{job_id}``).

    Requires the ``[cloud]`` extra for the REST server::

        pip install 'gantrygraph[cloud]'

    Args:
        llm:               Any LangChain ``BaseChatModel``.
        workspace:         Sandbox directory for file and shell operations.
        enable_suspension: Enable LangGraph interrupt-based HITL suspension.
        with_memory:       Add ``InMemoryVector`` for within-session recall.
        max_steps:         Hard step cap.
        **kwargs:          Forwarded to ``GantryEngine``.
    """
    from gantrygraph.actions.filesystem import FileSystemTools
    from gantrygraph.actions.shell import ShellTools
    from gantrygraph.engine.engine import GantryEngine
    from gantrygraph.memory.in_memory import InMemoryStore

    return GantryEngine(
        llm=llm,
        tools=[
            FileSystemTools(workspace=workspace),
            ShellTools(workspace=workspace),
        ],
        memory=InMemoryStore() if with_memory else None,
        enable_suspension=enable_suspension,
        max_steps=max_steps,
        **kwargs,
    )


# Backward compat aliases
qa_agent = code_agent
cloud_agent = api_agent
