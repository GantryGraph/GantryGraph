"""Configuration-driven GantryEngine setup.

``GantryConfig`` lets you fully specify an agent from a YAML file, environment
variables, or a plain dict — no code changes required between environments.

Quick start::

    from gantrygraph.config import GantryConfig
    from langchain_anthropic import ChatAnthropic

    config = GantryConfig(
        workspace="/app",
        memory="in_memory",
        telemetry_otlp_endpoint="http://collector:4317",
    )
    agent = config.build(llm=ChatAnthropic(model="claude-sonnet-4-6"))
    agent.run("Fix the failing tests")

Load from YAML::

    config = GantryConfig.from_yaml("agent.yaml")
    agent = config.build(llm=my_llm)

Load from environment variables (prefix ``CLAW_``)::

    # In shell:
    #   export CLAW_WORKSPACE=/app
    #   export CLAW_MEMORY=chroma
    #   export CLAW_TELEMETRY_OTLP_ENDPOINT=http://localhost:4317
    config = GantryConfig.from_env()
"""
from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from gantrygraph.engine.engine import GantryEngine


class GantryConfig(BaseModel):
    """Full declarative configuration for a ``GantryEngine`` instance.

    All fields have sensible defaults so you only need to set what you change.

    Attributes:
        max_steps:                  Hard upper bound on act-node executions.
        system_prompt:              Optional extra system prompt prepended to
                                    every run.
        workspace:                  If set, attaches ``FileSystemTools`` and
                                    ``ShellTool`` locked to this directory.
        shell_allowed_commands:     Allowlist for ``ShellTool``.  ``None`` =
                                    allow all executables.
        shell_timeout:              Per-command wall-clock limit (seconds).
        perception:                 Perception backend to attach.
                                    ``"none"`` | ``"desktop"`` | ``"web"``
        browser_headless:           Run browser in headless mode.
        memory:                     Long-term memory backend.
                                    ``"none"`` | ``"in_memory"`` | ``"chroma"``
        memory_persist_directory:   On-disk path for ``ChromaMemory``.
        memory_collection:          ChromaDB collection name.
        telemetry_service_name:     ``service.name`` OTel resource attribute.
        telemetry_otlp_endpoint:    If set, enables OTel export to this OTLP
                                    gRPC endpoint (e.g. Grafana Alloy, Datadog
                                    agent, Jaeger).
        enable_suspension:          Enable LangGraph interrupt-based HITL.
        guardrail_requires_approval: Tool names that require ``approval_callback``
                                    confirmation before execution.
        max_wall_seconds:           Wall-clock timeout for the entire ``arun()``
                                    call.  ``None`` = no timeout.
    """

    # ── core ──────────────────────────────────────────────────────────────────
    max_steps: int = Field(50, gt=0)
    max_wall_seconds: float | None = None
    system_prompt: str | None = None

    # ── tools ─────────────────────────────────────────────────────────────────
    workspace: str | None = None
    shell_allowed_commands: list[str] | None = None
    shell_timeout: float = Field(30.0, gt=0)

    # ── perception ────────────────────────────────────────────────────────────
    perception: Literal["none", "desktop", "web"] = "none"
    browser_headless: bool = True
    browser_url: str | None = None
    desktop_max_resolution: tuple[int, int] = (1280, 720)

    # ── memory ────────────────────────────────────────────────────────────────
    memory: Literal["none", "in_memory", "chroma"] = "none"
    memory_persist_directory: str | None = None
    memory_collection: str = "gantry_memory"

    # ── telemetry (OTel) ──────────────────────────────────────────────────────
    telemetry_service_name: str = "gantry-agent"
    telemetry_otlp_endpoint: str | None = None

    # ── security / suspension ─────────────────────────────────────────────────
    enable_suspension: bool = False
    guardrail_requires_approval: list[str] = Field(default_factory=list)

    # ── build ─────────────────────────────────────────────────────────────────

    def build(
        self,
        llm: Any,
        *,
        approval_callback: Any = None,
        on_event: Any = None,
        extra_tools: list[Any] | None = None,
    ) -> GantryEngine:
        """Assemble and return a configured ``GantryEngine``.

        Args:
            llm:               Any LangChain ``BaseChatModel``.
            approval_callback: Optional HITL callback (``(tool, args) → bool``).
            on_event:          Optional event callback; merged with the OTel
                               callback when ``telemetry_otlp_endpoint`` is set.
            extra_tools:       Additional tools appended after the built-in set.

        Returns:
            A ready-to-use ``GantryEngine`` instance.
        """
        from gantrygraph.core.base_perception import BasePerception
        from gantrygraph.engine.engine import GantryEngine
        from gantrygraph.memory.base import BaseMemory
        from gantrygraph.security.policies import GuardrailPolicy

        tools: list[Any] = list(extra_tools or [])
        perception_obj: BasePerception | None = None
        memory_obj: BaseMemory | None = None
        event_cb = on_event
        guardrail: GuardrailPolicy | None = None

        # ── tools ─────────────────────────────────────────────────────────────
        if self.workspace is not None:
            from gantrygraph.actions.filesystem import FileSystemTools
            from gantrygraph.actions.shell import ShellTool

            tools = [
                FileSystemTools(workspace=self.workspace),
                ShellTool(
                    workspace=self.workspace,
                    allowed_commands=self.shell_allowed_commands,
                    timeout=self.shell_timeout,
                ),
                *tools,
            ]

        # ── perception ────────────────────────────────────────────────────────
        if self.perception == "desktop":
            from gantrygraph.perception.desktop import DesktopScreen

            perception_obj = DesktopScreen(
                max_resolution=self.desktop_max_resolution
            )
        elif self.perception == "web":
            from gantrygraph.perception.web import WebPage

            perception_obj = WebPage(
                url=self.browser_url,
                headless=self.browser_headless,
            )

        # ── memory ────────────────────────────────────────────────────────────
        if self.memory == "in_memory":
            from gantrygraph.memory.in_memory import InMemoryVector

            memory_obj = InMemoryVector()
        elif self.memory == "chroma":
            from gantrygraph.memory.chroma import ChromaMemory

            memory_obj = ChromaMemory(
                collection_name=self.memory_collection,
                persist_directory=self.memory_persist_directory,
            )

        # ── telemetry ─────────────────────────────────────────────────────────
        if self.telemetry_otlp_endpoint is not None:
            from gantrygraph.telemetry.otel import OTelExporter

            otel_cb = OTelExporter(
                service_name=self.telemetry_service_name,
                otlp_endpoint=self.telemetry_otlp_endpoint,
            ).as_event_callback()

            # Merge with any user-supplied on_event
            if on_event is not None:
                _user_cb = on_event

                async def _merged_cb(event: Any) -> None:
                    from gantrygraph._utils import ensure_awaitable

                    otel_cb(event)
                    await ensure_awaitable(_user_cb, event)

                event_cb = _merged_cb
            else:
                event_cb = otel_cb

        # ── guardrail ─────────────────────────────────────────────────────────
        if self.guardrail_requires_approval:
            guardrail = GuardrailPolicy(
                requires_approval=set(self.guardrail_requires_approval)
            )

        # ── budget ────────────────────────────────────────────────────────────
        budget = None
        if self.max_wall_seconds is not None:
            from gantrygraph.security.policies import BudgetPolicy

            budget = BudgetPolicy(
                max_steps=self.max_steps,
                max_wall_seconds=self.max_wall_seconds,
            )

        return GantryEngine(
            llm=llm,
            perception=perception_obj,
            tools=tools,
            approval_callback=approval_callback,
            on_event=event_cb,
            max_steps=self.max_steps,
            system_prompt=self.system_prompt,
            memory=memory_obj,
            guardrail=guardrail,
            enable_suspension=self.enable_suspension,
            budget=budget,
        )

    # ── class-level constructors ──────────────────────────────────────────────

    @classmethod
    def from_yaml(cls, path: str) -> GantryConfig:
        """Load config from a YAML file.

        Requires ``pyyaml``::

            pip install pyyaml

        Example YAML::

            max_steps: 30
            workspace: /app
            memory: chroma
            memory_persist_directory: /data/memory
            telemetry_otlp_endpoint: http://localhost:4317
            guardrail_requires_approval:
              - shell_run
              - file_delete
        """
        try:
            import yaml  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError(
                "GantryConfig.from_yaml() requires pyyaml: pip install pyyaml"
            ) from exc

        with open(path) as fh:
            data = yaml.safe_load(fh) or {}
        return cls(**data)

    @classmethod
    def from_env(cls, prefix: str = "CLAW_") -> GantryConfig:
        """Load config from environment variables.

        Each field maps to ``{prefix}{FIELD_NAME_UPPERCASE}``.

        Example::

            CLAW_MAX_STEPS=30
            CLAW_WORKSPACE=/app
            CLAW_MEMORY=chroma
            CLAW_MEMORY_PERSIST_DIRECTORY=/data/memory
            CLAW_TELEMETRY_OTLP_ENDPOINT=http://collector:4317
            CLAW_ENABLE_SUSPENSION=true
            CLAW_GUARDRAIL_REQUIRES_APPROVAL=shell_run,file_delete
            CLAW_DESKTOP_MAX_RESOLUTION=1280,720
        """
        data: dict[str, Any] = {}
        p = prefix.upper()

        _int_fields = {"max_steps"}
        _float_fields = {"shell_timeout", "max_wall_seconds"}
        _bool_fields = {"browser_headless", "enable_suspension"}
        _list_fields = {"shell_allowed_commands", "guardrail_requires_approval"}
        # tuple[int, int] fields — parsed from "W,H" or "W H"
        _tuple_int2_fields = {"desktop_max_resolution"}

        for field_name in cls.model_fields:
            env_key = f"{p}{field_name.upper()}"
            raw = os.environ.get(env_key)
            if raw is None:
                continue
            if field_name in _bool_fields:
                data[field_name] = raw.lower() in ("1", "true", "yes")
            elif field_name in _int_fields:
                data[field_name] = int(raw)
            elif field_name in _float_fields:
                data[field_name] = float(raw)
            elif field_name in _list_fields:
                data[field_name] = [s.strip() for s in raw.split(",") if s.strip()]
            elif field_name in _tuple_int2_fields:
                sep = "," if "," in raw else " "
                parts = [s.strip() for s in raw.split(sep) if s.strip()]
                if len(parts) != 2:  # noqa: PLR2004
                    raise ValueError(
                        f"Environment variable {env_key} must be 'W,H' (e.g. '1280,720')."
                    )
                data[field_name] = (int(parts[0]), int(parts[1]))
            else:
                data[field_name] = raw

        return cls(**data)
