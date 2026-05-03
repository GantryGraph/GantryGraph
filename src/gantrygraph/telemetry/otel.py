"""OpenTelemetry exporter that converts GantryEvents into OTel spans.

One root span ``gantrygraph.task`` wraps the full agent run.  Each tool execution
gets a child span ``gantrygraph.tool.<name>`` with step and tool attributes.

Requires the ``[telemetry]`` extra::

    pip install 'gantrygraph[telemetry]'

Compatible backends: Datadog, Grafana Tempo, Jaeger, Zipkin, or any
OTLP-compatible collector.  Pass ``otlp_endpoint=None`` to log spans to
stdout (useful during development).
"""
from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from gantrygraph.core.events import GantryEvent

try:
    from opentelemetry import trace
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
    from opentelemetry.trace import StatusCode

    _HAS_OTEL = True
except ImportError:
    _HAS_OTEL = False

_INSTALL_MSG = (
    "OTelExporter requires opentelemetry: "
    "pip install 'gantrygraph[telemetry]'"
)


class OTelExporter:
    """Converts ``GantryEvent`` callbacks into OpenTelemetry spans.

    Usage::

        exporter = OTelExporter(service_name="qa-agent")
        agent = GantryEngine(llm=..., on_event=exporter.as_event_callback())

    Args:
        service_name:   ``service.name`` attribute attached to all spans.
        otlp_endpoint:  gRPC endpoint for an OTLP collector
                        (e.g. ``"http://localhost:4317"`` for Grafana Alloy).
                        ``None`` prints spans to stdout via ConsoleSpanExporter.
    """

    def __init__(
        self,
        service_name: str = "gantry-agent",
        otlp_endpoint: str | None = None,
    ) -> None:
        if not _HAS_OTEL:
            raise ImportError(_INSTALL_MSG)

        resource = Resource.create({"service.name": service_name})
        provider = TracerProvider(resource=resource)

        if otlp_endpoint:
            try:
                from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (  # type: ignore[import-not-found]
                    OTLPSpanExporter,
                )
                provider.add_span_processor(
                    BatchSpanProcessor(OTLPSpanExporter(endpoint=otlp_endpoint))
                )
            except ImportError as exc:
                raise ImportError(
                    "OTLP export requires: "
                    "pip install opentelemetry-exporter-otlp-proto-grpc"
                ) from exc
        else:
            provider.add_span_processor(
                BatchSpanProcessor(ConsoleSpanExporter())
            )

        self._provider = provider
        self._tracer: Any = provider.get_tracer(service_name)

    def as_event_callback(self) -> Callable[[GantryEvent], None]:
        """Return a ``GantryEvent`` callback that creates OTel spans.

        Each call to the returned function is stateful — the same callback
        instance tracks the root span across the full run.  Do **not** share
        a single callback instance between concurrent agent runs.
        """
        tracer = self._tracer
        state: dict[str, Any] = {
            "task_span": None,
            "act_spans": {},  # step → Span
        }

        def _callback(event: GantryEvent) -> None:
            etype = event.event_type
            step = event.step

            if etype == "observe" and state["task_span"] is None:
                # Open the root span on the first observe event
                span = tracer.start_span("gantrygraph.task")
                span.set_attribute("step.start", step)
                state["task_span"] = span

            elif etype == "act":
                # One child span per executed tool
                for tool_name in event.data.get("tools_executed", []):
                    span = tracer.start_span(
                        f"gantrygraph.tool.{tool_name}",
                        context=trace.set_span_in_context(state["task_span"])
                        if state["task_span"]
                        else None,
                    )
                    span.set_attribute("tool.name", tool_name)
                    span.set_attribute("step", step)
                    state["act_spans"][step] = span

            elif etype == "think":
                # Close act span from previous step when LLM responds
                prev_span = state["act_spans"].pop(step - 1, None)
                if prev_span is not None:
                    prev_span.end()

            elif etype in ("done", "error"):
                # Close all open tool spans first
                for s in state["act_spans"].values():
                    s.end()
                state["act_spans"].clear()

                task_span: Any = state["task_span"]
                if task_span is not None:
                    task_span.set_attribute("step.end", step)
                    if etype == "error":
                        task_span.set_status(StatusCode.ERROR, event.data.get("error", ""))
                    else:
                        task_span.set_status(StatusCode.OK)
                    task_span.end()
                    state["task_span"] = None

        return _callback

    def force_flush(self, timeout_ms: int = 5_000) -> None:
        """Flush pending spans to the exporter (call before process exit)."""
        self._provider.force_flush(timeout_millis=timeout_ms)
