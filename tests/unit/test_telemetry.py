"""Unit tests for the gantrygraph.telemetry module."""

from __future__ import annotations

import pytest

try:
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    _HAS_OTEL = True
except ImportError:
    _HAS_OTEL = False

pytestmark = pytest.mark.skipif(not _HAS_OTEL, reason="Requires [telemetry] extra")


def _make_exporter_with_spy():
    """Returns (OTelExporter, InMemorySpanExporter) wired together."""
    from opentelemetry import trace as trace_api
    from opentelemetry.sdk.resources import Resource

    from gantrygraph.telemetry.otel import OTelExporter

    spy = InMemorySpanExporter()
    provider = TracerProvider(resource=Resource.create({"service.name": "test"}))
    provider.add_span_processor(SimpleSpanProcessor(spy))
    trace_api.set_tracer_provider(provider)

    exporter = OTelExporter.__new__(OTelExporter)
    exporter._provider = provider
    exporter._tracer = provider.get_tracer("test")
    return exporter, spy


# ── OTelExporter construction ─────────────────────────────────────────────────


def test_otel_exporter_constructs_without_endpoint() -> None:
    from gantrygraph.telemetry.otel import OTelExporter

    exporter = OTelExporter(service_name="test-agent")
    assert exporter is not None


def test_otel_exporter_raises_without_extra() -> None:
    """Import guard: if opentelemetry is missing, ImportError with install hint."""
    import importlib
    import sys

    # Temporarily hide the opentelemetry module to simulate missing extra
    otel_modules = {k: v for k, v in sys.modules.items() if "opentelemetry" in k}
    for k in otel_modules:
        sys.modules[k] = None  # type: ignore[assignment]

    # Reload the module to re-evaluate the import guard
    import gantrygraph.telemetry.otel as otel_mod

    importlib.reload(otel_mod)

    try:
        if not otel_mod._HAS_OTEL:
            with pytest.raises(ImportError, match="gantrygraph\\[telemetry\\]"):
                otel_mod.OTelExporter()
    finally:
        # Restore modules
        for k, v in otel_modules.items():
            sys.modules[k] = v
        importlib.reload(otel_mod)


# ── as_event_callback ─────────────────────────────────────────────────────────


def test_callback_creates_task_span_on_observe() -> None:
    from gantrygraph.core.events import GantryEvent

    exporter, spy = _make_exporter_with_spy()
    cb = exporter.as_event_callback()

    cb(GantryEvent("observe", 0, {"width": 1920, "height": 1080}))
    cb(GantryEvent("done", 0, {}))

    spans = spy.get_finished_spans()
    names = [s.name for s in spans]
    assert "gantrygraph.task" in names


def test_callback_creates_tool_span_on_act() -> None:
    from gantrygraph.core.events import GantryEvent

    exporter, spy = _make_exporter_with_spy()
    cb = exporter.as_event_callback()

    cb(GantryEvent("observe", 0, {}))
    cb(GantryEvent("act", 1, {"tools_executed": ["shell_run"]}))
    cb(GantryEvent("done", 1, {}))

    spans = spy.get_finished_spans()
    names = [s.name for s in spans]
    assert "gantrygraph.tool.shell_run" in names


def test_callback_closes_all_spans_on_done() -> None:
    from gantrygraph.core.events import GantryEvent

    exporter, spy = _make_exporter_with_spy()
    cb = exporter.as_event_callback()

    cb(GantryEvent("observe", 0, {}))
    cb(GantryEvent("act", 1, {"tools_executed": ["file_read", "file_write"]}))
    cb(GantryEvent("done", 2, {}))

    spans = spy.get_finished_spans()
    # All spans must be finished (end_time is set)
    assert all(s.end_time is not None for s in spans)


def test_callback_error_event_sets_error_status() -> None:
    from opentelemetry.trace import StatusCode

    from gantrygraph.core.events import GantryEvent

    exporter, spy = _make_exporter_with_spy()
    cb = exporter.as_event_callback()

    cb(GantryEvent("observe", 0, {}))
    cb(GantryEvent("error", 1, {"error": "tool crashed"}))

    spans = spy.get_finished_spans()
    task_spans = [s for s in spans if s.name == "gantrygraph.task"]
    assert len(task_spans) == 1
    assert task_spans[0].status.status_code == StatusCode.ERROR


def test_force_flush_does_not_raise() -> None:
    from gantrygraph.telemetry.otel import OTelExporter

    exporter = OTelExporter(service_name="flush-test")
    exporter.force_flush(timeout_ms=100)  # should not raise


# ── Engine + OTel integration ─────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_engine_with_otel_callback_records_spans() -> None:
    from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
    from langchain_core.messages import AIMessage

    from gantrygraph import GantryEngine

    exporter, spy = _make_exporter_with_spy()
    cb = exporter.as_event_callback()

    llm = FakeMessagesListChatModel(responses=[AIMessage(content="All done.")])
    agent = GantryEngine(llm=llm, on_event=cb, max_steps=5)
    result = await agent.arun("Test task")

    assert isinstance(result, str)
    spans = spy.get_finished_spans()
    # At minimum: the gantrygraph.task root span should be recorded
    assert any(s.name == "gantrygraph.task" for s in spans)
