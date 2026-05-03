"""OpenTelemetry integration for gantrygraph agents.

Quick start::

    from gantrygraph.telemetry import OTelExporter

    exporter = OTelExporter(service_name="my-agent")
    agent = GantryEngine(llm=..., on_event=exporter.as_event_callback())

For OTLP export to Datadog / Grafana::

    exporter = OTelExporter(
        service_name="my-agent",
        otlp_endpoint="http://localhost:4317",
    )

Requires::

    pip install 'gantrygraph[telemetry]'
"""

from gantrygraph.telemetry.otel import OTelExporter

__all__ = ["OTelExporter"]
