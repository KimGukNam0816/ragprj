"""FastAPI's own OTel spans would become the parent of every shipping_rag_query root once Langfuse sets the
global tracer provider, so the root stops being a root in Langfuse. The app must not open request spans."""
from fastapi.testclient import TestClient
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter


def test_requests_open_no_fastapi_span_when_a_global_tracer_provider_exists(monkeypatch):
    from backend.app import main
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr('opentelemetry.trace.get_tracer_provider', lambda: provider)   # what Langfuse's client does globally
    with TestClient(main.app) as client:
        assert client.get('/health').status_code == 200
    assert exporter.get_finished_spans() == ()
