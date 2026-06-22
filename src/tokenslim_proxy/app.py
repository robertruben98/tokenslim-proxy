"""FastAPI application: compress chat bodies, forward to the real upstream.

Routes (P0 foundation):
    POST /v1/messages           -> Anthropic, compressed
    POST /v1/chat/completions   -> OpenAI, compressed
    GET  /healthz               -> liveness
    GET  /healthz/upstream      -> upstream config readiness
    GET  /metrics               -> Prometheus text exposition

Deferred to later milestones (issues left open): SSE streaming passthrough,
cache-prefix stabilization, /v1/responses, Bedrock, Vertex, and `wrap`.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import httpx
from fastapi import FastAPI, Request, Response

from .compression import compress_messages_body
from .config import ProxyConfig
from .metrics import Metrics
from .upstream import filter_request_headers, filter_response_headers


def _json_error(status: int, message: str) -> Response:
    return Response(
        content=json.dumps({"error": {"type": "tokenslim_proxy_error", "message": message}}),
        status_code=status,
        media_type="application/json",
    )


def create_app(config: ProxyConfig | None = None) -> FastAPI:
    """Build the proxy app. A shared :class:`httpx.AsyncClient` is opened on
    startup and closed on shutdown via the lifespan handler."""
    cfg = config or ProxyConfig.from_env()
    metrics = Metrics()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        try:
            yield
        finally:
            client: httpx.AsyncClient | None = getattr(app.state, "client", None)
            if client is not None:
                await client.aclose()

    app = FastAPI(title="tokenslim-proxy", version="0.0.1", lifespan=lifespan)
    app.state.config = cfg
    app.state.metrics = metrics
    app.state.client = None

    def _get_client() -> httpx.AsyncClient:
        """Lazily open the shared upstream client.

        Created on first use rather than in the lifespan handler so the app
        works under ASGI transports (e.g. tests) that don't fire lifespan.
        """
        if app.state.client is None:
            app.state.client = httpx.AsyncClient(timeout=cfg.upstream_timeout)
        return app.state.client

    async def _proxy_chat(request: Request, *, upstream_base: str, path: str) -> Response:
        """Shared body: read JSON, compress messages, forward, relay response."""
        raw = await request.body()
        try:
            body: dict[str, Any] = json.loads(raw) if raw else {}
        except json.JSONDecodeError:
            return _json_error(400, "Request body is not valid JSON.")
        if not isinstance(body, dict):
            return _json_error(400, "Request body must be a JSON object.")

        outcome = compress_messages_body(body, enabled=cfg.compression_enabled)
        metrics.record(orig=outcome.orig_tokens, new=outcome.new_tokens)

        forward_body = json.dumps(outcome.body).encode("utf-8")
        headers = filter_request_headers(request.headers)
        headers["content-type"] = "application/json"

        client = _get_client()
        url = f"{upstream_base}{path}"
        try:
            upstream = await client.post(url, content=forward_body, headers=headers)
        except httpx.RequestError as exc:
            return _json_error(502, f"Upstream request failed: {exc}")

        return Response(
            content=upstream.content,
            status_code=upstream.status_code,
            headers=filter_response_headers(upstream.headers),
            media_type=upstream.headers.get("content-type"),
        )

    @app.post("/v1/messages")
    async def anthropic_messages(request: Request) -> Response:  # noqa: D401
        """Anthropic /v1/messages — compress and forward to api.anthropic.com."""
        return await _proxy_chat(request, upstream_base=cfg.anthropic_base, path="/v1/messages")

    @app.post("/v1/chat/completions")
    async def openai_chat(request: Request) -> Response:  # noqa: D401
        """OpenAI /v1/chat/completions — compress and forward to api.openai.com."""
        return await _proxy_chat(
            request, upstream_base=cfg.openai_base, path="/v1/chat/completions"
        )

    @app.get("/healthz")
    async def healthz() -> dict[str, Any]:
        """Liveness probe."""
        return {"status": "ok", "service": "tokenslim-proxy", "version": "0.0.1"}

    @app.get("/healthz/upstream")
    async def healthz_upstream() -> dict[str, Any]:
        """Report resolved upstream configuration (no network call)."""
        return {
            "status": "ok",
            "anthropic_base": cfg.anthropic_base,
            "openai_base": cfg.openai_base,
            "compression_enabled": cfg.compression_enabled,
        }

    @app.get("/metrics")
    async def prometheus_metrics() -> Response:
        """Prometheus text-format scrape of cumulative compression counters."""
        return Response(content=metrics.render(), media_type="text/plain; version=0.0.4")

    return app


# Module-level app for `uvicorn tokenslim_proxy.app:app`.
app = create_app()
