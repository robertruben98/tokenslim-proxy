"""FastAPI application: compress chat bodies, forward to the real upstream.

Routes:
    POST /v1/messages           -> Anthropic, compressed (streams when stream=true)
    POST /v1/chat/completions   -> OpenAI, compressed (streams when stream=true)
    POST /v1/responses          -> OpenAI Responses API, compressed (streams too)
    POST /v1/vertex/messages    -> Vertex AI (Anthropic), compressed, ADC auth
    GET  /healthz               -> liveness
    GET  /healthz/upstream      -> upstream config readiness
    GET  /metrics               -> Prometheus text exposition

M2 adds SSE streaming passthrough and cache-prefix stabilization. The Vertex
route reuses the Anthropic message compression and attaches a GCP ADC bearer
token. Deferred (issue left open): the Bedrock (SigV4) native route.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import httpx
from fastapi import FastAPI, Request, Response
from fastapi.responses import StreamingResponse

from .cache import add_anthropic_cache_breakpoint
from .compression import compress_messages_body
from .config import ProxyConfig
from .metrics import Metrics
from .responses import compress_responses_body
from .upstream import filter_request_headers, filter_response_headers
from .vertex import GoogleADCTokenProvider, TokenProvider, VertexAuthError, rawpredict_path


def _json_error(status: int, message: str) -> Response:
    return Response(
        content=json.dumps({"error": {"type": "tokenslim_proxy_error", "message": message}}),
        status_code=status,
        media_type="application/json",
    )


def create_app(
    config: ProxyConfig | None = None, *, vertex_token_provider: TokenProvider | None = None
) -> FastAPI:
    """Build the proxy app. A shared :class:`httpx.AsyncClient` is opened on
    startup and closed on shutdown via the lifespan handler.

    ``vertex_token_provider`` is an injectable zero-arg callable returning a GCP
    bearer token for the Vertex route; it defaults to a lazy Application
    Default Credentials provider (tests pass a fake to avoid real credentials).
    """
    cfg = config or ProxyConfig.from_env()
    metrics = Metrics()
    vertex_token: TokenProvider = vertex_token_provider or GoogleADCTokenProvider()

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

    async def _read_body(request: Request) -> dict[str, Any] | Response:
        """Parse the JSON request body, or return a 400 ``Response`` on error."""
        raw = await request.body()
        try:
            body: Any = json.loads(raw) if raw else {}
        except json.JSONDecodeError:
            return _json_error(400, "Request body is not valid JSON.")
        if not isinstance(body, dict):
            return _json_error(400, "Request body must be a JSON object.")
        return body

    async def _forward(
        request: Request,
        *,
        out_body: dict[str, Any],
        upstream_base: str,
        path: str,
        extra_headers: dict[str, str] | None = None,
    ) -> Response:
        """Forward a (compressed) body to ``upstream_base + path``.

        Streams the upstream response chunk-by-chunk when the client requested
        ``"stream": true``; otherwise buffers and returns it whole. Auth and
        other client headers are preserved by :func:`filter_request_headers`;
        ``extra_headers`` (e.g. a minted Vertex bearer token) override them.
        """
        forward_body = json.dumps(out_body).encode("utf-8")
        headers = filter_request_headers(dict(request.headers))
        headers["content-type"] = "application/json"
        if extra_headers:
            headers.update(extra_headers)

        client = _get_client()
        url = f"{upstream_base}{path}"

        if bool(out_body.get("stream")):
            return await _stream_upstream(client, url, forward_body, headers)

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

    async def _proxy_chat(
        request: Request, *, upstream_base: str, path: str, anthropic: bool = False
    ) -> Response:
        """Read JSON, compress the ``messages`` array, forward, relay response."""
        body = await _read_body(request)
        if isinstance(body, Response):
            return body

        outcome = compress_messages_body(
            body,
            enabled=cfg.compression_enabled,
            stable_prefix=cfg.cache_prefix_stable,
        )
        metrics.record(orig=outcome.orig_tokens, new=outcome.new_tokens)

        out_body = outcome.body
        if anthropic and cfg.anthropic_cache_breakpoint:
            out_body = add_anthropic_cache_breakpoint(out_body)

        return await _forward(request, out_body=out_body, upstream_base=upstream_base, path=path)

    async def _proxy_responses(request: Request) -> Response:
        """Read JSON, compress the Responses-API ``input``, forward, relay."""
        body = await _read_body(request)
        if isinstance(body, Response):
            return body

        outcome = compress_responses_body(body, enabled=cfg.compression_enabled)
        metrics.record(orig=outcome.orig_tokens, new=outcome.new_tokens)

        return await _forward(
            request,
            out_body=outcome.body,
            upstream_base=cfg.openai_base,
            path="/v1/responses",
        )

    async def _proxy_vertex(request: Request) -> Response:
        """Compress an Anthropic-style body and forward to Vertex AI with ADC.

        Vertex takes the ``model`` in the URL and an ``anthropic_version`` in
        the body, so those are rewritten before forwarding. The client's auth
        header is replaced with a freshly minted Application Default Credentials
        bearer token (its own token does not sign the body, so recompressing is
        safe). The route is disabled with a 500 hint until a project is set.
        """
        if not cfg.vertex_project:
            return _json_error(
                500,
                "Vertex route is not configured. Set TOKENSLIM_PROXY_VERTEX_PROJECT "
                "(and optionally TOKENSLIM_PROXY_VERTEX_LOCATION).",
            )

        body = await _read_body(request)
        if isinstance(body, Response):
            return body

        model = body.get("model")
        if not isinstance(model, str) or not model:
            return _json_error(400, "Vertex request body must include a string 'model'.")

        outcome = compress_messages_body(
            body,
            enabled=cfg.compression_enabled,
            stable_prefix=cfg.cache_prefix_stable,
        )
        metrics.record(orig=outcome.orig_tokens, new=outcome.new_tokens)

        out_body = dict(outcome.body)
        out_body.pop("model", None)  # model is carried in the URL for Vertex
        out_body.setdefault("anthropic_version", cfg.vertex_anthropic_version)
        if cfg.anthropic_cache_breakpoint:
            out_body = add_anthropic_cache_breakpoint(out_body)

        try:
            token = await asyncio.to_thread(vertex_token)
        except VertexAuthError as exc:
            return _json_error(500, str(exc))

        path = rawpredict_path(
            cfg.vertex_project,
            cfg.vertex_location,
            model,
            stream=bool(out_body.get("stream")),
        )
        return await _forward(
            request,
            out_body=out_body,
            upstream_base=cfg.vertex_endpoint_base,
            path=path,
            extra_headers={"authorization": f"Bearer {token}"},
        )

    async def _stream_upstream(
        client: httpx.AsyncClient, url: str, forward_body: bytes, headers: dict[str, str]
    ) -> Response:
        """Open a streaming upstream POST and relay raw bytes as they arrive.

        The SSE event stream is forwarded verbatim chunk-by-chunk (no buffering
        of the whole body and no re-parsing), so ``event:`` / ``data:`` framing
        is preserved byte-for-byte. The upstream context stays open for the life
        of the generator and is closed when the client stream ends.
        """
        cm = client.stream("POST", url, content=forward_body, headers=headers)
        try:
            upstream = await cm.__aenter__()
        except httpx.RequestError as exc:
            return _json_error(502, f"Upstream request failed: {exc}")

        async def body_iter() -> AsyncIterator[bytes]:
            try:
                async for chunk in upstream.aiter_raw():
                    if chunk:
                        yield chunk
            finally:
                await cm.__aexit__(None, None, None)

        return StreamingResponse(
            body_iter(),
            status_code=upstream.status_code,
            headers=filter_response_headers(upstream.headers),
            media_type=upstream.headers.get("content-type", "text/event-stream"),
        )

    @app.post("/v1/messages")
    async def anthropic_messages(request: Request) -> Response:  # noqa: D401
        """Anthropic /v1/messages — compress and forward to api.anthropic.com."""
        return await _proxy_chat(
            request, upstream_base=cfg.anthropic_base, path="/v1/messages", anthropic=True
        )

    @app.post("/v1/chat/completions")
    async def openai_chat(request: Request) -> Response:  # noqa: D401
        """OpenAI /v1/chat/completions — compress and forward to api.openai.com."""
        return await _proxy_chat(
            request, upstream_base=cfg.openai_base, path="/v1/chat/completions"
        )

    @app.post("/v1/responses")
    async def openai_responses(request: Request) -> Response:  # noqa: D401
        """OpenAI /v1/responses — compress the ``input`` and forward to OpenAI."""
        return await _proxy_responses(request)

    @app.post("/v1/vertex/messages")
    async def vertex_messages(request: Request) -> Response:  # noqa: D401
        """Vertex AI (Anthropic) — compress and forward with a GCP ADC token."""
        return await _proxy_vertex(request)

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
            "cache_prefix_stable": cfg.cache_prefix_stable,
            "anthropic_cache_breakpoint": cfg.anthropic_cache_breakpoint,
            "vertex_enabled": bool(cfg.vertex_project),
            "vertex_base": cfg.vertex_endpoint_base,
            "vertex_location": cfg.vertex_location,
        }

    @app.get("/metrics")
    async def prometheus_metrics() -> Response:
        """Prometheus text-format scrape of cumulative compression counters."""
        return Response(content=metrics.render(), media_type="text/plain; version=0.0.4")

    return app


# Module-level app for `uvicorn tokenslim_proxy.app:app`.
app = create_app()
