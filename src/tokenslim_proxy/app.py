"""FastAPI application: compress chat bodies, forward to the real upstream.

Routes:
    POST /v1/messages           -> Anthropic, compressed (streams when stream=true)
    POST /v1/chat/completions   -> OpenAI, compressed (streams when stream=true)
    POST /v1/responses          -> OpenAI Responses API, compressed (streams too)
    POST /model/{id}/invoke[-with-response-stream]
                                -> AWS Bedrock, compressed + SigV4-signed
    GET  /healthz               -> liveness
    GET  /healthz/upstream      -> upstream config readiness
    GET  /metrics               -> Prometheus text exposition

M2 adds SSE streaming passthrough and cache-prefix stabilization; the native
Bedrock route (SigV4) lands here too. Deferred (issue left open): Vertex.
"""

from __future__ import annotations

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
from .sigv4 import AwsCredentials, sign_request
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

    async def _send_upstream(
        url: str, forward_body: bytes, headers: dict[str, str], *, stream: bool
    ) -> Response:
        """POST a prepared body upstream and relay the reply.

        Streams the response chunk-by-chunk when ``stream`` is set (raw bytes, so
        both OpenAI/Anthropic SSE and Bedrock's binary EventStream framing survive
        intact); otherwise buffers it and returns it whole. Callers own the
        headers, so this works for both header-relayed and SigV4-signed requests.
        """
        client = _get_client()
        if stream:
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

    async def _forward(
        request: Request, *, out_body: dict[str, Any], upstream_base: str, path: str
    ) -> Response:
        """Forward a (compressed) body to ``upstream_base + path``.

        Streams the upstream response chunk-by-chunk when the client requested
        ``"stream": true``; otherwise buffers and returns it whole. Auth and
        other client headers are preserved by :func:`filter_request_headers`.
        """
        forward_body = json.dumps(out_body).encode("utf-8")
        headers = filter_request_headers(dict(request.headers))
        headers["content-type"] = "application/json"
        return await _send_upstream(
            f"{upstream_base}{path}", forward_body, headers, stream=bool(out_body.get("stream"))
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

    async def _proxy_bedrock(request: Request, *, model_id: str, stream: bool) -> Response:
        """Compress an Anthropic-on-Bedrock body, SigV4-sign it, forward to Bedrock.

        Bedrock's ``InvokeModel`` body carries the same ``messages`` array as
        ``/v1/messages`` (the model id lives in the URL, not the body), so the
        same compression applies. Unlike the header-relaying routes, Bedrock
        authenticates with SigV4, so the proxy holds the AWS credentials and
        signs the **compressed** body itself (the client sends no auth header).
        ``stream`` selects the ``invoke-with-response-stream`` action and relays
        Bedrock's binary EventStream response verbatim.
        """
        body = await _read_body(request)
        if isinstance(body, Response):
            return body

        credentials = AwsCredentials.from_env()
        if credentials is None:
            return _json_error(
                500,
                "AWS credentials not configured "
                "(set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY).",
            )

        outcome = compress_messages_body(
            body, enabled=cfg.compression_enabled, stable_prefix=cfg.cache_prefix_stable
        )
        metrics.record(orig=outcome.orig_tokens, new=outcome.new_tokens)

        forward_body = json.dumps(outcome.body).encode("utf-8")
        action = "invoke-with-response-stream" if stream else "invoke"
        url = f"{cfg.bedrock_endpoint()}/model/{model_id}/{action}"
        accept = "application/vnd.amazon.eventstream" if stream else "application/json"
        headers = sign_request(
            method="POST",
            url=url,
            payload=forward_body,
            region=cfg.aws_region,
            service="bedrock",
            credentials=credentials,
            headers={"content-type": "application/json", "accept": accept},
        )
        return await _send_upstream(url, forward_body, headers, stream=stream)

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

    @app.post("/model/{model_id:path}/invoke")
    async def bedrock_invoke(model_id: str, request: Request) -> Response:  # noqa: D401
        """Bedrock InvokeModel — compress, SigV4-sign, forward (buffered)."""
        return await _proxy_bedrock(request, model_id=model_id, stream=False)

    @app.post("/model/{model_id:path}/invoke-with-response-stream")
    async def bedrock_invoke_stream(model_id: str, request: Request) -> Response:  # noqa: D401
        """Bedrock InvokeModelWithResponseStream — compress, sign, relay EventStream."""
        return await _proxy_bedrock(request, model_id=model_id, stream=True)

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
            "bedrock_base": cfg.bedrock_endpoint(),
            "aws_region": cfg.aws_region,
            "compression_enabled": cfg.compression_enabled,
            "cache_prefix_stable": cfg.cache_prefix_stable,
            "anthropic_cache_breakpoint": cfg.anthropic_cache_breakpoint,
        }

    @app.get("/metrics")
    async def prometheus_metrics() -> Response:
        """Prometheus text-format scrape of cumulative compression counters."""
        return Response(content=metrics.render(), media_type="text/plain; version=0.0.4")

    return app


# Module-level app for `uvicorn tokenslim_proxy.app:app`.
app = create_app()
