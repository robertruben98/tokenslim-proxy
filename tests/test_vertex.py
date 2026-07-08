"""Google Vertex AI route tests — no real ADC or network (respx + fake token)."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator

import httpx
import pytest
import respx

from tests.conftest import BIG_JSON
from tokenslim_proxy import ProxyConfig, create_app
from tokenslim_proxy.vertex import VertexAuthError, rawpredict_path, vertex_base_url

VERTEX_BASE = "https://vertex.test"
FAKE_TOKEN = "ya29.fake-adc-token"


def _asgi_client(app) -> httpx.AsyncClient:
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://testserver")


def _url(model: str, *, stream: bool = False) -> str:
    """Full expected Vertex URL (path format is pinned by ``test_rawpredict_path``)."""
    return VERTEX_BASE + rawpredict_path("my-proj", "us-east5", model, stream=stream)


def _vertex_app(*, token_provider=lambda: FAKE_TOKEN, **cfg_overrides):
    cfg = ProxyConfig(
        vertex_project="my-proj",
        vertex_location="us-east5",
        vertex_base=VERTEX_BASE,
        **cfg_overrides,
    )
    return create_app(cfg, vertex_token_provider=token_provider)


# --- URL / config helpers (pure, no app) ------------------------------------


def test_rawpredict_path_selects_verb_and_encodes_model():
    assert rawpredict_path("p", "us-east5", "claude-sonnet-4-6") == (
        "/v1/projects/p/locations/us-east5/publishers/anthropic/models/"
        "claude-sonnet-4-6:rawPredict"
    )
    # Streaming verb + an @version model id (kept intact).
    assert rawpredict_path("p", "us-east5", "claude-3-7-sonnet@20250219", stream=True) == (
        "/v1/projects/p/locations/us-east5/publishers/anthropic/models/"
        "claude-3-7-sonnet@20250219:streamRawPredict"
    )


def test_vertex_endpoint_base_derives_from_location():
    assert ProxyConfig(vertex_location="europe-west1").vertex_endpoint_base == vertex_base_url(
        "europe-west1"
    )
    # Explicit override wins.
    assert ProxyConfig(vertex_base="https://x.test").vertex_endpoint_base == "https://x.test"


# --- Route behaviour --------------------------------------------------------


@respx.mock
async def test_vertex_forwards_compressed_body_with_adc_token():
    respx.routes.clear()
    url = _url("claude-sonnet-4-6")
    route = respx.post(url).mock(
        return_value=httpx.Response(200, json={"id": "msg_1", "role": "assistant"})
    )

    request_body = {
        "model": "claude-sonnet-4-6",
        "max_tokens": 1024,
        "messages": [
            {"role": "user", "content": "summarize this"},
            {"role": "user", "content": BIG_JSON},
        ],
    }

    async with _asgi_client(_vertex_app()) as client:
        resp = await client.post("/v1/vertex/messages", json=request_body)

    assert resp.status_code == 200
    assert resp.json()["id"] == "msg_1"
    assert route.called

    sent = route.calls.last.request
    assert str(sent.url) == url
    # ADC bearer token is injected.
    assert sent.headers["authorization"] == f"Bearer {FAKE_TOKEN}"

    forwarded = json.loads(sent.content)
    # model moves to the URL; anthropic_version is added for Vertex.
    assert "model" not in forwarded
    assert forwarded["anthropic_version"] == "vertex-2023-10-16"
    assert forwarded["max_tokens"] == 1024
    # The big block was compressed.
    big_out = forwarded["messages"][1]["content"]
    assert len(big_out) < len(BIG_JSON)
    assert "__tokenslim_ccr__" in big_out


@respx.mock
async def test_vertex_preserves_client_anthropic_version():
    respx.routes.clear()
    url = _url("claude-sonnet-4-6")
    route = respx.post(url).mock(return_value=httpx.Response(200, json={"ok": True}))

    body = {
        "model": "claude-sonnet-4-6",
        "anthropic_version": "vertex-2099-01-01",
        "messages": [{"role": "user", "content": "hi"}],
    }
    async with _asgi_client(_vertex_app()) as client:
        resp = await client.post("/v1/vertex/messages", json=body)

    assert resp.status_code == 200
    forwarded = json.loads(route.calls.last.request.content)
    assert forwarded["anthropic_version"] == "vertex-2099-01-01"


@respx.mock
async def test_vertex_streaming_uses_stream_verb():
    sse = [b"event: message_start\n", b'data: {"type":"message_start"}\n\n', b"data: {}\n\n"]

    async def _chunks() -> AsyncIterator[bytes]:
        for ev in sse:
            yield ev

    respx.routes.clear()
    url = _url("claude-sonnet-4-6", stream=True)
    route = respx.post(url).mock(
        return_value=httpx.Response(
            200, headers={"content-type": "text/event-stream"}, content=_chunks()
        )
    )

    body = {
        "model": "claude-sonnet-4-6",
        "stream": True,
        "messages": [{"role": "user", "content": BIG_JSON}],
    }

    received = bytearray()
    async with _asgi_client(_vertex_app()) as client:
        async with client.stream("POST", "/v1/vertex/messages", json=body) as resp:
            assert resp.status_code == 200
            assert resp.headers["content-type"].startswith("text/event-stream")
            async for chunk in resp.aiter_raw():
                received.extend(chunk)

    assert route.called
    assert bytes(received) == b"".join(sse)


async def test_vertex_disabled_without_project():
    # No project configured -> route returns a 500 config hint (no upstream call).
    app = create_app(ProxyConfig(), vertex_token_provider=lambda: FAKE_TOKEN)
    async with _asgi_client(app) as client:
        resp = await client.post(
            "/v1/vertex/messages",
            json={"model": "claude-sonnet-4-6", "messages": [{"role": "user", "content": "hi"}]},
        )
    assert resp.status_code == 500
    assert resp.json()["error"]["type"] == "tokenslim_proxy_error"


async def test_vertex_requires_model():
    async with _asgi_client(_vertex_app()) as client:
        resp = await client.post(
            "/v1/vertex/messages", json={"messages": [{"role": "user", "content": "hi"}]}
        )
    assert resp.status_code == 400


async def test_vertex_auth_error_maps_to_500():
    def _boom() -> str:
        raise VertexAuthError("no credentials")

    async with _asgi_client(_vertex_app(token_provider=_boom)) as client:
        resp = await client.post(
            "/v1/vertex/messages",
            json={"model": "claude-sonnet-4-6", "messages": [{"role": "user", "content": "hi"}]},
        )
    assert resp.status_code == 500
    assert "no credentials" in resp.json()["error"]["message"]


async def test_healthz_upstream_reports_vertex():
    async with _asgi_client(_vertex_app()) as client:
        resp = await client.get("/healthz/upstream")
    body = resp.json()
    assert body["vertex_enabled"] is True
    assert body["vertex_base"] == VERTEX_BASE
    assert body["vertex_location"] == "us-east5"


@pytest.mark.parametrize("stream", [False, True])
async def test_vertex_auth_error_before_upstream(stream):
    # An auth failure must short-circuit before any upstream call is attempted.
    def _boom() -> str:
        raise VertexAuthError("nope")

    with respx.mock:
        route = respx.route(url__regex=r".*").mock(return_value=httpx.Response(200))
        async with _asgi_client(_vertex_app(token_provider=_boom)) as client:
            resp = await client.post(
                "/v1/vertex/messages",
                json={
                    "model": "claude-sonnet-4-6",
                    "stream": stream,
                    "messages": [{"role": "user", "content": "hi"}],
                },
            )
    assert resp.status_code == 500
    assert not route.called
