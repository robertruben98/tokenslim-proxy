"""M2: SSE streaming passthrough tests (mocked httpx, no network/keys)."""

from __future__ import annotations

from collections.abc import AsyncIterator

import httpx
import respx

from tests.conftest import BIG_JSON

# A representative Anthropic-style SSE event stream, split so the upstream
# emits it as several chunks (event framing spans chunk boundaries).
SSE_EVENTS = [
    b"event: message_start\n",
    b'data: {"type":"message_start"}\n\n',
    b"event: content_block_delta\n",
    b'data: {"type":"content_block_delta","delta":{"text":"Hel"}}\n\n',
    b'data: {"type":"content_block_delta","delta":{"text":"lo"}}\n\n',
    b"event: message_stop\n",
    b"data: {}\n\n",
]
SSE_FULL = b"".join(SSE_EVENTS)


async def _sse_chunks() -> AsyncIterator[bytes]:
    """Yield the SSE body as separate chunks, like a real streaming upstream."""
    for ev in SSE_EVENTS:
        yield ev


def _asgi_client(app) -> httpx.AsyncClient:
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://testserver")


@respx.mock
async def test_stream_relays_sse_chunks_intact(app):
    respx.routes.clear()
    respx.post("https://anthropic.test/v1/messages").mock(
        return_value=httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=_sse_chunks(),
        )
    )

    body = {
        "model": "claude-sonnet-4-6",
        "stream": True,
        "messages": [{"role": "user", "content": BIG_JSON}],
    }

    received = bytearray()
    async with _asgi_client(app) as client:
        async with client.stream(
            "POST", "/v1/messages", json=body, headers={"x-api-key": "k"}
        ) as resp:
            assert resp.status_code == 200
            assert resp.headers["content-type"].startswith("text/event-stream")
            async for chunk in resp.aiter_raw():
                received.extend(chunk)

    # The concatenated client stream equals the upstream events byte-for-byte,
    # so event:/data: framing is preserved.
    assert bytes(received) == SSE_FULL
    # And it really arrived as distinct SSE events (framing not collapsed).
    assert received.count(b"\n\n") == 4


@respx.mock
async def test_non_stream_path_is_buffered_as_before(app):
    respx.routes.clear()
    respx.post("https://openai.test/v1/chat/completions").mock(
        return_value=httpx.Response(200, json={"id": "chatcmpl-1"})
    )
    body = {
        "model": "gpt-4o",
        "messages": [{"role": "tool", "content": BIG_JSON}],
        # no "stream" key -> buffered path
    }
    async with _asgi_client(app) as client:
        resp = await client.post("/v1/chat/completions", json=body)
    assert resp.status_code == 200
    assert resp.json()["id"] == "chatcmpl-1"
    assert "text/event-stream" not in resp.headers.get("content-type", "")


@respx.mock
async def test_stream_upstream_error_maps_to_502(app):
    respx.routes.clear()
    respx.post("https://anthropic.test/v1/messages").mock(side_effect=httpx.ConnectError("boom"))
    body = {
        "model": "claude-sonnet-4-6",
        "stream": True,
        "messages": [{"role": "user", "content": "hi"}],
    }
    async with _asgi_client(app) as client:
        resp = await client.post("/v1/messages", json=body)
    assert resp.status_code == 502
