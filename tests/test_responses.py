"""OpenAI /v1/responses compression tests — no real keys, no network (respx).

Covers: input-item compression (typed text blocks + function_call_output),
upstream selection + auth headers, non-stream buffering, SSE stream relay, the
no-input passthrough, and compression-disabled verbatim forwarding.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator

import httpx
import respx

from tests.conftest import BIG_JSON


def _asgi_client(app) -> httpx.AsyncClient:
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://testserver")


# SSE stream reused shape from the Responses API (output_text deltas).
SSE_EVENTS = [
    b"event: response.created\n",
    b'data: {"type":"response.created"}\n\n',
    b"event: response.output_text.delta\n",
    b'data: {"type":"response.output_text.delta","delta":"Hel"}\n\n',
    b'data: {"type":"response.output_text.delta","delta":"lo"}\n\n',
    b"event: response.completed\n",
    b"data: {}\n\n",
]
SSE_FULL = b"".join(SSE_EVENTS)


async def _sse_chunks() -> AsyncIterator[bytes]:
    for ev in SSE_EVENTS:
        yield ev


@respx.mock
async def test_responses_compresses_items_and_forwards_to_openai(app):
    respx.routes.clear()
    route = respx.post("https://openai.test/v1/responses").mock(
        return_value=httpx.Response(200, json={"id": "resp_1", "object": "response"})
    )

    request_body = {
        "model": "gpt-4o",
        "input": [
            {"role": "user", "content": [{"type": "input_text", "text": "hi"}]},
            {"role": "user", "content": [{"type": "input_text", "text": BIG_JSON}]},
            {"type": "function_call_output", "call_id": "c1", "output": BIG_JSON},
            {"type": "reasoning", "summary": []},
        ],
    }

    async with _asgi_client(app) as client:
        resp = await client.post(
            "/v1/responses",
            json=request_body,
            headers={"authorization": "Bearer sk-openai-test"},
        )

    assert resp.status_code == 200
    assert resp.json()["id"] == "resp_1"

    # Upstream selection + auth preserved.
    assert route.called
    sent = route.calls.last.request
    assert str(sent.url) == "https://openai.test/v1/responses"
    assert sent.headers["authorization"] == "Bearer sk-openai-test"

    forwarded = json.loads(sent.content)
    # The big input_text block is compressed but keeps its Responses-API type.
    big_block = forwarded["input"][1]["content"][0]
    assert big_block["type"] == "input_text"
    assert len(big_block["text"]) < len(BIG_JSON)
    assert "__tokenslim_ccr__" in big_block["text"]
    # The function_call_output's output blob is compressed too, call_id intact.
    fco = forwarded["input"][2]
    assert fco["call_id"] == "c1"
    assert len(fco["output"]) < len(BIG_JSON)
    # Non-text items and non-input fields are forwarded untouched.
    assert forwarded["input"][3] == {"type": "reasoning", "summary": []}
    assert forwarded["model"] == "gpt-4o"


@respx.mock
async def test_responses_string_input_is_compressed(app):
    respx.routes.clear()
    route = respx.post("https://openai.test/v1/responses").mock(
        return_value=httpx.Response(200, json={"id": "resp_2"})
    )
    body = {"model": "gpt-4o", "input": BIG_JSON}
    async with _asgi_client(app) as client:
        resp = await client.post("/v1/responses", json=body)
    assert resp.status_code == 200
    forwarded = json.loads(route.calls.last.request.content)
    assert len(forwarded["input"]) < len(BIG_JSON)
    assert "__tokenslim_ccr__" in forwarded["input"]


@respx.mock
async def test_responses_stream_relays_sse_chunks_intact(app):
    respx.routes.clear()
    respx.post("https://openai.test/v1/responses").mock(
        return_value=httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=_sse_chunks(),
        )
    )

    body = {
        "model": "gpt-4o",
        "stream": True,
        "input": [{"role": "user", "content": [{"type": "input_text", "text": BIG_JSON}]}],
    }

    received = bytearray()
    async with _asgi_client(app) as client:
        async with client.stream("POST", "/v1/responses", json=body) as resp:
            assert resp.status_code == 200
            assert resp.headers["content-type"].startswith("text/event-stream")
            async for chunk in resp.aiter_raw():
                received.extend(chunk)

    assert bytes(received) == SSE_FULL
    assert received.count(b"\n\n") == 4


@respx.mock
async def test_responses_non_stream_is_buffered(app):
    respx.routes.clear()
    respx.post("https://openai.test/v1/responses").mock(
        return_value=httpx.Response(200, json={"id": "resp_3"})
    )
    body = {"model": "gpt-4o", "input": [{"type": "function_call_output", "output": BIG_JSON}]}
    async with _asgi_client(app) as client:
        resp = await client.post("/v1/responses", json=body)
    assert resp.status_code == 200
    assert resp.json()["id"] == "resp_3"
    assert "text/event-stream" not in resp.headers.get("content-type", "")


@respx.mock
async def test_responses_without_input_is_forwarded_unchanged(app):
    respx.routes.clear()
    route = respx.post("https://openai.test/v1/responses").mock(
        return_value=httpx.Response(200, json={"ok": True})
    )
    body = {"model": "gpt-4o", "instructions": "be terse"}
    async with _asgi_client(app) as client:
        resp = await client.post("/v1/responses", json=body)
    assert resp.status_code == 200
    forwarded = json.loads(route.calls.last.request.content)
    assert forwarded == body


@respx.mock
async def test_responses_upstream_failure_maps_to_502(app):
    respx.routes.clear()
    respx.post("https://openai.test/v1/responses").mock(side_effect=httpx.ConnectError("boom"))
    body = {"model": "gpt-4o", "input": "hi"}
    async with _asgi_client(app) as client:
        resp = await client.post("/v1/responses", json=body)
    assert resp.status_code == 502
    assert resp.json()["error"]["type"] == "tokenslim_proxy_error"


async def test_responses_compression_disabled_forwards_verbatim(config):
    from dataclasses import replace

    from tokenslim_proxy import create_app

    disabled = create_app(replace(config, compression_enabled=False))
    with respx.mock:
        route = respx.post("https://openai.test/v1/responses").mock(
            return_value=httpx.Response(200, json={"ok": True})
        )
        body = {
            "model": "gpt-4o",
            "input": [{"role": "user", "content": [{"type": "input_text", "text": BIG_JSON}]}],
        }
        async with _asgi_client(disabled) as client:
            await client.post("/v1/responses", json=body)
    forwarded = json.loads(route.calls.last.request.content)
    assert forwarded["input"][0]["content"][0]["text"] == BIG_JSON
