"""Proxy behavior tests — no real API keys, no network (respx intercepts httpx)."""

from __future__ import annotations

import json

import httpx
import pytest
import respx

from tests.conftest import BIG_JSON


def _asgi_client(app) -> httpx.AsyncClient:
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://testserver")


async def test_healthz(app):
    async with _asgi_client(app) as client:
        resp = await client.get("/healthz")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["service"] == "tokenslim-proxy"


async def test_healthz_upstream_reports_config(app):
    async with _asgi_client(app) as client:
        resp = await client.get("/healthz/upstream")
    assert resp.status_code == 200
    body = resp.json()
    assert body["anthropic_base"] == "https://anthropic.test"
    assert body["openai_base"] == "https://openai.test"
    assert body["compression_enabled"] is True


@respx.mock
async def test_anthropic_messages_forwards_compressed_body_to_anthropic(app):
    # respx must NOT intercept the in-process ASGI call (host=testserver); it
    # only intercepts the proxy's outbound call to the fake Anthropic host.
    respx.routes.clear()
    route = respx.post("https://anthropic.test/v1/messages").mock(
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

    async with _asgi_client(app) as client:
        resp = await client.post(
            "/v1/messages",
            json=request_body,
            headers={"x-api-key": "sk-ant-test", "anthropic-version": "2023-06-01"},
        )

    assert resp.status_code == 200
    assert resp.json()["id"] == "msg_1"

    # (b) upstream selection + headers: hit the Anthropic host with auth preserved.
    assert route.called
    sent = route.calls.last.request
    assert str(sent.url) == "https://anthropic.test/v1/messages"
    assert sent.headers["x-api-key"] == "sk-ant-test"
    assert sent.headers["anthropic-version"] == "2023-06-01"

    # (a) forwarded body is compressed: smaller and still valid JSON.
    forwarded = json.loads(sent.content)
    big_in = request_body["messages"][1]["content"]
    big_out = forwarded["messages"][1]["content"]
    assert len(big_out) < len(big_in)
    assert json.loads(big_out) is not None
    assert "__tokenslim_ccr__" in big_out  # CCR elision marker present
    # Non-message fields are forwarded untouched.
    assert forwarded["model"] == "claude-sonnet-4-6"
    assert forwarded["max_tokens"] == 1024


@respx.mock
async def test_openai_chat_forwards_to_openai_host(app):
    respx.routes.clear()
    route = respx.post("https://openai.test/v1/chat/completions").mock(
        return_value=httpx.Response(200, json={"id": "chatcmpl-1", "object": "chat.completion"})
    )

    request_body = {
        "model": "gpt-4o",
        "messages": [
            {"role": "system", "content": "you are helpful"},
            {"role": "tool", "tool_call_id": "c1", "content": BIG_JSON},
        ],
    }

    async with _asgi_client(app) as client:
        resp = await client.post(
            "/v1/chat/completions",
            json=request_body,
            headers={"authorization": "Bearer sk-openai-test"},
        )

    assert resp.status_code == 200
    assert route.called
    sent = route.calls.last.request
    assert str(sent.url) == "https://openai.test/v1/chat/completions"
    # Bearer auth preserved verbatim.
    assert sent.headers["authorization"] == "Bearer sk-openai-test"

    forwarded = json.loads(sent.content)
    tool_in = request_body["messages"][1]["content"]
    tool_out = forwarded["messages"][1]["content"]
    # Smaller, still valid JSON. (The core now elides via a CCR marker rather
    # than byte-exact minify, so the output no longer equals the input.)
    assert len(tool_out) < len(tool_in)
    assert json.loads(tool_out) is not None
    assert "__tokenslim_ccr__" in tool_out


@respx.mock
async def test_metrics_count_after_request(app):
    respx.routes.clear()
    respx.post("https://openai.test/v1/chat/completions").mock(
        return_value=httpx.Response(200, json={"ok": True})
    )

    body = {"model": "gpt-4o", "messages": [{"role": "tool", "content": BIG_JSON}]}
    async with _asgi_client(app) as client:
        await client.post("/v1/chat/completions", json=body)
        metrics = await client.get("/metrics")

    assert metrics.status_code == 200
    text = metrics.text
    assert "tokenslim_proxy_requests_total 1" in text
    assert "tokenslim_proxy_saved_tokens_total" in text
    # Saved tokens must be positive given the big JSON payload.
    saved_line = next(
        line for line in text.splitlines() if line.startswith("tokenslim_proxy_saved_tokens_total ")
    )
    assert int(saved_line.split()[1]) > 0


@respx.mock
async def test_upstream_failure_maps_to_502(app):
    respx.routes.clear()
    respx.post("https://anthropic.test/v1/messages").mock(side_effect=httpx.ConnectError("boom"))
    body = {"model": "claude-sonnet-4-6", "messages": [{"role": "user", "content": "hi"}]}
    async with _asgi_client(app) as client:
        resp = await client.post("/v1/messages", json=body)
    assert resp.status_code == 502
    assert resp.json()["error"]["type"] == "tokenslim_proxy_error"


async def test_malformed_json_body_returns_400(app):
    async with _asgi_client(app) as client:
        resp = await client.post(
            "/v1/messages",
            content=b"{not json",
            headers={"content-type": "application/json"},
        )
    assert resp.status_code == 400


@respx.mock
async def test_body_without_messages_is_forwarded_unchanged(app):
    respx.routes.clear()
    route = respx.post("https://openai.test/v1/chat/completions").mock(
        return_value=httpx.Response(200, json={"ok": True})
    )
    body = {"model": "gpt-4o", "input": "no messages key here"}
    async with _asgi_client(app) as client:
        resp = await client.post("/v1/chat/completions", json=body)
    assert resp.status_code == 200
    forwarded = json.loads(route.calls.last.request.content)
    assert forwarded == body


@pytest.mark.parametrize("disabled_app_messages", [True])
async def test_compression_disabled_forwards_verbatim(config, disabled_app_messages):
    from dataclasses import replace

    from tokenslim_proxy import create_app

    disabled = create_app(replace(config, compression_enabled=False))
    with respx.mock:
        route = respx.post("https://openai.test/v1/chat/completions").mock(
            return_value=httpx.Response(200, json={"ok": True})
        )
        body = {"model": "gpt-4o", "messages": [{"role": "tool", "content": BIG_JSON}]}
        async with _asgi_client(disabled) as client:
            await client.post("/v1/chat/completions", json=body)
    forwarded = json.loads(route.calls.last.request.content)
    # Verbatim: the big block is unchanged when compression is off.
    assert forwarded["messages"][0]["content"] == BIG_JSON


@respx.mock
async def test_openai_responses_compression(app):
    respx.routes.clear()
    route = respx.post("https://openai.test/v1/responses").mock(
        return_value=httpx.Response(200, json={"id": "resp_1", "object": "response"})
    )

    request_body = {
        "model": "gpt-4o",
        "input": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": BIG_JSON,
                    }
                ]
            }
        ]
    }

    async with _asgi_client(app) as client:
        resp = await client.post(
            "/v1/responses",
            json=request_body,
            headers={"authorization": "Bearer sk-openai-test"},
        )

    assert resp.status_code == 200
    assert route.called
    sent = route.calls.last.request
    assert str(sent.url) == "https://openai.test/v1/responses"
    assert sent.headers["authorization"] == "Bearer sk-openai-test"

    forwarded = json.loads(sent.content)
    input_item_out = forwarded["input"][0]["content"][0]["text"]
    assert len(input_item_out) < len(BIG_JSON)
    assert json.loads(input_item_out) is not None
    assert "__tokenslim_ccr__" in input_item_out

