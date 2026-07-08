"""AWS Bedrock route + SigV4 signing tests (mocked httpx, no network/keys).

The SigV4 signatures asserted here were validated byte-for-byte against
``botocore.auth.SigV4Auth`` during development; the frozen vector below is a
regression guard so the hand-rolled signer can never silently drift.
"""

from __future__ import annotations

import datetime
import json
from collections.abc import AsyncIterator
from dataclasses import replace

import httpx
import pytest
import respx

from tests.conftest import BIG_JSON
from tokenslim_proxy import ProxyConfig, create_app
from tokenslim_proxy.sigv4 import AwsCredentials, sign_request

UTC = datetime.timezone.utc
MODEL_ID = "anthropic.claude-3-5-sonnet-20240620-v1:0"


def _asgi_client(app) -> httpx.AsyncClient:
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://testserver")


@pytest.fixture
def bedrock_app():
    """App with Bedrock pointed at a fake host so respx can intercept by host."""
    return create_app(
        ProxyConfig(bedrock_base="https://bedrock.test", aws_region="us-east-1")
    )


@pytest.fixture
def aws_creds(monkeypatch):
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIDEXAMPLE")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "wJalrXUtnFEMI/K7MDENG+bPxRfiCYEXAMPLEKEY")
    monkeypatch.delenv("AWS_SESSION_TOKEN", raising=False)


# --- SigV4 unit coverage ---------------------------------------------------


def test_sigv4_frozen_known_answer():
    """A fixed date + keys + body reproduce an exact Authorization header."""
    creds = AwsCredentials("AKIDEXAMPLE", "wJalrXUtnFEMI/K7MDENG+bPxRfiCYEXAMPLEKEY")
    url = f"https://bedrock-runtime.us-east-1.amazonaws.com/model/{MODEL_ID}/invoke"
    payload = (
        b'{"anthropic_version":"bedrock-2023-05-31","max_tokens":16,'
        b'"messages":[{"role":"user","content":"hi"}]}'
    )
    out = sign_request(
        method="POST",
        url=url,
        payload=payload,
        region="us-east-1",
        service="bedrock",
        credentials=creds,
        headers={"content-type": "application/json"},
        now=datetime.datetime(2024, 1, 15, 12, 0, 0, tzinfo=UTC),
    )
    assert out["x-amz-date"] == "20240115T120000Z"
    assert out["authorization"] == (
        "AWS4-HMAC-SHA256 "
        "Credential=AKIDEXAMPLE/20240115/us-east-1/bedrock/aws4_request, "
        "SignedHeaders=content-type;host;x-amz-date, "
        "Signature=0333b9a3fa7516f7f144b7ead2e3a2a0385a29b73dee002a0964bb5298a34db4"
    )
    # host is signed but not returned (the HTTP client sets it from the URL).
    assert "host" not in out


def test_sigv4_session_token_is_signed_and_sent():
    creds = AwsCredentials("AK", "SK", session_token="SESSIONTOKEN==")
    out = sign_request(
        method="POST",
        url="https://bedrock.test/model/m/invoke",
        payload=b"{}",
        region="us-east-1",
        service="bedrock",
        credentials=creds,
        headers={"content-type": "application/json"},
        now=datetime.datetime(2024, 1, 15, 12, 0, 0, tzinfo=UTC),
    )
    assert out["x-amz-security-token"] == "SESSIONTOKEN=="
    assert "x-amz-security-token" in out["authorization"]


def test_sigv4_body_change_changes_signature():
    """The signature covers the body, so compression must precede signing."""
    kw = dict(
        method="POST",
        url="https://bedrock.test/model/m/invoke",
        region="us-east-1",
        service="bedrock",
        credentials=AwsCredentials("AK", "SK"),
        headers={"content-type": "application/json"},
        now=datetime.datetime(2024, 1, 15, 12, 0, 0, tzinfo=UTC),
    )
    a = sign_request(payload=b'{"a":1}', **kw)
    b = sign_request(payload=b'{"a":2}', **kw)
    assert a["authorization"] != b["authorization"]


def test_credentials_from_env_missing_returns_none():
    assert AwsCredentials.from_env({}) is None
    assert AwsCredentials.from_env({"AWS_ACCESS_KEY_ID": "x"}) is None
    creds = AwsCredentials.from_env(
        {"AWS_ACCESS_KEY_ID": "x", "AWS_SECRET_ACCESS_KEY": "y", "AWS_SESSION_TOKEN": "z"}
    )
    assert creds == AwsCredentials("x", "y", "z")


def test_config_bedrock_endpoint_default_and_override():
    assert (
        ProxyConfig(aws_region="eu-west-1").bedrock_endpoint()
        == "https://bedrock-runtime.eu-west-1.amazonaws.com"
    )
    assert (
        ProxyConfig(bedrock_base="https://vpce.test").bedrock_endpoint() == "https://vpce.test"
    )


# --- route behavior --------------------------------------------------------


@respx.mock
async def test_bedrock_invoke_compresses_and_signs(bedrock_app, aws_creds):
    respx.routes.clear()
    route = respx.post(f"https://bedrock.test/model/{MODEL_ID}/invoke").mock(
        return_value=httpx.Response(200, json={"id": "msg_1", "role": "assistant"})
    )

    request_body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 1024,
        "messages": [
            {"role": "user", "content": "summarize this"},
            {"role": "user", "content": BIG_JSON},
        ],
    }

    async with _asgi_client(bedrock_app) as client:
        resp = await client.post(f"/model/{MODEL_ID}/invoke", json=request_body)

    assert resp.status_code == 200
    assert resp.json()["id"] == "msg_1"

    assert route.called
    sent = route.calls.last.request
    assert str(sent.url) == f"https://bedrock.test/model/{MODEL_ID}/invoke"
    # SigV4 headers present; the proxy signed, the client sent no auth.
    assert sent.headers["authorization"].startswith("AWS4-HMAC-SHA256 ")
    assert "SignedHeaders=" in sent.headers["authorization"]
    assert sent.headers["x-amz-date"].endswith("Z")

    # Body compressed: the big block shrank and still parses as JSON.
    forwarded = json.loads(sent.content)
    big_out = forwarded["messages"][1]["content"]
    assert len(big_out) < len(request_body["messages"][1]["content"])
    assert "__tokenslim_ccr__" in big_out
    # Non-message fields pass through untouched.
    assert forwarded["anthropic_version"] == "bedrock-2023-05-31"
    assert forwarded["max_tokens"] == 1024


@respx.mock
async def test_bedrock_signature_matches_forwarded_body(bedrock_app, aws_creds):
    """The forwarded body verifies against its own Authorization signature."""
    respx.routes.clear()
    route = respx.post(f"https://bedrock.test/model/{MODEL_ID}/invoke").mock(
        return_value=httpx.Response(200, json={"ok": True})
    )
    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "messages": [{"role": "user", "content": BIG_JSON}],
    }
    async with _asgi_client(bedrock_app) as client:
        await client.post(f"/model/{MODEL_ID}/invoke", json=body)

    sent = route.calls.last.request
    auth = sent.headers["authorization"]
    # Re-sign the exact forwarded bytes with the same date; signatures must match.
    amz = sent.headers["x-amz-date"]
    now = datetime.datetime.strptime(amz, "%Y%m%dT%H%M%SZ").replace(tzinfo=UTC)
    expected = sign_request(
        method="POST",
        url=f"https://bedrock.test/model/{MODEL_ID}/invoke",
        payload=sent.content,
        region="us-east-1",
        service="bedrock",
        credentials=AwsCredentials("AKIDEXAMPLE", "wJalrXUtnFEMI/K7MDENG+bPxRfiCYEXAMPLEKEY"),
        headers={"content-type": "application/json", "accept": "application/json"},
        now=now,
    )
    assert auth == expected["authorization"]


@respx.mock
async def test_bedrock_missing_credentials_returns_500(bedrock_app, monkeypatch):
    monkeypatch.delenv("AWS_ACCESS_KEY_ID", raising=False)
    monkeypatch.delenv("AWS_SECRET_ACCESS_KEY", raising=False)
    respx.routes.clear()
    route = respx.post(f"https://bedrock.test/model/{MODEL_ID}/invoke").mock(
        return_value=httpx.Response(200, json={"ok": True})
    )
    body = {"messages": [{"role": "user", "content": "hi"}]}
    async with _asgi_client(bedrock_app) as client:
        resp = await client.post(f"/model/{MODEL_ID}/invoke", json=body)
    assert resp.status_code == 500
    assert resp.json()["error"]["type"] == "tokenslim_proxy_error"
    # No unsigned request ever reached the upstream.
    assert not route.called


@respx.mock
async def test_bedrock_stream_relays_eventstream_bytes(bedrock_app, aws_creds):
    """invoke-with-response-stream relays Bedrock's binary EventStream intact."""
    # Bedrock's response is application/vnd.amazon.eventstream (binary frames),
    # not SSE. The proxy must relay the raw bytes untouched, chunk by chunk.
    frames = [
        b"\x00\x00\x00\x10\x00\x00\x00\x00frame-a",
        b"\x00\x00\x00\x0c\x00\x00\x00\x00frame-b",
    ]
    full = b"".join(frames)

    async def _chunks() -> AsyncIterator[bytes]:
        for f in frames:
            yield f

    respx.routes.clear()
    respx.post(f"https://bedrock.test/model/{MODEL_ID}/invoke-with-response-stream").mock(
        return_value=httpx.Response(
            200,
            headers={"content-type": "application/vnd.amazon.eventstream"},
            content=_chunks(),
        )
    )

    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "messages": [{"role": "user", "content": BIG_JSON}],
    }
    received = bytearray()
    async with _asgi_client(bedrock_app) as client:
        async with client.stream(
            "POST", f"/model/{MODEL_ID}/invoke-with-response-stream", json=body
        ) as resp:
            assert resp.status_code == 200
            assert resp.headers["content-type"].startswith("application/vnd.amazon.eventstream")
            async for chunk in resp.aiter_raw():
                received.extend(chunk)

    assert bytes(received) == full


@respx.mock
async def test_bedrock_stream_sends_stream_action_and_accept(bedrock_app, aws_creds):
    respx.routes.clear()
    route = respx.post(
        f"https://bedrock.test/model/{MODEL_ID}/invoke-with-response-stream"
    ).mock(return_value=httpx.Response(200, content=b"ok"))
    body = {"messages": [{"role": "user", "content": "hi"}]}
    async with _asgi_client(bedrock_app) as client:
        async with client.stream(
            "POST", f"/model/{MODEL_ID}/invoke-with-response-stream", json=body
        ) as resp:
            async for _ in resp.aiter_raw():
                pass
    sent = route.calls.last.request
    assert sent.headers["accept"] == "application/vnd.amazon.eventstream"
    assert sent.headers["authorization"].startswith("AWS4-HMAC-SHA256 ")


@respx.mock
async def test_bedrock_compression_disabled_forwards_verbatim(aws_creds):
    disabled = create_app(
        replace(
            ProxyConfig(bedrock_base="https://bedrock.test"), compression_enabled=False
        )
    )
    respx.routes.clear()
    route = respx.post(f"https://bedrock.test/model/{MODEL_ID}/invoke").mock(
        return_value=httpx.Response(200, json={"ok": True})
    )
    body = {"messages": [{"role": "user", "content": BIG_JSON}]}
    async with _asgi_client(disabled) as client:
        await client.post(f"/model/{MODEL_ID}/invoke", json=body)
    forwarded = json.loads(route.calls.last.request.content)
    assert forwarded["messages"][0]["content"] == BIG_JSON
