"""Header hygiene and upstream forwarding helpers."""

from __future__ import annotations

import httpx

# Request headers we must not blindly relay: the body changes after
# compression (so the client's length is wrong) and hop-by-hop headers are
# meaningless to the upstream. Auth headers (authorization, x-api-key,
# anthropic-version, openai-organization, …) are intentionally preserved.
_DROP_REQUEST_HEADERS = {
    "host",
    "content-length",
    "connection",
    "keep-alive",
    "proxy-authorization",
    "proxy-connection",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
    "accept-encoding",
}

# Response headers that describe the proxy's own transport, not the upstream
# payload, and would corrupt the client's view if relayed verbatim.
_DROP_RESPONSE_HEADERS = {
    "content-length",
    "content-encoding",
    "connection",
    "keep-alive",
    "transfer-encoding",
}


def filter_request_headers(headers: httpx.Headers | dict[str, str]) -> dict[str, str]:
    """Return forwardable request headers, preserving auth and dropping hop-by-hop."""
    items = headers.items() if hasattr(headers, "items") else dict(headers).items()
    return {k: v for k, v in items if k.lower() not in _DROP_REQUEST_HEADERS}


def filter_response_headers(headers: httpx.Headers | dict[str, str]) -> dict[str, str]:
    """Return relayable response headers, dropping transport-specific ones."""
    items = headers.items() if hasattr(headers, "items") else dict(headers).items()
    return {k: v for k, v in items if k.lower() not in _DROP_RESPONSE_HEADERS}
