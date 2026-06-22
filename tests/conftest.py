"""Shared fixtures: an app + ASGI-transport httpx client, no network or keys."""

from __future__ import annotations

import json

import pytest

from tokenslim_proxy import ProxyConfig, create_app

# A pretty-printed JSON tool payload, large enough that the core compresses it.
BIG_JSON = json.dumps(
    {"rows": [{"id": i, "name": f"row-{i}", "ok": True} for i in range(60)]},
    indent=2,
)


@pytest.fixture
def config() -> ProxyConfig:
    """Config pointed at fake upstreams so respx can intercept by host."""
    return ProxyConfig(
        anthropic_base="https://anthropic.test",
        openai_base="https://openai.test",
    )


@pytest.fixture
def app(config: ProxyConfig):
    return create_app(config)
