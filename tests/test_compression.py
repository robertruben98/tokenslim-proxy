"""Unit tests for the compression adapter (no HTTP)."""

from __future__ import annotations

import json

from tests.conftest import BIG_JSON
from tokenslim_proxy import compress_messages_body


def test_compresses_and_does_not_mutate_input():
    body = {"model": "gpt-4o", "messages": [{"role": "tool", "content": BIG_JSON}]}
    snapshot = json.dumps(body)

    outcome = compress_messages_body(body)

    assert outcome.changed is True
    assert outcome.new_tokens < outcome.orig_tokens
    assert 0.0 < outcome.ratio <= 1.0
    # Input object untouched.
    assert json.dumps(body) == snapshot
    # Result is smaller, still valid JSON, and carries the CCR elision marker
    # (the core compresses via marker-based elision, not byte-exact minify).
    out_content = outcome.body["messages"][0]["content"]
    assert len(out_content) < len(BIG_JSON)
    assert json.loads(out_content) is not None
    assert "__tokenslim_ccr__" in out_content


def test_no_messages_key_passes_through():
    body = {"model": "gpt-4o", "input": "x"}
    outcome = compress_messages_body(body)
    assert outcome.changed is False
    assert outcome.body == body
    assert outcome.orig_tokens == 0


def test_disabled_forwards_without_change():
    body = {"messages": [{"role": "tool", "content": BIG_JSON}]}
    outcome = compress_messages_body(body, enabled=False)
    assert outcome.changed is False
    assert outcome.body["messages"][0]["content"] == BIG_JSON
