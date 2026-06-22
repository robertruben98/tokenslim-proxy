"""M2: cache-prefix stabilization tests (no HTTP)."""

from __future__ import annotations

import json

from tests.conftest import BIG_JSON
from tokenslim_proxy.cache import add_anthropic_cache_breakpoint
from tokenslim_proxy.compression import compress_messages_body

# A second large blob so turn 2 adds genuinely new, compressible content.
BIG_JSON_2 = json.dumps({"events": [{"i": i, "v": i * 3} for i in range(70)]}, indent=2)


def _contents(outcome):
    return [m["content"] for m in outcome.body["messages"]]


def test_prefix_bytes_identical_across_turns():
    # Turn 1: one big user message + an assistant reply.
    turn1 = {
        "messages": [
            {"role": "user", "content": BIG_JSON},
            {"role": "assistant", "content": "here is the summary"},
        ]
    }
    # Turn 2: the SAME prefix, plus a new big user message.
    turn2 = {
        "messages": [
            {"role": "user", "content": BIG_JSON},
            {"role": "assistant", "content": "here is the summary"},
            {"role": "user", "content": BIG_JSON_2},
        ]
    }

    out1 = compress_messages_body(turn1, stable_prefix=True)
    out2 = compress_messages_body(turn2, stable_prefix=True)

    c1 = _contents(out1)
    c2 = _contents(out2)

    # The shared prefix (messages 0 and 1) compresses to identical bytes, so a
    # provider prompt cache built in turn 1 still hits in turn 2.
    assert c2[0] == c1[0]
    assert c2[1] == c1[1]
    # The new tail was actually compressed.
    assert out2.changed is True
    assert len(c2[2]) < len(BIG_JSON_2)


def test_stable_and_whole_array_both_compress():
    body = {"messages": [{"role": "tool", "content": BIG_JSON}]}
    stable = compress_messages_body(body, stable_prefix=True)
    whole = compress_messages_body(body, stable_prefix=False)
    # Single-message case: both paths yield the same compressed result.
    assert stable.body["messages"][0]["content"] == whole.body["messages"][0]["content"]
    assert stable.changed and whole.changed


def test_stable_compression_is_deterministic():
    body = {"messages": [{"role": "user", "content": BIG_JSON}]}
    a = compress_messages_body(body, stable_prefix=True)
    b = compress_messages_body(body, stable_prefix=True)
    assert a.body == b.body


def test_disabled_passes_through_even_with_stable():
    body = {"messages": [{"role": "tool", "content": BIG_JSON}]}
    out = compress_messages_body(body, enabled=False, stable_prefix=True)
    assert out.changed is False
    assert out.body["messages"][0]["content"] == BIG_JSON


# --- Anthropic cache_control breakpoint -------------------------------------


def test_breakpoint_added_to_last_prefix_message():
    body = {
        "messages": [
            {"role": "user", "content": "system-ish big context"},
            {"role": "assistant", "content": "ack"},
            {"role": "user", "content": "newest turn"},
        ]
    }
    out = add_anthropic_cache_breakpoint(body)
    msgs = out["messages"]
    # The breakpoint lands on the last PREFIX message (index 1), not the newest.
    assert isinstance(msgs[1]["content"], list)
    assert msgs[1]["content"][-1]["cache_control"] == {"type": "ephemeral"}
    # Newest turn is untouched.
    assert msgs[2]["content"] == "newest turn"
    # Input not mutated.
    assert isinstance(body["messages"][1]["content"], str)


def test_breakpoint_noop_when_client_already_set_one():
    body = {
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": "x", "cache_control": {"type": "ephemeral"}}],
            },
            {"role": "user", "content": "newest"},
        ]
    }
    out = add_anthropic_cache_breakpoint(body)
    assert out is body  # untouched


def test_breakpoint_noop_for_single_message():
    body = {"messages": [{"role": "user", "content": "only one"}]}
    assert add_anthropic_cache_breakpoint(body) is body
