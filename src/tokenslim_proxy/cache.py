"""Anthropic prompt-cache breakpoint insertion.

Anthropic prompt caching is opt-in per request: you mark a content block with
``"cache_control": {"type": "ephemeral"}`` and everything up to and including
that block becomes a cached prefix. To get cache *hits* across turns the
breakpoint must sit at a **stable boundary** — the end of the conversation
prefix that does not change turn-to-turn — which is exactly the part this proxy
keeps byte-stable (see ``compression._compress_stable``).

This module inserts a single breakpoint on the last block of the last prefix
message (every message except the final, newest turn). It is idempotent: if a
``cache_control`` already exists anywhere in the body it leaves the body alone,
so a client that manages its own breakpoints is never overridden.
"""

from __future__ import annotations

from typing import Any


def _has_cache_control(messages: list[Any]) -> bool:
    """True if any message already carries a ``cache_control`` marker."""
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        content = msg.get("content")
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and "cache_control" in block:
                    return True
    return False


def _string_to_blocks(text: str) -> list[dict[str, Any]]:
    """Promote a string content to the block form Anthropic also accepts."""
    return [{"type": "text", "text": text}]


def add_anthropic_cache_breakpoint(body: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of ``body`` with one ephemeral breakpoint at the prefix end.

    No-op (returns the input unchanged) when: there are fewer than two messages
    (no stable prefix to cache), or the client already set a ``cache_control``.
    String-valued content on the chosen message is promoted to block form so
    the marker can be attached.
    """
    messages = body.get("messages")
    if not isinstance(messages, list) or len(messages) < 2:
        return body
    if _has_cache_control(messages):
        return body

    # The stable prefix is everything except the final (newest) turn.
    prefix_index = len(messages) - 2
    target = messages[prefix_index]
    if not isinstance(target, dict):
        return body

    new_messages = list(messages)
    new_target = dict(target)
    content = new_target.get("content")

    if isinstance(content, str):
        blocks = _string_to_blocks(content)
    elif isinstance(content, list):
        blocks = [dict(b) if isinstance(b, dict) else b for b in content]
    else:
        return body

    if not blocks:
        return body

    last = blocks[-1]
    if not isinstance(last, dict):
        return body
    last["cache_control"] = {"type": "ephemeral"}
    new_target["content"] = blocks
    new_messages[prefix_index] = new_target

    new_body = dict(body)
    new_body["messages"] = new_messages
    return new_body
