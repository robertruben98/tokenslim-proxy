"""Compression for the OpenAI Responses API (``POST /v1/responses``).

The Responses API does not use a ``messages`` array; its prompt lives under
``input``, which is either a plain string or a list of *items*. Items carry
text in a few different shapes, none of which the ``tokenslim`` core recognizes
verbatim:

* a message item — ``{"role", "content": <str | [blocks]>}`` (optionally with
  ``"type": "message"``). Content blocks are typed ``input_text`` /
  ``output_text`` (not the core's ``text``).
* a tool result — ``{"type": "function_call_output", "output": <str>}``, whose
  ``output`` is often a large JSON/log blob worth compressing.

The core only compresses string content and ``{"type": "text", "text": ...}``
blocks (verified against the live core). So this adapter *bridges* each Responses
text payload into that shape, runs the core on it in isolation (one payload at a
time, mirroring ``compression._compress_stable`` so already-sent prefix bytes
stay byte-stable across turns and provider caches keep hitting), then writes the
compressed text back into the original item shape. Everything else in the body
(model, instructions, tools, reasoning, …) is forwarded untouched.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from tokenslim import compress as _core_compress

# Responses-API content-block types whose ``text`` field we compress. They map
# onto the core's ``text`` block, which is the only block type it rewrites.
_TEXT_BLOCK_TYPES = {"input_text", "output_text", "text"}


@dataclass(frozen=True)
class ResponsesOutcome:
    """Result of compressing one Responses-API request body."""

    body: dict[str, Any]
    orig_tokens: int
    new_tokens: int
    ratio: float
    changed: bool


def _compress_text(text: str, *, enabled: bool) -> tuple[str, int, int]:
    """Compress a single text payload through the core, in isolation.

    Bridges the string into a one-message array so the core treats it as plain
    content, then returns ``(compressed_text, orig_tokens, new_tokens)``.
    """
    out, stats = _core_compress([{"role": "user", "content": text}], enabled=enabled)
    new_text = out[0]["content"] if isinstance(out[0].get("content"), str) else text
    return new_text, stats.orig_tokens, stats.new_tokens


def _compress_blocks(blocks: list[Any], *, enabled: bool) -> tuple[list[Any], int, int]:
    """Compress the ``text`` of each recognized block, preserving block shape.

    Non-text blocks (images, files, refusals, …) and non-dict entries pass
    through untouched. The block's ``type`` is preserved, so an ``input_text``
    block stays ``input_text`` on the way out.
    """
    out: list[Any] = []
    orig_total = 0
    new_total = 0
    for block in blocks:
        if (
            isinstance(block, dict)
            and block.get("type") in _TEXT_BLOCK_TYPES
            and isinstance(block.get("text"), str)
        ):
            new_text, orig, new = _compress_text(block["text"], enabled=enabled)
            new_block = dict(block)
            new_block["text"] = new_text
            out.append(new_block)
            orig_total += orig
            new_total += new
        else:
            out.append(block)
    return out, orig_total, new_total


def _compress_item(item: Any, *, enabled: bool) -> tuple[Any, int, int]:
    """Compress one ``input`` item, returning ``(item, orig, new)``.

    Handles three shapes: list-of-blocks ``content``, string ``content`` on a
    message item, and the string ``output`` of a ``function_call_output``. Any
    other item (function_call, reasoning, item references, …) is passed through.
    """
    if not isinstance(item, dict):
        return item, 0, 0

    content = item.get("content")
    if isinstance(content, list):
        new_blocks, orig, new = _compress_blocks(content, enabled=enabled)
        new_item = dict(item)
        new_item["content"] = new_blocks
        return new_item, orig, new

    if isinstance(content, str):
        new_text, orig, new = _compress_text(content, enabled=enabled)
        new_item = dict(item)
        new_item["content"] = new_text
        return new_item, orig, new

    if item.get("type") == "function_call_output" and isinstance(item.get("output"), str):
        new_text, orig, new = _compress_text(item["output"], enabled=enabled)
        new_item = dict(item)
        new_item["output"] = new_text
        return new_item, orig, new

    return item, 0, 0


def compress_responses_body(body: dict[str, Any], *, enabled: bool = True) -> ResponsesOutcome:
    """Compress the ``input`` of a Responses-API request body.

    The input ``body`` is never mutated; a shallow copy with a rewritten
    ``input`` is returned. ``input`` may be:

    * a string — compressed as a single payload;
    * a list of items — each item compressed in isolation (stable prefix);
    * absent / any other type — forwarded unchanged with zeroed stats.

    When ``enabled`` is ``False`` the body is forwarded verbatim (the core still
    reports token counts so ``/metrics`` stays meaningful).
    """
    raw_input = body.get("input")

    if isinstance(raw_input, str):
        new_input, orig_tokens, new_tokens = _compress_text(raw_input, enabled=enabled)
    elif isinstance(raw_input, list) and raw_input:
        out_items: list[Any] = []
        orig_tokens = 0
        new_tokens = 0
        for item in raw_input:
            new_item, orig, new = _compress_item(item, enabled=enabled)
            out_items.append(new_item)
            orig_tokens += orig
            new_tokens += new
        new_input = out_items
    else:
        return ResponsesOutcome(body=body, orig_tokens=0, new_tokens=0, ratio=0.0, changed=False)

    new_body = dict(body)
    new_body["input"] = new_input

    ratio = 0.0 if orig_tokens <= 0 else 1.0 - (new_tokens / orig_tokens)
    changed = new_tokens < orig_tokens
    return ResponsesOutcome(
        body=new_body,
        orig_tokens=orig_tokens,
        new_tokens=new_tokens,
        ratio=ratio,
        changed=changed,
    )
