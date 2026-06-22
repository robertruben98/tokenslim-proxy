# tokenslim-proxy

OpenAI/Anthropic-compatible HTTP proxy that transparently compresses LLM
context. **Zero client code changes** — point your SDK's base URL at the proxy
and it compresses large tool-result / text blocks via the
[`tokenslim`](https://github.com/robertruben98/tokenslim) core before
forwarding to the real provider, preserving your auth headers.

Built on **FastAPI + httpx**.

## Routes

| Method | Path | Behavior |
| --- | --- | --- |
| `POST` | `/v1/messages` | Compress + forward to `api.anthropic.com` (streams when `stream:true`). |
| `POST` | `/v1/chat/completions` | Compress + forward to `api.openai.com` (streams when `stream:true`). |
| `POST` | `/v1/responses` | Compress the Responses-API `input` + forward to `api.openai.com` (streams when `stream:true`). |
| `GET` | `/healthz` | Liveness. |
| `GET` | `/healthz/upstream` | Resolved upstream config (no network call). |
| `GET` | `/metrics` | Prometheus text exposition of compression counters. |

## Run

```bash
pip install -e .            # installs the tokenslim core via git
tokenslim-proxy            # or: uvicorn tokenslim_proxy.app:app --port 8788
```

Then point a client at it, e.g. OpenAI:

```bash
export OPENAI_BASE_URL=http://127.0.0.1:8788/v1
```

or Anthropic:

```bash
export ANTHROPIC_BASE_URL=http://127.0.0.1:8788
```

### Wrap a coding agent

`tokenslim wrap <agent>` sets the right base-URL env vars for a coding agent and
execs its binary so its API calls route through the proxy — no per-agent config:

```bash
tokenslim wrap claude            # runs `claude` with ANTHROPIC_BASE_URL → proxy
tokenslim wrap codex             # OPENAI_BASE_URL / OPENAI_API_BASE → proxy
tokenslim wrap aider --model gpt-4o   # extra args pass through verbatim
```

Supported agents: `claude`, `codex`, `cursor`, `aider`, `copilot`. Your API
keys stay in the environment untouched (the proxy forwards them upstream). If
the agent's binary isn't on `PATH`, `wrap` fails with a clear install hint.

## Configuration (env)

| Variable | Default | Description |
| --- | --- | --- |
| `TOKENSLIM_PROXY_HOST` | `127.0.0.1` | Bind interface. |
| `TOKENSLIM_PROXY_PORT` | `8788` | Bind port. |
| `TOKENSLIM_PROXY_ANTHROPIC_BASE` | `https://api.anthropic.com` | Anthropic upstream. |
| `TOKENSLIM_PROXY_OPENAI_BASE` | `https://api.openai.com` | OpenAI upstream. |
| `TOKENSLIM_PROXY_UPSTREAM_TIMEOUT` | `60` | Per-request timeout (s). |
| `TOKENSLIM_PROXY_COMPRESSION` | `1` | Master on/off switch. |
| `TOKENSLIM_PROXY_CACHE_PREFIX_STABLE` | `1` | Compress each message in isolation so the already-sent prefix stays byte-stable across turns. |
| `TOKENSLIM_PROXY_ANTHROPIC_CACHE_BREAKPOINT` | `0` | Insert an Anthropic `cache_control` breakpoint at the prefix boundary (opt-in). |

## How it works

Both `/v1/messages` and `/v1/chat/completions` carry a `messages` array. On each
request the proxy parses the JSON body, runs that array through the core
`compress()`, re-serializes, and forwards to the configured upstream — relaying
all headers except hop-by-hop / length / encoding ones, so `authorization`,
`x-api-key` and `anthropic-version` are preserved. Non-`messages` fields (model,
tools, system, temperature…) are passed through untouched.

### Responses API (`/v1/responses`)

The Responses API has no `messages` array; its prompt lives under `input`,
either a string or a list of *items*. The proxy iterates the items and
compresses the text it finds — typed content blocks (`input_text` /
`output_text` / `text`) and the `output` of a `function_call_output` — bridging
each payload into the core's text shape and writing the compressed text back
into the original item shape (so an `input_text` block stays `input_text`).
Items the core can't help with (`reasoning`, `function_call`, item refs…) and
all non-`input` fields are forwarded untouched. Each text payload is compressed
in isolation, so the same cache-prefix stability applies as for chat.

### Streaming (SSE)

When the request body has `"stream": true`, the proxy opens a streaming upstream
POST (`httpx.stream`) and relays the raw SSE byte stream to the client
chunk-by-chunk via a `StreamingResponse` — without buffering the whole body and
without re-parsing, so `event:` / `data:` framing is preserved byte-for-byte.
The non-streaming path is unchanged (buffered).

### Cache-prefix stabilization

With `cache_prefix_stable` on (default), each message is compressed **in
isolation**, so a message's compressed bytes depend only on its own content —
never on its position or neighbours. Across turns the shared conversation prefix
therefore compresses to identical bytes, so provider KV/prompt caches keep
hitting; only the new tail content changes. Optionally
(`anthropic_cache_breakpoint`), an Anthropic request gets a single
`cache_control: {type: ephemeral}` marker placed on the last prefix message — at
the same stable boundary — and is left alone if the client already set one.

## Development

```bash
pip install -e ".[dev]"
ruff check .
python -m pytest -q
```

Tests use `respx` to intercept httpx, so they need **no API keys and no
network**.

## Roadmap (deferred / open issues)

- AWS Bedrock (SigV4) (`#8`) and Google Vertex (ADC) (`#9`)

Done: SSE streaming passthrough (`#6`), cache-prefix stabilization (`#7`),
OpenAI `/v1/responses` (`#5`), `tokenslim wrap <agent>` launcher (`#10`).

## License

Apache-2.0
