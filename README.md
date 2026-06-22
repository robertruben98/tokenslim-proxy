# tokenslim-proxy

OpenAI/Anthropic-compatible HTTP proxy that transparently compresses LLM
context. **Zero client code changes** — point your SDK's base URL at the proxy
and it compresses large tool-result / text blocks via the
[`tokenslim`](https://github.com/robertruben98/tokenslim) core before
forwarding to the real provider, preserving your auth headers.

Built on **FastAPI + httpx**.

## Routes (M0 foundation)

| Method | Path | Behavior |
| --- | --- | --- |
| `POST` | `/v1/messages` | Compress + forward to `api.anthropic.com`. |
| `POST` | `/v1/chat/completions` | Compress + forward to `api.openai.com`. |
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

## Configuration (env)

| Variable | Default | Description |
| --- | --- | --- |
| `TOKENSLIM_PROXY_HOST` | `127.0.0.1` | Bind interface. |
| `TOKENSLIM_PROXY_PORT` | `8788` | Bind port. |
| `TOKENSLIM_PROXY_ANTHROPIC_BASE` | `https://api.anthropic.com` | Anthropic upstream. |
| `TOKENSLIM_PROXY_OPENAI_BASE` | `https://api.openai.com` | OpenAI upstream. |
| `TOKENSLIM_PROXY_UPSTREAM_TIMEOUT` | `60` | Per-request timeout (s). |
| `TOKENSLIM_PROXY_COMPRESSION` | `1` | Master on/off switch. |

## How it works

Both `/v1/messages` and `/v1/chat/completions` carry a `messages` array. On each
request the proxy parses the JSON body, runs that array through the core
`compress()`, re-serializes, and forwards to the configured upstream — relaying
all headers except hop-by-hop / length / encoding ones, so `authorization`,
`x-api-key` and `anthropic-version` are preserved. The response is streamed back
with transport headers stripped. Non-`messages` fields (model, tools, system,
temperature…) are passed through untouched.

## Development

```bash
pip install -e ".[dev]"
ruff check .
python -m pytest -q
```

Tests use `respx` to intercept httpx, so they need **no API keys and no
network**.

## Roadmap (deferred / open issues)

- SSE streaming passthrough (`#6`)
- Cache-prefix stabilization for KV/prompt-cache hits (`#7`)
- OpenAI `/v1/responses` (`#5`)
- AWS Bedrock (SigV4) (`#8`) and Google Vertex (ADC) (`#9`)
- `tokenslim wrap <agent>` launcher (`#10`)

## License

Apache-2.0
