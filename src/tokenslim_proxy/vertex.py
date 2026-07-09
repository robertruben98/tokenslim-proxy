"""Google Vertex AI route helpers: endpoint URLs + ADC bearer-token auth.

Vertex serves Anthropic Claude models at a regional ``aiplatform`` endpoint::

    {base}/v1/projects/{project}/locations/{location}/publishers/anthropic/models/{model}:rawPredict

(``:streamRawPredict`` for streaming). The forwarded body is the ordinary
Anthropic ``/v1/messages`` shape with ``model`` dropped (it lives in the URL)
and an ``anthropic_version`` of ``vertex-2023-10-16`` added.

Auth is a short-lived OAuth2 bearer token from Google Application Default
Credentials (ADC). Unlike Bedrock's SigV4 the token does not sign the request
body, so the proxy is free to compress the body and simply attach a fresh
token. Token minting goes through the injectable :data:`TokenProvider`, so the
route can be exercised in tests without real credentials or a network call.
"""

from __future__ import annotations

from collections.abc import Callable
from urllib.parse import quote

# A zero-arg callable returning a valid OAuth2 access token string.
TokenProvider = Callable[[], str]

# Scope the Vertex AI prediction API requires.
_CLOUD_PLATFORM_SCOPE = "https://www.googleapis.com/auth/cloud-platform"

# ``anthropic_version`` value the Vertex Anthropic endpoint expects in the body.
VERTEX_ANTHROPIC_VERSION = "vertex-2023-10-16"


class VertexAuthError(RuntimeError):
    """Raised when an ADC bearer token cannot be obtained."""


def vertex_base_url(location: str) -> str:
    """Regional Vertex ``aiplatform`` base URL for ``location`` (e.g. ``us-east5``)."""
    return f"https://{location}-aiplatform.googleapis.com"


def rawpredict_path(project: str, location: str, model: str, *, stream: bool = False) -> str:
    """Path of the Vertex Anthropic ``rawPredict`` endpoint for ``model``.

    Uses ``streamRawPredict`` when ``stream`` is set. The ``model`` segment is
    percent-encoded (Vertex model ids may carry an ``@version`` suffix).
    """
    verb = "streamRawPredict" if stream else "rawPredict"
    model_seg = quote(model, safe="@.")
    return (
        f"/v1/projects/{project}/locations/{location}"
        f"/publishers/anthropic/models/{model_seg}:{verb}"
    )


class GoogleADCTokenProvider:
    """Mint OAuth2 bearer tokens from Google Application Default Credentials.

    ``google.auth`` is an optional dependency loaded lazily on first call, so
    importing the proxy (and its other routes) works without it installed.
    Resolved credentials are cached and refreshed as the token nears expiry.
    """

    def __init__(self) -> None:
        self._credentials: object | None = None
        self._transport: object | None = None

    def __call__(self) -> str:
        creds, transport = self._ensure()
        if not getattr(creds, "valid", False):
            creds.refresh(transport)
        token = getattr(creds, "token", None)
        if not token:
            raise VertexAuthError("Application Default Credentials returned no access token.")
        return token

    def _ensure(self) -> tuple[object, object]:
        if self._credentials is None:
            try:
                import google.auth
                from google.auth.transport.requests import Request as _Request
            except ImportError as exc:
                raise VertexAuthError(
                    "The Vertex route needs Application Default Credentials. Install the "
                    "optional dependency (pip install 'tokenslim-proxy[vertex]') and run "
                    "`gcloud auth application-default login`."
                ) from exc
            creds, _project = google.auth.default(scopes=[_CLOUD_PLATFORM_SCOPE])
            self._credentials = creds
            self._transport = _Request()
        return self._credentials, self._transport
