"""Minimal AWS Signature Version 4 signing, for the native Bedrock route.

Bedrock's ``InvokeModel`` endpoint authenticates with SigV4 rather than a bearer
key, so — unlike the Anthropic/OpenAI routes, which relay the client's auth
header — the proxy holds AWS credentials and (re-)signs each outbound request.
Signing must happen *after* compression, since the signature covers the body.

This is a self-contained implementation over ``hashlib``/``hmac`` (no boto3), so
the proxy keeps its lean dependency set and the whole path stays offline-testable.
It implements the header-based ("Authorization") scheme for a single signed POST
and is validated byte-for-byte against ``botocore.auth.SigV4Auth`` (see tests).
"""

from __future__ import annotations

import datetime
import hashlib
import hmac
import os
from dataclasses import dataclass
from urllib.parse import parse_qsl, quote, urlsplit

_ALGORITHM = "AWS4-HMAC-SHA256"
_AWS4_REQUEST = "aws4_request"

# Headers left out of the signature: they are set by the signing step itself or
# rewritten in transit by clients/proxies. Mirrors botocore's blacklist so our
# canonical request matches theirs.
_UNSIGNED_HEADERS = {"authorization", "user-agent", "x-amzn-trace-id", "expect"}


@dataclass(frozen=True)
class AwsCredentials:
    """AWS credentials used to sign Bedrock requests."""

    access_key: str
    secret_key: str
    session_token: str | None = None

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> AwsCredentials | None:
        """Load credentials from the standard AWS env vars, or ``None`` if unset.

        Returns ``None`` when either the access key or secret key is missing, so
        the route can surface a clear "credentials not configured" error instead
        of sending an unsigned request.
        """
        e = os.environ if env is None else env
        access = e.get("AWS_ACCESS_KEY_ID")
        secret = e.get("AWS_SECRET_ACCESS_KEY")
        if not access or not secret:
            return None
        return cls(access_key=access, secret_key=secret, session_token=e.get("AWS_SESSION_TOKEN"))


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _hmac(key: bytes, msg: str) -> bytes:
    return hmac.new(key, msg.encode("utf-8"), hashlib.sha256).digest()


def _signing_key(secret: str, date_stamp: str, region: str, service: str) -> bytes:
    """Derive the SigV4 signing key by chained HMAC (date → region → service)."""
    k_date = _hmac(f"AWS4{secret}".encode(), date_stamp)
    k_region = _hmac(k_date, region)
    k_service = _hmac(k_region, service)
    return _hmac(k_service, _AWS4_REQUEST)


def _canonical_uri(path: str) -> str:
    """URI-encode the path segments once (bedrock is not S3), keeping ``/``."""
    return quote(path, safe="/~") if path else "/"


def _canonical_query(query: str) -> str:
    """Sort and URI-encode query parameters into the canonical form."""
    if not query:
        return ""
    pairs = sorted(
        (quote(k, safe="~"), quote(v, safe="~"))
        for k, v in parse_qsl(query, keep_blank_values=True)
    )
    return "&".join(f"{k}={v}" for k, v in pairs)


def sign_request(
    *,
    method: str,
    url: str,
    payload: bytes,
    region: str,
    service: str,
    credentials: AwsCredentials,
    headers: dict[str, str] | None = None,
    now: datetime.datetime | None = None,
) -> dict[str, str]:
    """Sign a request and return the headers to send (minus ``host``).

    The returned dict is ready to hand to ``httpx`` as ``headers``: it carries
    the caller's headers (e.g. ``content-type``/``accept``) plus ``x-amz-date``,
    ``authorization`` and — for temporary credentials — ``x-amz-security-token``.
    ``host`` is *signed* but omitted from the result, since the HTTP client sets
    it from the URL (and it must match, which it does for the same URL).

    Args:
        method: HTTP method (e.g. ``"POST"``).
        url: Absolute upstream URL being signed.
        payload: The exact request body bytes the signature must cover.
        region: AWS region (e.g. ``"us-east-1"``).
        service: AWS service name (``"bedrock"`` for the Bedrock runtime).
        credentials: Access/secret key and optional session token.
        headers: Extra request headers to include in the signature.
        now: Signing timestamp; defaults to the current UTC time (injectable for
            deterministic tests).
    """
    now = now or datetime.datetime.now(datetime.timezone.utc)
    amz_date = now.strftime("%Y%m%dT%H%M%SZ")
    date_stamp = now.strftime("%Y%m%d")

    split = urlsplit(url)
    signed = {k.lower(): v for k, v in (headers or {}).items()}
    signed["host"] = split.netloc
    signed["x-amz-date"] = amz_date
    if credentials.session_token:
        signed["x-amz-security-token"] = credentials.session_token

    canon = sorted(
        (name, " ".join(str(value).split()))
        for name, value in signed.items()
        if name not in _UNSIGNED_HEADERS
    )
    canonical_headers = "".join(f"{name}:{value}\n" for name, value in canon)
    signed_headers = ";".join(name for name, _ in canon)

    canonical_request = "\n".join(
        [
            method.upper(),
            _canonical_uri(split.path),
            _canonical_query(split.query),
            canonical_headers,
            signed_headers,
            _sha256_hex(payload),
        ]
    )

    scope = f"{date_stamp}/{region}/{service}/{_AWS4_REQUEST}"
    string_to_sign = "\n".join(
        [_ALGORITHM, amz_date, scope, _sha256_hex(canonical_request.encode("utf-8"))]
    )
    signature = hmac.new(
        _signing_key(credentials.secret_key, date_stamp, region, service),
        string_to_sign.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()

    authorization = (
        f"{_ALGORITHM} Credential={credentials.access_key}/{scope}, "
        f"SignedHeaders={signed_headers}, Signature={signature}"
    )

    out = {name: value for name, value in signed.items() if name != "host"}
    out["authorization"] = authorization
    return out
