"""Argon2 and API Key authentication middleware for FastMCP HTTP/SSE servers."""

from __future__ import annotations

import os
import secrets
from typing import TYPE_CHECKING
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import PlainTextResponse, Response
from starlette.types import ASGIApp

if TYPE_CHECKING:
    from argon2 import PasswordHasher

try:
    from argon2 import PasswordHasher as _PasswordHasher
    from argon2.exceptions import VerificationError as _VerificationError
    from argon2.exceptions import VerifyMismatchError as _VerifyMismatchError

    HAS_ARGON2 = True
except ImportError:
    _PasswordHasher = None
    _VerificationError = Exception
    _VerifyMismatchError = Exception
    HAS_ARGON2 = False


class APIKeyAuthenticator:
    """Validates incoming API keys against an Argon2 hash or plain secret key.

    Directly models aurora-mcp (pkg/auth/auth.go):
    - Reads rawHash from direct string or '${ENV_VAR}' environment variable syntax.
    - Validates x-api-key / Authorization Bearer token against an Argon2 hash.
    - If target is a plain secret key, performs constant-time string comparison.
    """

    def __init__(self, raw_hash_or_key: str | None = None) -> None:
        self.target_hash: str = ""
        self._hasher: PasswordHasher | None = _PasswordHasher() if _PasswordHasher is not None else None

        if raw_hash_or_key:
            trimmed = raw_hash_or_key.strip()
            if trimmed.startswith("${") and trimmed.endswith("}"):
                env_key = trimmed[2:-1].strip()
                self.target_hash = os.getenv(env_key, "").strip()
            else:
                self.target_hash = trimmed

    @classmethod
    def from_env(cls, env_var_name: str = "FUNCPie_API_KEY") -> APIKeyAuthenticator:
        """Create an authenticator reading from the specified environment variable."""
        key = os.getenv(env_var_name) or os.getenv("MCP_API_KEY")
        return cls(key)

    def is_enabled(self) -> bool:
        """Return True if an API key requirement is configured and non-empty."""
        return len(self.target_hash) > 0

    def verify(self, candidate_key: str) -> bool:
        """Verify candidate key against Argon2 target hash (or plain key)."""
        if not self.is_enabled() or not candidate_key:
            return False

        # Argon2 hash verification (starts with $argon2)
        if self.target_hash.startswith("$argon2"):
            if self._hasher is not None:
                try:
                    return self._hasher.verify(self.target_hash, candidate_key)
                except (_VerificationError, _VerifyMismatchError):
                    return False
            return False

        # Plain key constant-time comparison
        return secrets.compare_digest(candidate_key, self.target_hash)

    def extract_key(self, request: Request) -> str | None:
        """Extract API key from HTTP request headers (X-API-Key or Authorization: Bearer)."""
        key = request.headers.get("x-api-key") or request.headers.get("X-API-Key")
        if key:
            return key.strip()

        auth_header = request.headers.get("authorization")
        if auth_header:
            parts = auth_header.strip().split()
            if len(parts) == 2 and parts[0].lower() == "bearer":
                return parts[1].strip()

        return None


class APIKeyMiddleware(BaseHTTPMiddleware):
    """Starlette middleware intercepting HTTP requests to enforce Argon2/API Key authentication."""

    def __init__(self, app: ASGIApp, authenticator: APIKeyAuthenticator) -> None:
        super().__init__(app)
        self.authenticator = authenticator

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        if not self.authenticator.is_enabled():
            return await call_next(request)

        key = self.authenticator.extract_key(request)
        if not key or not self.authenticator.verify(key):
            return PlainTextResponse("unauthorized", status_code=401)

        return await call_next(request)
