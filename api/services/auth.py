"""
PermitIQ Authentication & Authorization Module

Provides:
- Multi-key API authentication with tier-based access (free/pro/enterprise)
- Per-key rate limiting with tier-specific limits
- Usage logging for billing data
- Development mode (no auth) when no keys are configured

Environment variables:
- PERMITIQ_API_KEYS: comma-separated "key:tier" pairs (e.g., "abc123:pro,def456:enterprise")
- PERMITIQ_API_KEY: single key for backwards compatibility (treated as "pro" tier)
- PERMITIQ_REQUIRE_AUTH: if "true", non-public non-prediction endpoints also require auth
"""

from __future__ import annotations

import os
import time
import logging
from collections import defaultdict
from datetime import datetime, timezone
from typing import Optional, Tuple, Dict

from fastapi import HTTPException, Security
from fastapi.security import APIKeyHeader
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger("permitiq")

# ---------------------
# Configuration
# ---------------------

# Endpoints that never require authentication
PUBLIC_ENDPOINTS = frozenset({
    "/health",
    "/docs",
    "/redoc",
    "/openapi.json",
    "/stats",
})

# Endpoints that always require authentication (when auth is enabled)
PREDICTION_ENDPOINTS = frozenset({
    "/analyze_proposal",
    "/batch_predict",
    "/compare",
})

# Tier definitions: rate limits and access rules
TIER_CONFIG = {
    "free": {
        "predictions_per_day": 5,
        "requests_per_minute": 30,
        "allow_batch": False,
        "description": "Free tier — basic endpoints, 5 predictions/day",
    },
    "pro": {
        "predictions_per_day": None,  # unlimited
        "requests_per_minute": 120,
        "allow_batch": False,
        "description": "Pro tier — all endpoints, unlimited predictions",
    },
    "enterprise": {
        "predictions_per_day": None,  # unlimited
        "requests_per_minute": 600,
        "allow_batch": True,
        "description": "Enterprise tier — everything including batch predictions",
    },
}

# ---------------------
# Key store
# ---------------------

# Maps full API key -> {"tier": str, "key_id": str (last 4 chars)}
_api_keys: dict[str, dict] = {}

# Per-key rate tracking: key -> list of timestamps
_rate_buckets: dict[str, list[float]] = defaultdict(list)

# Per-key daily prediction count: key -> {"date": str, "count": int}
_prediction_counts: dict[str, dict] = defaultdict(lambda: {"date": "", "count": 0})

# Whether auth is configured at all
_auth_enabled: bool = False

# Whether non-public, non-prediction endpoints require auth
_require_auth_all: bool = False

# Usage log file handle
_usage_log_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "usage.log")


def _mask_key(key: str) -> str:
    """Return last 4 characters of key for logging."""
    if len(key) <= 4:
        return "****"
    return "..." + key[-4:]


def init_auth():
    """
    Initialize authentication from environment variables.
    Call this at startup.
    """
    global _auth_enabled, _require_auth_all

    # Load multi-key config: PERMITIQ_API_KEYS="key1:tier1,key2:tier2"
    multi_keys = os.environ.get("PERMITIQ_API_KEYS", "").strip()
    if multi_keys:
        for entry in multi_keys.split(","):
            entry = entry.strip()
            if not entry:
                continue
            if ":" in entry:
                key, tier = entry.rsplit(":", 1)
                tier = tier.strip().lower()
            else:
                key = entry
                tier = "pro"
            key = key.strip()
            if tier not in TIER_CONFIG:
                logger.warning("Unknown tier '%s' for key %s — defaulting to 'free'", tier, _mask_key(key))
                tier = "free"
            _api_keys[key] = {"tier": tier, "key_id": _mask_key(key)}
            logger.info("Registered API key %s (tier: %s)", _mask_key(key), tier)

    # Backwards compatibility: single PERMITIQ_API_KEY
    single_key = os.environ.get("PERMITIQ_API_KEY", "").strip()
    if single_key and single_key not in _api_keys:
        _api_keys[single_key] = {"tier": "pro", "key_id": _mask_key(single_key)}
        logger.info("Registered legacy API key %s (tier: pro)", _mask_key(single_key))

    _auth_enabled = len(_api_keys) > 0
    _require_auth_all = os.environ.get("PERMITIQ_REQUIRE_AUTH", "").strip().lower() == "true"

    if _auth_enabled:
        logger.info("Authentication enabled — %d API key(s) registered", len(_api_keys))
    else:
        logger.info("Authentication disabled — development mode (no API keys configured)")


def is_auth_enabled() -> bool:
    """Check if authentication is currently active."""
    return _auth_enabled


# ---------------------
# Header extraction
# ---------------------

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(key: str = Security(api_key_header)):
    """
    FastAPI dependency for verifying API keys.
    Returns None in dev mode (no keys configured).
    Returns the key info dict if valid.
    Raises HTTPException(403) if invalid.
    """
    if not _auth_enabled:
        return None  # Dev mode — open access

    if not key:
        raise HTTPException(
            status_code=403,
            detail="API key required. Set X-API-Key header.",
        )

    key_info = _api_keys.get(key)
    if key_info is None:
        raise HTTPException(
            status_code=403,
            detail="Invalid API key.",
        )

    return key_info


# ---------------------
# Usage logging
# ---------------------

def _log_usage(key_id: str, endpoint: str, tier: str, method: str = "GET"):
    """Append a usage log entry to api/usage.log."""
    try:
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        line = f"{ts}\t{key_id}\t{tier}\t{method}\t{endpoint}\n"
        with open(_usage_log_path, "a") as f:
            f.write(line)
    except Exception:
        pass  # Never let logging failures break the request


# ---------------------
# Rate limiting helpers
# ---------------------

def _check_rate_limit(key: str, tier: str) -> tuple[bool, int, int]:
    """
    Check per-minute rate limit for a key.
    Returns (allowed, remaining, reset_seconds).
    """
    config = TIER_CONFIG.get(tier, TIER_CONFIG["free"])
    limit = config["requests_per_minute"]

    now = time.time()
    bucket = _rate_buckets[key]
    # Prune old entries
    bucket[:] = [t for t in bucket if now - t < 60]
    remaining = max(0, limit - len(bucket))
    reset_seconds = int(60 - (now - bucket[0])) if bucket else 60

    if len(bucket) >= limit:
        return False, 0, reset_seconds

    bucket.append(now)
    return True, remaining - 1, reset_seconds


def _check_prediction_limit(key: str, tier: str) -> tuple[bool, int | None]:
    """
    Check daily prediction limit for a key.
    Returns (allowed, remaining_or_None_if_unlimited).
    """
    config = TIER_CONFIG.get(tier, TIER_CONFIG["free"])
    daily_limit = config["predictions_per_day"]

    if daily_limit is None:
        return True, None  # unlimited

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    entry = _prediction_counts[key]

    # Reset if new day
    if entry["date"] != today:
        entry["date"] = today
        entry["count"] = 0

    if entry["count"] >= daily_limit:
        return False, 0

    entry["count"] += 1
    return True, daily_limit - entry["count"]


def _is_prediction_endpoint(path: str) -> bool:
    """Check if path is a prediction endpoint."""
    return path in PREDICTION_ENDPOINTS


def _is_public_endpoint(path: str) -> bool:
    """Check if path is a public endpoint (no auth needed)."""
    # Exact match or prefix match for docs
    if path in PUBLIC_ENDPOINTS:
        return True
    # FastAPI serves docs assets under these paths
    if path.startswith("/docs") or path.startswith("/redoc"):
        return True
    return False


# ---------------------
# Auth + Rate Limit Middleware
# ---------------------

class AuthMiddleware(BaseHTTPMiddleware):
    """
    Combined authentication and rate limiting middleware.

    Logic:
    - Public endpoints (/health, /docs, /redoc, /openapi.json, /stats): always allowed
    - If auth disabled (no keys configured): everything allowed (dev mode)
    - Prediction endpoints: always require valid API key
    - Other endpoints: require auth only if PERMITIQ_REQUIRE_AUTH=true
    - Rate limiting applied per API key (tier-specific limits)
    - Prediction daily limits enforced for free tier
    """

    async def dispatch(self, request: Request, call_next):
        path = request.url.path
        method = request.method

        # Public endpoints — always allowed, no auth
        if _is_public_endpoint(path):
            return await call_next(request)

        # Dev mode — no auth configured, allow everything with IP-based rate limiting
        if not _auth_enabled:
            return await call_next(request)

        # --- Auth is enabled from here ---

        api_key = request.headers.get("X-API-Key", "")
        is_prediction = _is_prediction_endpoint(path)

        # Determine if auth is required for this endpoint
        auth_required = is_prediction or _require_auth_all

        if not auth_required:
            # Non-prediction, non-required endpoint — allow without key but apply
            # rate limiting if key is present
            if not api_key:
                return await call_next(request)

        # Auth required or key was provided — validate it
        if not api_key:
            return JSONResponse(
                status_code=403,
                content={"error": "API key required. Set X-API-Key header."},
            )

        key_info = _api_keys.get(api_key)
        if key_info is None:
            return JSONResponse(
                status_code=403,
                content={"error": "Invalid API key."},
            )

        tier = key_info["tier"]
        key_id = key_info["key_id"]

        # Batch prediction — enterprise only
        if path == "/batch_predict" and tier != "enterprise":
            return JSONResponse(
                status_code=403,
                content={
                    "error": f"Batch predictions require enterprise tier. Current tier: {tier}.",
                },
            )

        # Per-key rate limiting
        allowed, remaining, reset = _check_rate_limit(api_key, tier)
        if not allowed:
            return JSONResponse(
                status_code=429,
                content={"error": "Rate limit exceeded. Try again later."},
                headers={
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": str(reset),
                    "Retry-After": str(reset),
                },
            )

        # Daily prediction limit (free tier)
        if is_prediction:
            pred_allowed, pred_remaining = _check_prediction_limit(api_key, tier)
            if not pred_allowed:
                return JSONResponse(
                    status_code=429,
                    content={
                        "error": "Daily prediction limit reached for free tier (5/day). Upgrade to pro for unlimited predictions.",
                    },
                )

        # Log usage
        _log_usage(key_id, path, tier, method)

        # Execute request and add rate limit headers to response
        response = await call_next(request)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        response.headers["X-RateLimit-Reset"] = str(reset)
        response.headers["X-RateLimit-Tier"] = tier

        return response
