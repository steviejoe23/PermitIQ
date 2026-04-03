"""
Shared fixtures for PermitIQ integration tests.
"""

import httpx
import pytest

API_URL = "http://127.0.0.1:8000"

# Primary test parcel: 1100051000 — 57 Centre Street, Roxbury
PRIMARY_PARCEL = "1100051000"

# Secondary test parcel: 0100001000 — East Boston
SECONDARY_PARCEL = "0100001000"


@pytest.fixture(scope="session")
def client():
    """HTTP client that talks to the running API on port 8000."""
    with httpx.Client(base_url=API_URL, timeout=30) as c:
        # Smoke test — fail fast if API is not running
        try:
            r = c.get("/health")
            r.raise_for_status()
        except httpx.ConnectError:
            pytest.skip("API not running on port 8000 — start it with: cd api && uvicorn main:app --port 8000")
        yield c
