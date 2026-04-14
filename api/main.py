"""
PermitIQ API v3 — Boston Zoning Intelligence & ZBA Risk Assessment Engine

App shell: FastAPI creation, middleware, startup event, router includes, error handler.
All business logic lives in route modules under api/routes/.
"""

import os
import sys
import time
import logging
import traceback
from collections import defaultdict

from fastapi import FastAPI, HTTPException, Security, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

# model_classes must be importable as top-level module for pickle deserialization
_services_dir = os.path.join(os.path.dirname(__file__), 'services')
if _services_dir not in sys.path:
    sys.path.insert(0, _services_dir)
try:
    from model_classes import StackingEnsemble, ManualCalibratedModel
except ImportError:
    StackingEnsemble = ManualCalibratedModel = None


# =========================
# STRUCTURED LOGGING
# =========================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("permitiq")


# =========================
# APP CREATION
# =========================

app = FastAPI(
    title="PermitIQ API",
    description="""Boston Zoning Intelligence & ZBA Risk Assessment Engine.

Quantifies the risk of ZBA approval/denial for development projects in Boston,
trained on 7,500+ real ZBA decisions with leakage-free pre-hearing features.

## Key Endpoints
- **Search** — Look up any Boston address and see ZBA history
- **Risk Assessment** — ML-powered approval probability for a proposed project
- **Compare** — What-if scenario analysis with real model deltas
- **Recommend** — Find the best parcels for your project type
- **Market Intel** — Attorney leaderboards, variance stats, neighborhood rankings, trends

## Authentication
Set X-API-Key header. Prediction endpoints always require a key when auth is enabled.
Tiers: free (5 predictions/day), pro (unlimited), enterprise (unlimited + batch).
If no keys are configured, the API runs in open development mode.

## Important
All probabilities are statistical risk assessments, not predictions or guarantees.
Consult a qualified zoning attorney before making financial decisions.
""",
    version="3.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

_allowed_origins = os.environ.get("PERMITIQ_CORS_ORIGINS", "").split(",")
_allowed_origins = [o.strip() for o in _allowed_origins if o.strip()]
if not _allowed_origins:
    _allowed_origins = [
        "http://localhost:8501",
        "http://127.0.0.1:8501",
        "https://permitiq-boston.streamlit.app",
        "https://permitiq.dev",
        "https://www.permitiq.dev",
    ]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "X-API-Key", "Authorization"],
)


# =========================
# AUTHENTICATION (api/services/auth.py)
# =========================

from api.services.auth import (
    init_auth,
    verify_api_key,
    api_key_header,
    AuthMiddleware,
    is_auth_enabled,
)

# Initialize auth keys from environment
init_auth()

# Add auth + rate limiting middleware
app.add_middleware(AuthMiddleware)


# =========================
# REQUEST LOGGING MIDDLEWARE
# =========================

class RequestLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        start = time.perf_counter()
        response = await call_next(request)
        elapsed_ms = (time.perf_counter() - start) * 1000
        if request.url.path != "/health":
            logger.info("%s %s %d %.0fms", request.method, request.url.path, response.status_code, elapsed_ms)
        return response


app.add_middleware(RequestLoggingMiddleware)


# =========================
# ROUTER INCLUDES
# =========================

from api.constants import VARIANCE_TYPES, PROJECT_TYPES

# Core routes
from api.routes.parcels import router as parcels_router
from api.routes.search import router as search_router
from api.routes.zoning import router as zoning_router
from api.routes.prediction import router as prediction_router
from api.routes.platform import router as platform_router
from api.routes.recommend import router as recommend_router

app.include_router(parcels_router)
app.include_router(search_router)
app.include_router(zoning_router)
app.include_router(prediction_router)
app.include_router(platform_router)
app.include_router(recommend_router)

# Market intel (existing module with init() pattern)
market_init = None
try:
    from api.routes.market_intel import router as market_router, init as market_init
    app.include_router(market_router)
    logger.info("Market intel router included")
except Exception as e:
    logger.warning("Market intel router not loaded: %s", e)

# Attorney routes (existing module with init() pattern)
attorney_init = None
try:
    from api.routes.attorneys import router as attorney_router, init as attorney_init
    app.include_router(attorney_router)
    logger.info("Attorney router included")
except Exception as e:
    logger.warning("Attorney router not loaded: %s", e)

# Filing strategy routes
filing_strategy_init = None
try:
    from api.routes.filing_strategy import router as filing_router, init as filing_strategy_init
    app.include_router(filing_router)
    logger.info("Filing strategy router included")
except Exception as e:
    logger.warning("Filing strategy router not loaded: %s", e)

# Board member routes
try:
    from api.routes.board_members import router as board_router, init as board_init
    app.include_router(board_router)
    board_init()
    logger.info("Board members router included")
except Exception as e:
    logger.warning("Board members router not loaded: %s", e)

# Opposition risk routes
try:
    from api.routes.opposition import router as opposition_router, init as opposition_init
    app.include_router(opposition_router)
    opposition_init()
    logger.info("Opposition router included")
except Exception as e:
    logger.warning("Opposition router not loaded: %s", e)

# Parcel risk score routes
try:
    from api.routes.risk_score import router as risk_router, init as risk_init
    app.include_router(risk_router)
    risk_init()
    logger.info("Risk score router included")
except Exception as e:
    logger.warning("Risk score router not loaded: %s", e)

# Hearing prep report routes
try:
    from api.routes.hearing_prep import router as hearing_prep_router
    app.include_router(hearing_prep_router)
    logger.info("Hearing prep router included")
except Exception as e:
    logger.warning("Hearing prep router not loaded: %s", e)


# =========================
# STARTUP EVENT
# =========================

@app.on_event("startup")
def load_data():
    """Load all data at startup — delegates to data_loader module."""
    from api.services.data_loader import load_all
    load_all(
        market_init=market_init,
        attorney_init=attorney_init,
        filing_strategy_init=filing_strategy_init,
        variance_types=VARIANCE_TYPES,
        project_types=PROJECT_TYPES,
    )


# =========================
# GLOBAL ERROR HANDLER
# =========================

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Catch unhandled exceptions and return a clean JSON error."""
    logger.error("Unhandled error on %s: %s", request.url, traceback.format_exc())
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": "An unexpected error occurred. Please try again or contact support.",
            "path": str(request.url.path)
        }
    )
