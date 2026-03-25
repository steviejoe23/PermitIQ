"""
PermitIQ API v2 — Full Feature Set
FastAPI backend for Boston zoning intelligence
"""

from fastapi import FastAPI, HTTPException, Depends, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field
from typing import List, Optional
import geopandas as gpd
import pandas as pd
import numpy as np
import joblib
import re
import os
import logging
import traceback
from functools import lru_cache
try:
    from services.database import query_parcel, query_parcels_nearby, db_available
except ImportError:
    try:
        from api.services.database import query_parcel, query_parcels_nearby, db_available
    except ImportError:
        query_parcel = query_parcels_nearby = None
        db_available = lambda: False

# =========================
# STRUCTURED LOGGING
# =========================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("permitiq")

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

## Important
All probabilities are statistical risk assessments, not predictions or guarantees.
Consult a qualified zoning attorney before making financial decisions.
""",
    version="3.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# API KEY AUTHENTICATION (optional)
# =========================
# Set PERMITIQ_API_KEY env var to enable authentication.
# If not set, API runs in open/demo mode.

API_KEY = os.environ.get("PERMITIQ_API_KEY")
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(key: str = Security(api_key_header)):
    """Verify API key if authentication is enabled."""
    if API_KEY is None:
        return None  # Auth disabled — open mode
    if key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid or missing API key. Set X-API-Key header.")
    return key


# =========================
# REQUEST LOGGING & RATE LIMITING MIDDLEWARE
# =========================
import time
from collections import defaultdict
from starlette.middleware.base import BaseHTTPMiddleware

# Simple in-memory rate limiter: max requests per IP per minute
RATE_LIMIT = int(os.environ.get('RATE_LIMIT_PER_MINUTE', 120))
_rate_buckets = defaultdict(list)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        # Rate limiting (skip health checks and localhost/testing)
        client_ip = request.client.host if request.client else "unknown"
        if request.url.path != "/health" and client_ip not in ("127.0.0.1", "localhost", "testclient"):
            now = time.time()
            # Clean old entries (older than 60s)
            _rate_buckets[client_ip] = [t for t in _rate_buckets[client_ip] if now - t < 60]
            if len(_rate_buckets[client_ip]) >= RATE_LIMIT:
                logger.warning("Rate limit exceeded for %s", client_ip)
                from starlette.responses import JSONResponse
                return JSONResponse(
                    status_code=429,
                    content={"error": "Rate limit exceeded. Max 60 requests per minute."}
                )
            _rate_buckets[client_ip].append(now)

        start = time.perf_counter()
        response = await call_next(request)
        elapsed_ms = (time.perf_counter() - start) * 1000
        if request.url.path != "/health":
            logger.info("%s %s %d %.0fms", request.method, request.url.path, response.status_code, elapsed_ms)
        return response


app.add_middleware(RequestLoggingMiddleware)


# =========================
# LOAD DATA (ON STARTUP)
# =========================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

GEOJSON_PATH = os.path.join(BASE_DIR, "../boston_parcels_zoning.geojson")
ZBA_DATA_PATH = os.path.join(BASE_DIR, "../zba_cases_cleaned.csv")
MODEL_PATH = os.path.join(BASE_DIR, "zba_model.pkl")
PROPERTY_PATH = os.path.join(BASE_DIR, "../property_assessment_fy2026.csv")

gdf = None
zba_df = None
model_package = None
parcel_addr_df = None  # address→parcel lookup


def safe_float(val, default=0.0):
    """Convert to float, replacing NaN/None with default."""
    if val is None:
        return default
    try:
        f = float(val)
        return default if np.isnan(f) or np.isinf(f) else f
    except (ValueError, TypeError):
        return default

def safe_int(val, default=0):
    """Convert to int, replacing NaN/None with default."""
    if val is None:
        return default
    try:
        f = float(val)
        return default if np.isnan(f) or np.isinf(f) else int(f)
    except (ValueError, TypeError):
        return default

def safe_str(val, default=""):
    """Convert to string, replacing NaN/None with default."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return default
    return str(val)


@app.on_event("startup")
def load_data():
    global gdf, zba_df, model_package, parcel_addr_df

    try:
        gdf = gpd.read_file(GEOJSON_PATH)
        gdf["parcel_id"] = gdf["parcel_id"].astype(str)
        gdf = gdf.set_index("parcel_id", drop=False)  # Index for O(1) lookups
        logger.info("GeoJSON loaded (%d parcels)", len(gdf))
    except Exception as e:
        logger.error("Failed to load GeoJSON: %s", e)

    try:
        zba_df = pd.read_csv(ZBA_DATA_PATH, low_memory=False)
        # Pre-compute normalized addresses for fast search
        if 'address_clean' in zba_df.columns:
            zba_df['_addr_norm'] = zba_df['address_clean'].apply(normalize_address)
        logger.info("ZBA dataset loaded (%d cases)", len(zba_df))
    except Exception as e:
        logger.error("Failed to load ZBA dataset: %s", e)

    try:
        model_package = joblib.load(MODEL_PATH)
        model_name = model_package.get('model_name', 'unknown')
        auc = model_package.get('auc_score', 0)
        n_features = len(model_package.get('feature_cols', []))
        logger.info("ML model loaded (%s, AUC: %.4f, %d features)", model_name, auc, n_features)
    except Exception as e:
        logger.warning("No trained model found, using fallback logic: %s", e)

    # Load property assessment for address→parcel geocoding (lightweight: just PID + address)
    try:
        _pa = pd.read_csv(PROPERTY_PATH, usecols=['PID', 'ST_NUM', 'ST_NAME'], low_memory=False)
        _pa = _pa.dropna(subset=['PID', 'ST_NUM', 'ST_NAME'])
        # Clean street numbers (remove .0 from float conversion)
        _pa['ST_NUM'] = _pa['ST_NUM'].astype(str).str.strip().str.replace(r'\.0$', '', regex=True)
        _pa['ST_NAME'] = _pa['ST_NAME'].astype(str).str.strip()
        _pa['address'] = _pa['ST_NUM'] + ' ' + _pa['ST_NAME']
        _pa['_addr_norm'] = _pa['address'].apply(normalize_address)
        # Pad PID to 10 digits to match GeoJSON parcel_id
        _pa['parcel_id'] = _pa['PID'].astype(str).str.zfill(10)
        parcel_addr_df = _pa[['parcel_id', 'address', '_addr_norm']].drop_duplicates('parcel_id')
        logger.info("Property address lookup loaded (%d parcels)", len(parcel_addr_df))
    except Exception as e:
        logger.warning("Could not load property assessment for geocoding: %s", e)

    # Initialize market intel router with loaded data
    if market_init is not None and zba_df is not None:
        market_init(zba_df, VARIANCE_TYPES, PROJECT_TYPES)
        logger.info("Market intel router initialized with %d cases", len(zba_df))


# =========================
# PARCEL LOOKUP
# =========================

@app.get("/parcels/{parcel_id}", tags=["Parcels"])
def get_parcel(parcel_id: str):
    """Look up zoning details and geometry for a Boston parcel by its 10-digit ID."""

    # Try PostGIS first (fast, low memory)
    if db_available() and query_parcel is not None:
        db_row = query_parcel(parcel_id)
        if db_row is not None:
            result = {
                "parcel_id": parcel_id,
                "zoning_code": db_row.get("primary_zoning", ""),
                "district": db_row.get("all_zoning_codes", ""),
                "article": db_row.get("article", ""),
                "multi_zoning": db_row.get("multi_zoning", False),
                "geometry": db_row.get("geometry"),
                "source": "postgis",
            }
            # Enrich with ZBA history
            _enrich_parcel_result(result, parcel_id)
            return result

    # Fallback to in-memory GeoJSON
    if gdf is None:
        raise HTTPException(status_code=500, detail="No parcel data available (PostGIS and GeoJSON both unavailable)")

    row = gdf.loc[[parcel_id]] if parcel_id in gdf.index else gdf.iloc[0:0]
    if row.empty:
        raise HTTPException(status_code=404, detail="Parcel not found")

    row = row.iloc[0]
    result = {
        "parcel_id": parcel_id,
        "zoning_code": str(row.get("primary_zoning") or ""),
        "district": str(row.get("districts") or ""),
        "article": str(row.get("article") or ""),
        "volume": str(row.get("volume") or ""),
        "zoning_summary": str(row.get("summary") or ""),
        "multi_zoning": bool(row.get("multi_zoning")),
        "zoning_count": int(row.get("zoning_count") or 0),
        "geometry": row.geometry.__geo_interface__,
        "source": "geojson",
    }
    _enrich_parcel_result(result, parcel_id)
    return result


def _enrich_parcel_result(result: dict, parcel_id: str):
    """Add ZBA case history to a parcel result."""
    district = result.get("district", "")
    if zba_df is not None and 'zoning_district' in zba_df.columns:
        ward_lookup = zba_df[
            (zba_df['zoning_district'] == district) & zba_df['ward'].notna()
        ]['ward']
        if not ward_lookup.empty:
            result["ward"] = str(int(ward_lookup.mode().iloc[0]))

    if zba_df is not None and 'pa_parcel_id' in zba_df.columns:
        pid_float = safe_float(parcel_id, default=None)
        if pid_float is not None:
            matches = zba_df[zba_df['pa_parcel_id'] == pid_float]
            if not matches.empty:
                result["zba_cases"] = len(matches)
                addr = matches['address_clean'].dropna().iloc[0] if 'address_clean' in matches.columns and matches['address_clean'].notna().any() else None
                if addr:
                    result["address"] = str(addr)


@app.get("/parcels/{parcel_id}/nearby_cases", tags=["Parcels"])
def nearby_cases(parcel_id: str, radius_m: int = 500, limit: int = 10):
    """Find ZBA cases near a parcel. Uses parcel centroid + haversine distance."""
    if gdf is None:
        raise HTTPException(status_code=500, detail="GeoJSON not loaded")
    if zba_df is None:
        raise HTTPException(status_code=500, detail="ZBA data not loaded")

    row = gdf.loc[[parcel_id]] if parcel_id in gdf.index else gdf.iloc[0:0]
    if row.empty:
        raise HTTPException(status_code=404, detail="Parcel not found")

    centroid = row.iloc[0].geometry.centroid
    parcel_lat, parcel_lon = centroid.y, centroid.x

    # Find cases with coordinates (from property assessment lat/lon or address geocoding)
    # For now, match by district — same neighborhood cases
    district = str(row.iloc[0].get("districts") or "")

    if not district:
        return {"parcel_id": parcel_id, "cases": [], "total": 0}

    z_col = 'zoning_district' if 'zoning_district' in zba_df.columns else None
    if z_col is None:
        return {"parcel_id": parcel_id, "cases": [], "total": 0}

    nearby = zba_df[
        (zba_df[z_col] == district) &
        (zba_df['decision_clean'].notna())
    ].sort_values('case_number', ascending=False).head(limit)

    cases = []
    for _, c in nearby.iterrows():
        cases.append({
            "case_number": str(c.get('case_number', '')),
            "address": str(c.get('address_clean', '')),
            "decision": str(c.get('decision_clean', '')),
            "ward": str(c.get('ward', '')),
            "date": str(c.get('hearing_date', c.get('filing_date', '')))[:10],
        })

    return {
        "parcel_id": parcel_id,
        "district": district,
        "parcel_lat": parcel_lat,
        "parcel_lon": parcel_lon,
        "cases": cases,
        "total": len(cases),
    }


@app.get("/geocode", tags=["Parcels"])
def geocode_address(q: str):
    """
    Find a parcel ID from a street address. Uses the property assessment database (184K parcels).
    Returns the best-matching parcel(s) with zoning info.
    """
    if parcel_addr_df is None:
        raise HTTPException(status_code=503, detail="Property address data not loaded")
    if gdf is None:
        raise HTTPException(status_code=500, detail="GeoJSON not loaded")

    q_norm = normalize_address(q)
    if len(q_norm) < 3:
        return {"query": q, "results": []}

    # Word-boundary matching: each word in query must appear as a whole word
    words = q_norm.split()
    mask = pd.Series(True, index=parcel_addr_df.index)
    for word in words:
        if len(word) > 1:
            # Use word boundary for numbers to avoid "75" matching "175"
            if word.isdigit():
                mask = mask & parcel_addr_df['_addr_norm'].str.contains(r'(?:^|\s)' + re.escape(word) + r'(?:\s|$)', na=False, regex=True)
            else:
                mask = mask & parcel_addr_df['_addr_norm'].str.contains(word, na=False, regex=False)

    matches = parcel_addr_df[mask].head(10)

    results = []
    for _, row in matches.iterrows():
        pid = row['parcel_id']
        entry = {"parcel_id": pid, "address": row['address']}

        # Enrich with zoning from GeoJSON
        geo_match = gdf.loc[[pid]] if pid in gdf.index else gdf.iloc[0:0]
        if not geo_match.empty:
            geo_row = geo_match.iloc[0]
            entry["zoning_code"] = str(geo_row.get("primary_zoning") or "")
            entry["district"] = str(geo_row.get("districts") or "")
        results.append(entry)

    return {"query": q, "results": results, "total": len(results)}


# =========================
# ADDRESS SEARCH
# =========================

def normalize_address(addr):
    """Normalize an address string for better matching."""
    if not addr:
        return ""
    addr = str(addr).lower().strip()
    # Remove ward info
    addr = re.sub(r',?\s*ward\s*\d+', '', addr)
    # Normalize street suffixes
    addr = re.sub(r'\bstreet\b', 'st', addr)
    addr = re.sub(r'\bavenue\b', 'ave', addr)
    addr = re.sub(r'\broad\b', 'rd', addr)
    addr = re.sub(r'\bdrive\b', 'dr', addr)
    addr = re.sub(r'\bboulevard\b', 'blvd', addr)
    addr = re.sub(r'\blane\b', 'ln', addr)
    addr = re.sub(r'\bcourt\b', 'ct', addr)
    addr = re.sub(r'\bplace\b', 'pl', addr)
    addr = re.sub(r'\bterrace\b', 'ter', addr)
    # Collapse whitespace
    addr = re.sub(r'\s+', ' ', addr).strip()
    return addr


@lru_cache(maxsize=256)
def _cached_search(q_norm: str) -> list:
    """Cache search results by normalized query. Returns list of result dicts."""
    return _do_search(q_norm)


def _do_search(q_norm: str) -> list:
    """Core search logic, separated for caching."""

    # Filter to rows with usable addresses (start with a digit, > 5 chars)
    addr_df = zba_df[
        zba_df['address_clean'].notna() &
        zba_df['address_clean'].str.match(r'^\d', na=False) &
        (zba_df['address_clean'].str.len() > 5)
    ]

    # Use pre-computed normalized addresses (computed at startup)
    addr_norm_col = '_addr_norm' if '_addr_norm' in addr_df.columns else None
    if addr_norm_col is None:
        addr_df = addr_df.copy()
        addr_df['_addr_norm'] = addr_df['address_clean'].apply(normalize_address)

    # Try exact substring match first
    mask = addr_df['_addr_norm'].str.contains(q_norm, na=False, regex=False)

    # If no results, try matching individual words
    if mask.sum() == 0:
        words = q_norm.split()
        mask = pd.Series(True, index=addr_df.index)
        for word in words:
            if len(word) > 2:
                mask = mask & addr_df['_addr_norm'].str.contains(word, na=False, regex=False)

    matches = addr_df[mask]

    # Aggregate by normalized address — show case count and approval rate
    # Use whichever zoning column exists
    zoning_col = 'zoning_clean' if 'zoning_clean' in matches.columns else (
        'zoning_district' if 'zoning_district' in matches.columns else (
        'zoning' if 'zoning' in matches.columns else None))
    # Use whichever date column exists
    date_col = 'hearing_date' if 'hearing_date' in matches.columns else (
        'filing_date' if 'filing_date' in matches.columns else None)

    agg_dict = {
        'address': ('address_clean', 'first'),
        'ward': ('ward', 'first'),
        'total_cases': ('case_number', 'count'),
        'approved': ('decision_clean', lambda x: (x == 'APPROVED').sum()),
        'denied': ('decision_clean', lambda x: (x == 'DENIED').sum()),
        'latest_case': ('case_number', 'last'),
    }
    if zoning_col:
        agg_dict['zoning'] = (zoning_col, 'first')
    if date_col:
        agg_dict['latest_date'] = (date_col, 'max')

    grouped = matches.groupby('_addr_norm').agg(**agg_dict).sort_values('total_cases', ascending=False).head(10)

    results = []
    for _, row in grouped.iterrows():
        total = row['approved'] + row['denied']
        result_item = {
            "address": str(row['address']),
            "ward": str(row['ward']) if pd.notna(row['ward']) else "",
            "zoning": str(row.get('zoning', '')) if pd.notna(row.get('zoning', '')) else "",
            "total_cases": int(row['total_cases']),
            "approved": int(row['approved']),
            "denied": int(row['denied']),
            "approval_rate": round(row['approved'] / total, 2) if total > 0 else None,
            "latest_date": str(row.get('latest_date', ''))[:10] if pd.notna(row.get('latest_date', None)) else "",
            "latest_case": str(row['latest_case']),
        }
        results.append(result_item)

    return results


@app.get("/search", tags=["Search"])
def search_address(q: str):
    """Search for addresses with ZBA history. Returns aggregated results per address."""
    if zba_df is None:
        raise HTTPException(status_code=500, detail="ZBA data not loaded")

    q_norm = normalize_address(q)
    if len(q_norm) < 2:
        return {"query": q, "results": [], "total_results": 0}

    results = _cached_search(q_norm)
    return {"query": q, "results": results, "total_results": len(results)}


@app.get("/address/{address}/cases", tags=["Search"])
def get_address_cases(address: str, limit: int = 20):
    """Get all ZBA cases for a specific address with full details."""
    if zba_df is None:
        raise HTTPException(status_code=500, detail="ZBA data not loaded")

    addr_norm = normalize_address(address)

    addr_df = zba_df[zba_df['address_clean'].notna()]

    # Use pre-computed normalized addresses
    if '_addr_norm' in addr_df.columns:
        matches = addr_df[addr_df['_addr_norm'].str.contains(addr_norm, na=False, regex=False)]
    else:
        addr_df = addr_df.copy()
        addr_df['_addr_norm'] = addr_df['address_clean'].apply(normalize_address)
        matches = addr_df[addr_df['_addr_norm'].str.contains(addr_norm, na=False, regex=False)]
    sort_col = 'hearing_date' if 'hearing_date' in matches.columns else ('filing_date' if 'filing_date' in matches.columns else 'case_number')
    matches = matches.sort_values(sort_col, ascending=False, na_position='last').head(limit)

    # Determine best zoning column
    z_col = 'zoning_clean' if 'zoning_clean' in matches.columns else ('zoning_district' if 'zoning_district' in matches.columns else 'zoning')
    d_col = 'hearing_date' if 'hearing_date' in matches.columns else ('filing_date' if 'filing_date' in matches.columns else None)

    cases = []
    for _, row in matches.iterrows():
        cases.append({
            "case_number": str(row.get('case_number') or ''),
            "address": str(row.get('address_clean') or ''),
            "decision": str(row.get('decision_clean') or ''),
            "ward": str(row.get('ward') or ''),
            "zoning": str(row.get(z_col) or ''),
            "date": str(row.get(d_col) or '')[:10] if d_col else '',
            "variances": str(row.get('variance_types') or ''),
            "has_attorney": bool(row.get('has_attorney', 0)),
            "project_type": ', '.join([
                pt.replace('proj_', '') for pt in [
                    'proj_addition', 'proj_new_construction', 'proj_renovation',
                    'proj_conversion', 'proj_demolition', 'proj_multi_family',
                    'proj_single_family', 'proj_mixed_use', 'proj_adu', 'proj_roof_deck'
                ] if row.get(pt, 0) == 1
            ]) or 'unknown',
        })

    return {
        "address": address,
        "cases": cases,
        "total": len(cases)
    }


# =========================
# RESPONSE MODELS (OpenAPI schemas)
# =========================

class HealthResponse(BaseModel):
    status: str
    geojson_loaded: bool
    zba_loaded: bool
    model_loaded: bool
    total_parcels: int
    total_cases: int
    model_name: Optional[str]
    model_auc: Optional[float]
    features: int

class SearchResult(BaseModel):
    address: str
    ward: str = ""
    zoning: str = ""
    total_cases: int
    approved: int
    denied: int
    approval_rate: Optional[float]
    latest_date: str = ""
    latest_case: str = ""

class PredictionResponse(BaseModel):
    parcel_id: str
    zoning: str
    district: str
    proposed_use: str
    project_type: str
    variances: List[str]
    has_attorney: bool
    approval_probability: float
    probability_range: List[float]
    confidence: str
    based_on_cases: int
    ward_approval_rate: Optional[float]
    key_factors: List[str]
    top_drivers: list
    similar_cases: list
    estimated_timeline_days: Optional[dict]
    model: str
    model_auc: float = 0
    total_training_cases: int = 0
    disclaimer: str


class WardStatsResponse(BaseModel):
    ward: str
    total_cases: int
    approved: int
    denied: int
    approval_rate: float


class RecommendationResult(BaseModel):
    parcel_id: str
    approval_probability: float
    zoning_code: str = ""
    district: str = ""
    lat: Optional[float] = None
    lon: Optional[float] = None


class RecommendationResponse(BaseModel):
    query: dict
    total_candidates: int
    results_found: int
    parcels: List[RecommendationResult]
    disclaimer: str


# =========================
# ZBA PREDICTION ENGINE
# =========================

# =========================
# PYDANTIC MODELS
# =========================

class ProposalInput(BaseModel):
    parcel_id: str = Field(default="", description="Boston parcel ID for zoning lookup")
    proposed_use: str = Field(default="", alias="use_type", description="Proposed use type (also accepts 'use_type')")
    variances: List[str] = Field(default_factory=list, description="List of variance types requested")
    project_type: Optional[str] = Field(default=None, description="Project type (addition, new_construction, etc.)")
    ward: Optional[str] = Field(default=None, description="Boston ward number")
    has_attorney: bool = Field(default=False, description="Whether project has legal representation")
    proposed_units: int = Field(default=0, ge=0, le=500, description="Number of proposed units")
    proposed_stories: int = Field(default=0, ge=0, le=50, description="Number of proposed stories")

    class Config:
        populate_by_name = True


VARIANCE_TYPES = [
    'height', 'far', 'lot_area', 'lot_frontage',
    'front_setback', 'rear_setback', 'side_setback',
    'parking', 'conditional_use', 'open_space', 'density', 'nonconforming'
]

PROJECT_TYPES = [
    'demolition', 'new_construction', 'addition', 'conversion',
    'renovation', 'subdivision', 'adu', 'roof_deck', 'parking',
    'single_family', 'multi_family', 'mixed_use'
]


def build_features(parcel_row, proposed_use: str, variances: list,
                   project_type: str = None, ward: str = None,
                   has_attorney: bool = False, proposed_units: int = 0,
                   proposed_stories: int = 0):
    """Build feature vector for ML model prediction — 69 clean features.

    NOTE: We intentionally exclude outcome features (votes, provisos, appeal_sustained)
    that would cause data leakage. Only pre-hearing information is used.
    """

    overall_rate = 0.76
    if model_package:
        overall_rate = model_package.get('overall_approval_rate', 0.76)

    # --- District (for article inference) ---
    district = str(parcel_row.get('districts') or '') if parcel_row is not None else ''

    # --- Variance features ---
    num_variances = len(variances)
    var_features = {}
    for vt in VARIANCE_TYPES:
        matched = any(
            vt in v.lower().replace(' ', '_') or v.lower().replace(' ', '_') in vt
            for v in variances
        )
        var_features[f'var_{vt}'] = int(matched)

    # --- Specific violation types (derived from variances) ---
    variances_lower = ' '.join(v.lower() for v in variances)
    excessive_far = int('far' in variances_lower or 'floor area' in variances_lower)
    insufficient_lot = int('lot_area' in variances_lower or 'lot area' in variances_lower)
    insufficient_frontage = int('frontage' in variances_lower)
    insufficient_yard = int('setback' in variances_lower or 'yard' in variances_lower)
    insufficient_parking = int('parking' in variances_lower)

    # --- Use type ---
    use_lower = proposed_use.lower() if proposed_use else ""
    is_residential = int(bool(re.search(r'residential|dwelling|family|condo|apartment|housing', use_lower)))
    is_commercial = int(bool(re.search(r'commercial|retail|office|restaurant|store|shop', use_lower)))

    # --- Project type ---
    proj_features = {}
    for pt in PROJECT_TYPES:
        if project_type:
            proj_features[f'proj_{pt}'] = int(pt in project_type.lower().replace(' ', '_'))
        else:
            proj_features[f'proj_{pt}'] = int(pt.replace('_', ' ') in use_lower)

    # --- Ward and zoning rates ---
    ward_rate = overall_rate
    zoning_rate = overall_rate

    zoning = str(parcel_row.get('primary_zoning') or '') if parcel_row is not None else ''

    if model_package:
        ward_rates = model_package.get('ward_approval_rates', {})
        zoning_rates = model_package.get('zoning_approval_rates', {})
        top_zoning_list = model_package.get('top_zoning', [])

        ward_rate = ward_rates.get(str(ward or 'unknown'), overall_rate)
        zoning_group = zoning if zoning in top_zoning_list else 'other'
        zoning_rate = zoning_rates.get(zoning_group, overall_rate)

    # --- Property data from parcel/assessment lookup ---
    lot_size = 0
    total_value = 0
    property_age = 0
    living_area = 0
    is_high_value = 0
    value_per_sqft = 0

    # Try to get property data if we have a parcel match in the ZBA dataset
    if parcel_row is not None and zba_df is not None:
        parcel_id = str(parcel_row.get('parcel_id', ''))
        if parcel_id:
            pa_match = zba_df[zba_df['pa_parcel_id'].astype(str) == parcel_id]
            if not pa_match.empty:
                rec = pa_match.iloc[0]
                lot_size = float(rec.get('lot_size_sf', 0) or 0)
                total_value = float(rec.get('total_value', 0) or 0)
                property_age = float(rec.get('property_age', 0) or 0)
                living_area = float(rec.get('living_area', 0) or 0)
                is_high_value = int(total_value > 1_000_000)
                value_per_sqft = total_value / lot_size if lot_size > 0 else 0

    # --- Prior permits lookup ---
    prior_permits = 0
    has_prior_permits = 0

    # Try to find prior permit data from the ZBA dataset for this address/parcel
    if zba_df is not None and parcel_row is not None:
        parcel_id = str(parcel_row.get('parcel_id', ''))
        if parcel_id:
            pa_match = zba_df[zba_df['pa_parcel_id'].astype(str) == parcel_id]
            if not pa_match.empty:
                rec = pa_match.iloc[0]
                prior_permits = float(rec.get('prior_permits', 0) or 0)
                has_prior_permits = int(prior_permits > 0)

    # --- Attorney win rate lookup ---
    attorney_win_rate = overall_rate
    if model_package:
        atty_rates = model_package.get('attorney_win_rates', {})
        if atty_rates:
            attorney_win_rate = overall_rate  # Default for new/unknown attorneys

    # --- Build feature dict — PRE-HEARING ONLY (matches training v3) ---
    features = {
        # Variance (13) — from application
        'num_variances': num_variances,
        **var_features,
        # Violation types (5) — from zoning analysis
        'excessive_far': excessive_far,
        'insufficient_lot': insufficient_lot,
        'insufficient_frontage': insufficient_frontage,
        'insufficient_yard': insufficient_yard,
        'insufficient_parking': insufficient_parking,
        # Use (2) — from application
        'is_residential': is_residential,
        'is_commercial': is_commercial,
        # Representation (2) — known at filing
        'has_attorney': int(has_attorney),
        'bpda_involved': 0,
        # Project types (12) — from application
        **proj_features,
        # Legal (4) — from zoning code
        'article_80': int(proposed_units > 15 or proposed_stories > 4),
        'is_conditional_use': var_features.get('var_conditional_use', 0),
        'is_variance': int(num_variances > 0),
        'num_articles': max(1, num_variances),
        # Scale (2) — from application
        'proposed_units': proposed_units,
        'proposed_stories': proposed_stories,
        # Location (3) — historical rates from training data
        'ward_approval_rate': ward_rate,
        'zoning_approval_rate': zoning_rate,
        'attorney_win_rate': attorney_win_rate,
        # Recency (1)
        'year_recency': max(0, 2026 - 2020),
        # Property (6) — from tax assessor
        'lot_size_sf': lot_size,
        'total_value': total_value,
        'property_age': property_age,
        'living_area': living_area,
        'is_high_value': is_high_value,
        'value_per_sqft': value_per_sqft,
        # Permits (2) — from building permits DB
        'prior_permits': prior_permits,
        'has_prior_permits': has_prior_permits,
        # Interactions (3)
        'interact_height_stories': var_features.get('var_height', 0) * proposed_stories,
        'interact_attorney_variances': int(has_attorney) * num_variances,
        'interact_highvalue_permits': is_high_value * has_prior_permits,
        # Log transforms (2)
        'lot_size_log': float(np.log1p(lot_size)),
        'total_value_log': float(np.log1p(total_value)),
    }

    return features


def get_similar_cases(ward, variances, project_type=None, limit=5):
    """Find similar historical ZBA cases using relevance scoring."""
    if zba_df is None or len(zba_df) == 0:
        return [], 0, 0.0

    df = zba_df[zba_df['decision_clean'].notna()].copy()

    # Compute relevance score for each case
    df['_relevance'] = 0.0

    # Ward match: +3 points
    if ward:
        try:
            ward_float = float(ward)
            df.loc[df['ward'] == ward_float, '_relevance'] += 3.0
        except (ValueError, TypeError):
            pass

    # Variance overlap: +2 points per matching variance
    if variances and 'variance_types' in df.columns:
        vt = df['variance_types'].fillna('').str.lower()
        for v in variances:
            df.loc[vt.str.contains(v.lower(), na=False), '_relevance'] += 2.0

    # Project type match: +2 points
    if project_type:
        pt_col = f'proj_{project_type.lower().replace(" ", "_")}'
        if pt_col in df.columns:
            df.loc[df[pt_col] == 1, '_relevance'] += 2.0

    # Recency bonus: +1 for newer cases
    if 'year_recency' in df.columns:
        max_rec = df['year_recency'].max()
        if max_rec > 0:
            df['_relevance'] += df['year_recency'].fillna(0) / max_rec

    # Filter to at least somewhat relevant cases
    relevant = df[df['_relevance'] > 0].sort_values('_relevance', ascending=False)
    if len(relevant) < 3:
        relevant = df.sort_values('_relevance', ascending=False)

    total = len(relevant[relevant['_relevance'] >= 2])
    approval_rate = float((relevant.head(max(total, 50))['decision_clean'] == 'APPROVED').mean()) if len(relevant) > 0 else 0.0

    # Return top scored cases
    sample = relevant.head(limit)
    cases = []
    for _, row in sample.iterrows():
        cases.append({
            "case_number": str(row.get('case_number') or ''),
            "address": str(row.get('address_clean') or 'Unknown'),
            "decision": str(row.get('decision_clean') or ''),
            "ward": str(row.get('ward') or ''),
            "date": str(row.get('hearing_date') or row.get('filing_date') or '')[:10],
            "relevance_score": round(float(row.get('_relevance', 0)), 1),
        })

    return cases, total, approval_rate


def _estimate_timeline(ward=None):
    """Estimate decision timeline in days based on historical data for this ward."""
    if zba_df is None or 'filing_date' not in zba_df.columns or 'source_pdf' not in zba_df.columns:
        return None

    df = zba_df[zba_df['decision_clean'].notna()].copy()
    df['_filing_dt'] = pd.to_datetime(df['filing_date'], errors='coerce')
    df['_decision_dt'] = pd.to_datetime(
        df['source_pdf'].str.extract(r'Filed (.+?)\.pdf')[0], errors='coerce'
    )
    has_both = df[df['_filing_dt'].notna() & df['_decision_dt'].notna()].copy()
    has_both['_days'] = (has_both['_decision_dt'] - has_both['_filing_dt']).dt.days
    has_both = has_both[(has_both['_days'] > 0) & (has_both['_days'] < 1100)]

    if len(has_both) < 10:
        return None

    # Try ward-specific first
    if ward:
        try:
            ward_float = float(ward)
            ward_subset = has_both[has_both['ward'] == ward_float]
            if len(ward_subset) >= 10:
                return {
                    "median_days": int(ward_subset['_days'].median()),
                    "ward_specific": True,
                    "cases_used": int(len(ward_subset)),
                }
        except (ValueError, TypeError):
            pass

    # Fall back to overall
    return {
        "median_days": int(has_both['_days'].median()),
        "ward_specific": False,
        "cases_used": int(len(has_both)),
    }


@app.post("/analyze_proposal", tags=["Prediction"], dependencies=[Depends(verify_api_key)])
def analyze_proposal(payload: dict):
    """
    Analyze a development proposal and predict ZBA approval likelihood.

    Input:
    {
        "parcel_id": "0100001000",
        "proposed_use": "residential",
        "variances": ["height", "far", "parking"],
        "project_type": "addition",
        "ward": "17",
        "has_attorney": true
    }
    """
    if gdf is None:
        raise HTTPException(status_code=500, detail="GeoJSON not loaded")

    try:
        # Validate input with Pydantic (accepts raw dict too for backward compat)
        validated = ProposalInput(**payload)
    except Exception:
        # Fall through to manual extraction if Pydantic fails
        validated = None

    if validated:
        parcel_id = validated.parcel_id
        proposed_use = validated.proposed_use
        variances = validated.variances
        project_type = validated.project_type
        ward = validated.ward
        has_attorney = validated.has_attorney
        proposed_units = validated.proposed_units
        proposed_stories = validated.proposed_stories
    else:
        parcel_id = payload.get("parcel_id", "")
        proposed_use = payload.get("proposed_use", payload.get("use_type", payload.get("use", "")))
        variances = payload.get("variances", [])
        project_type = payload.get("project_type", None)
        ward = payload.get("ward", None)
        has_attorney = payload.get("has_attorney", False)
        proposed_units = safe_int(payload.get("proposed_units", 0))
        proposed_stories = safe_int(payload.get("proposed_stories", 0))

    # Lookup parcel
    parcel_row = None
    if parcel_id:
        match = gdf.loc[[str(parcel_id)]] if str(parcel_id) in gdf.index else gdf.iloc[0:0]
        if not match.empty:
            parcel_row = match.iloc[0]

    if parcel_row is None and parcel_id:
        # Don't error — still run prediction with what we have
        pass

    zoning = str(parcel_row.get("primary_zoning") or "") if parcel_row is not None else ""
    district = str(parcel_row.get("districts") or "") if parcel_row is not None else ""

    # Auto-detect ward from district if not provided
    if not ward and district and zba_df is not None and 'zoning_district' in zba_df.columns:
        ward_lookup = zba_df[
            (zba_df['zoning_district'] == district) & zba_df['ward'].notna()
        ]['ward']
        if not ward_lookup.empty:
            ward = str(int(ward_lookup.mode().iloc[0]))
            logger.info("Auto-detected ward %s from district %s", ward, district)

    # Build features
    features = build_features(parcel_row, proposed_use, variances, project_type, ward, has_attorney,
                              proposed_units, proposed_stories)

    # Get similar cases
    similar, total_similar, approval_rate_similar = get_similar_cases(ward, variances, project_type)

    # =========================
    # ML MODEL PREDICTION
    # =========================
    if model_package and 'model' in model_package:
        try:
            model = model_package['model']
            feature_cols = model_package['feature_cols']

            input_df = pd.DataFrame([features])

            # Ensure columns match training order
            for col in feature_cols:
                if col not in input_df.columns:
                    input_df[col] = 0
            input_df = input_df[feature_cols]

            prob = float(model.predict_proba(input_df)[0][1])

            # SHAP-based per-prediction explainability
            top_drivers = []
            shap_values_list = []
            try:
                import shap
                # Use base (uncalibrated) model for SHAP — TreeExplainer needs raw model
                shap_model = model_package.get('base_model', model)
                explainer = shap.TreeExplainer(shap_model)
                sv = explainer.shap_values(input_df)
                # For binary classification, sv may be [neg_class, pos_class] or just array
                if isinstance(sv, list):
                    sv = sv[1]  # positive class (approved)
                shap_row = sv[0]
                base_value = float(explainer.expected_value[1] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value)

                contributions = []
                for col, shap_val in zip(feature_cols, shap_row):
                    input_val = float(input_df[col].iloc[0])
                    contributions.append({
                        "feature": col.replace("_", " ").replace("var ", "").replace("proj ", "").title(),
                        "shap_value": round(float(shap_val), 4),
                        "input_value": round(input_val, 3) if abs(input_val) > 1 else int(input_val),
                        "direction": "increases" if shap_val > 0 else "decreases",
                    })
                contributions.sort(key=lambda x: -abs(x["shap_value"]))
                top_drivers = contributions[:10]
                shap_values_list = [{"feature": c["feature"], "value": c["shap_value"]} for c in contributions if abs(c["shap_value"]) > 0.001]
            except Exception as shap_err:
                logger.warning("SHAP computation failed, falling back to importance: %s", shap_err)
                # Fallback to global feature importance
                if hasattr(model, 'feature_importances_'):
                    importances = model.feature_importances_
                    input_vals = input_df.iloc[0]
                    contributions = []
                    for col, imp in zip(feature_cols, importances):
                        val = float(input_vals[col])
                        if imp > 0.01 and val != 0:
                            contributions.append({
                                "feature": col.replace("_", " ").replace("var ", "").replace("proj ", "").title(),
                                "shap_value": round(float(imp), 4),
                                "input_value": round(val, 3) if abs(val) > 1 else int(val),
                                "direction": "unknown",
                            })
                    contributions.sort(key=lambda x: -abs(x["shap_value"]))
                    top_drivers = contributions[:10]

            # Confidence interval estimate based on similar case count
            # Wider interval when we have fewer comparable cases
            if total_similar > 100:
                confidence = "high"
                margin = 0.05
            elif total_similar > 30:
                confidence = "medium"
                margin = 0.10
            else:
                confidence = "low"
                margin = 0.18
            prob_low = max(0.0, prob - margin)
            prob_high = min(1.0, prob + margin)

            # Key factors analysis
            key_factors = []
            if has_attorney:
                key_factors.append("Legal representation increases approval likelihood by ~18%")
            else:
                key_factors.append("No attorney — legal representation significantly increases approval odds")
            if len(variances) > 3:
                key_factors.append(f"Requesting {len(variances)} variances reduces approval odds — consider reducing scope")
            elif len(variances) == 1:
                key_factors.append("Single variance request — simpler cases have higher approval rates")
            if any('height' in v.lower() for v in variances):
                key_factors.append("Height variances face additional scrutiny from the Board")
            if any('parking' in v.lower() for v in variances):
                key_factors.append("Parking variances are common — 68% of cases include parking relief")
            if any('conditional' in v.lower() for v in variances):
                key_factors.append("Conditional use permits have different review criteria than variances")
            if project_type and 'multi_family' in (project_type or '').lower():
                key_factors.append("Multi-family projects have a 41% historical approval rate")
            elif project_type and 'addition' in (project_type or '').lower():
                key_factors.append("Additions/extensions have higher approval rates than new construction")
            elif project_type and 'new_construction' in (project_type or '').lower():
                key_factors.append("New construction faces more scrutiny — consider community engagement")
            if proposed_units > 5:
                key_factors.append(f"Large project ({proposed_units} units) — expect community opposition and Board questions")
            if approval_rate_similar > 0 and total_similar > 10:
                key_factors.append(f"Similar cases in this area have {approval_rate_similar:.0%} approval rate ({total_similar} cases)")
            elif total_similar <= 10 and total_similar > 0:
                key_factors.append(f"Limited historical data ({total_similar} similar cases) — prediction confidence is lower")

            return {
                "parcel_id": parcel_id,
                "zoning": zoning,
                "district": district,
                "proposed_use": proposed_use,
                "project_type": project_type or "not specified",
                "variances": variances,
                "has_attorney": has_attorney,
                "approval_probability": round(prob, 3),
                "probability_range": [round(prob_low, 3), round(prob_high, 3)],
                "confidence": confidence,
                "based_on_cases": total_similar,
                "ward_approval_rate": round(approval_rate_similar, 3) if total_similar > 0 else None,
                "key_factors": key_factors,
                "top_drivers": top_drivers,
                "similar_cases": similar,
                "estimated_timeline_days": _estimate_timeline(ward),
                "model": model_package.get('model_name', 'ml_model'),
                "model_auc": model_package.get('auc_score', 0),
                "total_training_cases": model_package.get('total_cases', 0),
                "disclaimer": "IMPORTANT: This is a statistical risk assessment based on historical ZBA decisions, not a prediction or guarantee of outcome. Actual results depend on factors not captured in this model including board composition, public testimony, and site-specific conditions. This does not constitute legal, financial, or professional advice. Always consult a qualified zoning attorney before making financial commitments based on this analysis."
            }
        except Exception as e:
            logger.error("ML model error: %s, falling back to heuristic", e)

    # =========================
    # DATA-DRIVEN HEURISTIC FALLBACK
    # =========================
    # Uses actual dataset statistics when the ML model is unavailable
    df_decided = zba_df[zba_df['decision_clean'].notna()] if zba_df is not None else pd.DataFrame()
    base_rate = float(df_decided['decision_clean'].eq('APPROVED').mean()) if len(df_decided) > 0 else 0.65
    score = base_rate

    # Variance penalty — computed from actual data
    if len(df_decided) > 0 and 'variance_types' in df_decided.columns:
        for v in variances:
            v_lower = v.lower()
            v_mask = df_decided['variance_types'].fillna('').str.contains(v_lower, na=False)
            v_cases = df_decided[v_mask]
            if len(v_cases) > 10:
                v_rate = float(v_cases['decision_clean'].eq('APPROVED').mean())
                # Adjust score toward the variance-specific rate
                score += (v_rate - base_rate) * 0.15
    else:
        # Static fallback if no data
        if len(variances) > 3:
            score -= 0.15
        elif len(variances) > 2:
            score -= 0.08

    # Attorney effect — from data
    if 'has_attorney' in df_decided.columns and len(df_decided) > 0:
        atty_cases = df_decided[df_decided['has_attorney'] == 1]
        no_atty_cases = df_decided[df_decided['has_attorney'] == 0]
        if len(atty_cases) > 10 and len(no_atty_cases) > 10:
            atty_boost = float(atty_cases['decision_clean'].eq('APPROVED').mean()) - float(no_atty_cases['decision_clean'].eq('APPROVED').mean())
            if has_attorney:
                score += atty_boost * 0.3
            else:
                score -= atty_boost * 0.15
    else:
        if has_attorney:
            score += 0.10

    # Ward-specific adjustment
    if approval_rate_similar > 0 and total_similar > 10:
        score = (score * 0.6) + (approval_rate_similar * 0.4)

    score = max(0.05, min(0.95, score))

    key_factors_heuristic = [
        f"Using data-driven heuristic ({len(df_decided):,} cases) — ML model not loaded",
    ]
    if has_attorney:
        key_factors_heuristic.append("Legal representation factored in from historical attorney effect")
    if approval_rate_similar > 0:
        key_factors_heuristic.append(f"Ward-adjusted using {total_similar} similar cases ({approval_rate_similar:.0%} approval)")

    return {
        "parcel_id": parcel_id,
        "zoning": zoning,
        "district": district,
        "proposed_use": proposed_use,
        "project_type": project_type or "not specified",
        "variances": variances,
        "has_attorney": has_attorney,
        "approval_probability": round(score, 3),
        "probability_range": [round(max(0, score - 0.15), 3), round(min(1, score + 0.15), 3)],
        "confidence": "low",
        "based_on_cases": total_similar,
        "ward_approval_rate": round(approval_rate_similar, 3) if total_similar > 0 else None,
        "key_factors": key_factors_heuristic,
        "top_drivers": [],
        "similar_cases": similar,
        "model": "data_driven_heuristic",
        "disclaimer": "IMPORTANT: This is a statistical risk assessment based on historical ZBA decisions, not a prediction or guarantee of outcome. Actual results depend on factors not captured in this model including board composition, public testimony, and site-specific conditions. This does not constitute legal, financial, or professional advice. Always consult a qualified zoning attorney before making financial commitments based on this analysis."
    }


# =========================
# BATCH PREDICTION
# =========================

@app.post("/batch_predict", tags=["Prediction"], dependencies=[Depends(verify_api_key)])
def batch_predict(payload: dict):
    """
    Predict approval likelihood for multiple proposals at once. Max 20 per request.

    Input: {"proposals": [<proposal1>, <proposal2>, ...]}
    Each proposal has the same schema as /analyze_proposal.
    Returns: {"results": [<result1>, <result2>, ...]}
    """
    proposals = payload.get("proposals", [])
    if not proposals:
        raise HTTPException(status_code=400, detail="No proposals provided")
    if len(proposals) > 20:
        raise HTTPException(status_code=400, detail="Maximum 20 proposals per batch request")

    results = []
    for i, p in enumerate(proposals):
        try:
            result = analyze_proposal(p)
            result["_index"] = i
            results.append(result)
        except Exception as e:
            results.append({"_index": i, "error": str(e)})

    return {"results": results, "total": len(results)}


# =========================
# COMPARE / WHAT-IF ENDPOINT
# =========================

@app.post("/compare", tags=["Prediction"], dependencies=[Depends(verify_api_key)])
def compare_scenarios(payload: dict):
    """
    Run the prediction under multiple scenarios and return the differences.
    Compares: with/without attorney, and with fewer variances.
    Returns actual model-computed probability differences.
    """
    if gdf is None:
        raise HTTPException(status_code=500, detail="GeoJSON not loaded")

    try:
        parcel_id = payload.get("parcel_id", "")
        proposed_use = payload.get("proposed_use", payload.get("use_type", ""))
        variances = payload.get("variances", [])
        project_type = payload.get("project_type", None)
        ward = payload.get("ward", None)
        has_attorney = payload.get("has_attorney", False)
        proposed_units = safe_int(payload.get("proposed_units", 0))
        proposed_stories = safe_int(payload.get("proposed_stories", 0))

        # Lookup parcel
        parcel_row = None
        if parcel_id:
            match = gdf.loc[[str(parcel_id)]] if str(parcel_id) in gdf.index else gdf.iloc[0:0]
            if not match.empty:
                parcel_row = match.iloc[0]

        district = str(parcel_row.get("districts") or "") if parcel_row is not None else ""

        # Auto-detect ward from district if not provided
        if not ward and district and zba_df is not None and 'zoning_district' in zba_df.columns:
            ward_lookup = zba_df[
                (zba_df['zoning_district'] == district) & zba_df['ward'].notna()
            ]['ward']
            if not ward_lookup.empty:
                ward = str(int(ward_lookup.mode().iloc[0]))

        def predict_prob(feat_dict):
            """Run a single prediction given a feature dict."""
            if model_package and 'model' in model_package:
                model = model_package['model']
                feature_cols = model_package['feature_cols']
                input_df = pd.DataFrame([feat_dict])
                for col in feature_cols:
                    if col not in input_df.columns:
                        input_df[col] = 0
                input_df = input_df[feature_cols]
                return float(model.predict_proba(input_df)[0][1])
            return None

        # --- Base scenario ---
        base_features = build_features(parcel_row, proposed_use, variances, project_type,
                                       ward, has_attorney, proposed_units, proposed_stories)
        base_prob = predict_prob(base_features)

        if base_prob is None:
            raise HTTPException(status_code=503, detail="No ML model loaded — cannot compare scenarios")

        scenarios = []

        # --- Attorney toggle ---
        alt_features = build_features(parcel_row, proposed_use, variances, project_type,
                                      ward, not has_attorney, proposed_units, proposed_stories)
        alt_prob = predict_prob(alt_features)
        diff = alt_prob - base_prob
        if has_attorney:
            scenarios.append({
                "scenario": "Without attorney",
                "probability": round(alt_prob, 3),
                "difference": round(diff, 3),
                "description": f"Removing attorney representation would change probability by {diff:+.1%}"
            })
        else:
            scenarios.append({
                "scenario": "With attorney",
                "probability": round(alt_prob, 3),
                "difference": round(diff, 3),
                "description": f"Adding attorney representation would change probability by {diff:+.1%}"
            })

        # --- Fewer variances (remove one at a time) ---
        if len(variances) > 1:
            fewer = variances[:max(1, len(variances) - 1)]
            fewer_features = build_features(parcel_row, proposed_use, fewer, project_type,
                                            ward, has_attorney, proposed_units, proposed_stories)
            fewer_prob = predict_prob(fewer_features)
            diff = fewer_prob - base_prob
            scenarios.append({
                "scenario": f"With {len(fewer)} variance(s) instead of {len(variances)}",
                "probability": round(fewer_prob, 3),
                "difference": round(diff, 3),
                "description": f"Reducing to {len(fewer)} variance(s) would change probability by {diff:+.1%}"
            })

        # --- Single variance only (if requesting 3+) ---
        if len(variances) >= 3:
            single = variances[:1]
            single_features = build_features(parcel_row, proposed_use, single, project_type,
                                             ward, has_attorney, proposed_units, proposed_stories)
            single_prob = predict_prob(single_features)
            diff = single_prob - base_prob
            scenarios.append({
                "scenario": f"With only 1 variance ({single[0]})",
                "probability": round(single_prob, 3),
                "difference": round(diff, 3),
                "description": f"Requesting only 1 variance would change probability by {diff:+.1%}"
            })

        # --- Different project type: addition (usually higher approval) ---
        if project_type and 'addition' not in (project_type or '').lower():
            add_features = build_features(parcel_row, proposed_use, variances, 'addition',
                                          ward, has_attorney, proposed_units, proposed_stories)
            add_prob = predict_prob(add_features)
            diff = add_prob - base_prob
            if abs(diff) > 0.01:
                scenarios.append({
                    "scenario": "As an Addition/Extension project",
                    "probability": round(add_prob, 3),
                    "difference": round(diff, 3),
                    "description": f"If framed as an addition/extension: probability changes by {diff:+.1%}"
                })

        # --- Best-case scenario: attorney + fewer variances ---
        best_attorney = True
        best_variances = variances[:1] if len(variances) > 1 else variances
        best_features = build_features(parcel_row, proposed_use, best_variances, project_type,
                                       ward, best_attorney, proposed_units, proposed_stories)
        best_prob = predict_prob(best_features)
        diff = best_prob - base_prob
        if abs(diff) > 0.02:
            scenarios.append({
                "scenario": "Best case (attorney + minimal variances)",
                "probability": round(best_prob, 3),
                "difference": round(diff, 3),
                "description": f"With attorney and {len(best_variances)} variance(s): {best_prob:.0%} ({diff:+.1%})"
            })

        return {
            "base_probability": round(base_prob, 3),
            "has_attorney": has_attorney,
            "num_variances": len(variances),
            "scenarios": scenarios
        }

    except Exception as e:
        logger.error("Compare endpoint error: %s", traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Comparison failed: {str(e)}")


# =========================
# MARKET INTELLIGENCE — see routes/market_intel.py
# Router is included at import time, data is injected in the startup event.
try:
    from routes.market_intel import router as market_router, init as market_init
    app.include_router(market_router)
    logger.info("Market intel router included")
except Exception as e:
    market_init = None
    logger.warning(f"Market intel router not loaded: {e}")


# PLATFORM STATS
# =========================

@app.get("/stats", tags=["Platform"])
def overall_stats():
    """Overall platform statistics for the dashboard."""
    if zba_df is None:
        raise HTTPException(status_code=500, detail="ZBA data not loaded")

    df = zba_df[zba_df['decision_clean'].notna()].copy()
    total = len(df)
    approved = int((df['decision_clean'] == 'APPROVED').sum())
    denied = total - approved

    # Ward stats
    df['_ward'] = df['ward'].apply(lambda w: str(int(float(w))) if pd.notna(w) else None)
    ward_rates = df[df['_ward'].notna()].groupby('_ward')['decision_clean'].apply(
        lambda x: (x == 'APPROVED').mean()
    )

    return {
        "total_cases": total,
        "total_approved": approved,
        "total_denied": denied,
        "overall_approval_rate": round(approved / total, 3) if total > 0 else 0,
        "total_wards": int(ward_rates.count()),
        "best_ward": str(ward_rates.idxmax()) if len(ward_rates) > 0 else None,
        "best_ward_rate": round(float(ward_rates.max()), 3) if len(ward_rates) > 0 else None,
        "worst_ward": str(ward_rates.idxmin()) if len(ward_rates) > 0 else None,
        "worst_ward_rate": round(float(ward_rates.min()), 3) if len(ward_rates) > 0 else None,
        "total_parcels": len(gdf) if gdf is not None else 0,
    }


@app.get("/autocomplete", tags=["Search"])
def autocomplete(q: str = "", limit: int = 10):
    """Address autocomplete from 175K property records."""
    if not q or len(q) < 3:
        return {"suggestions": []}
    if parcel_addr_df is None:
        return {"suggestions": []}

    q_norm = normalize_address(q)
    matches = parcel_addr_df[parcel_addr_df['_addr_norm'].str.contains(q_norm, na=False)].head(limit)

    return {
        "suggestions": [
            {"address": row['address'], "parcel_id": row['parcel_id']}
            for _, row in matches.iterrows()
        ]
    }


@app.get("/model_info", tags=["Platform"])
def model_info():
    """Full model metadata and version info."""
    if model_package is None:
        raise HTTPException(status_code=503, detail="No model loaded")

    return {
        "model_name": model_package.get('model_name'),
        "model_version": model_package.get('model_version'),
        "auc_score": model_package.get('auc_score'),
        "brier_score": model_package.get('brier_score'),
        "optimal_threshold": model_package.get('optimal_threshold'),
        "cv_auc_mean": model_package.get('cv_auc_mean'),
        "cv_auc_std": model_package.get('cv_auc_std'),
        "total_cases": model_package.get('total_cases'),
        "train_size": model_package.get('train_size'),
        "test_size": model_package.get('test_size'),
        "feature_count": len(model_package.get('feature_cols', [])),
        "feature_cols": model_package.get('feature_cols', []),
        "is_calibrated": model_package.get('is_calibrated', False),
        "leakage_free": model_package.get('leakage_free', False),
        "removed_features": model_package.get('removed_features', []),
        "trained_at": model_package.get('trained_at'),
        "dataset_hash": model_package.get('dataset_hash'),
    }


# =========================
# HEALTH CHECK
# =========================

@app.get("/health", tags=["Platform"])
def health():
    return {
        "status": "ok",
        "geojson_loaded": gdf is not None,
        "zba_loaded": zba_df is not None,
        "model_loaded": model_package is not None,
        "geocoder_loaded": parcel_addr_df is not None,
        "total_parcels": len(gdf) if gdf is not None else 0,
        "total_cases": len(zba_df) if zba_df is not None else 0,
        "geocoder_parcels": len(parcel_addr_df) if parcel_addr_df is not None else 0,
        "model_name": model_package.get('model_name') if model_package else None,
        "model_auc": model_package.get('auc_score') if model_package else None,
        "model_brier": model_package.get('brier_score') if model_package else None,
        "optimal_threshold": model_package.get('optimal_threshold') if model_package else None,
        "features": len(model_package.get('feature_cols', [])) if model_package else 0,
        "postgis_available": db_available(),
        "leakage_free": model_package.get('leakage_free', False) if model_package else False,
        "model_version": model_package.get('model_version') if model_package else None,
    }


@app.get("/data_status", tags=["Platform"])
def data_status():
    """Data freshness and pipeline status — check if OCR is running, when data was last updated."""
    import datetime

    status = {}

    # Check data file timestamps
    for name, path in [
        ("zba_cases_cleaned", ZBA_DATA_PATH),
        ("zba_model", MODEL_PATH),
        ("geojson", GEOJSON_PATH),
        ("property_assessment", PROPERTY_PATH),
    ]:
        full_path = os.path.abspath(path)
        if os.path.exists(full_path):
            mtime = os.path.getmtime(full_path)
            dt = datetime.datetime.fromtimestamp(mtime)
            age_hours = (time.time() - mtime) / 3600
            status[name] = {
                "path": full_path,
                "last_modified": dt.isoformat(),
                "age_hours": round(age_hours, 1),
                "size_mb": round(os.path.getsize(full_path) / (1024 * 1024), 1),
            }
        else:
            status[name] = {"path": full_path, "exists": False}

    # Check if OCR is running (checkpoint file exists)
    checkpoint_path = os.path.join(BASE_DIR, "../ocr_checkpoint.json")
    if os.path.exists(checkpoint_path):
        try:
            import json
            with open(checkpoint_path) as f:
                cp = json.load(f)
            completed = len(cp.get("completed", []))
            status["ocr_pipeline"] = {
                "running": True,
                "completed_pdfs": completed,
                "checkpoint_file": checkpoint_path,
            }
        except Exception:
            status["ocr_pipeline"] = {"running": True, "error": "Could not read checkpoint"}
    else:
        status["ocr_pipeline"] = {"running": False}

    return status


# =========================
# SITE SELECTION / RECOMMENDATIONS
# =========================

@app.get("/recommend", tags=["Recommendation"])
def recommend_parcels(
    use_type: str = "residential",
    project_type: str = "new_construction",
    variances: str = "",
    min_approval_rate: float = 0.5,
    ward: str = None,
    limit: int = 20,
):
    """
    Recommend the best parcels in Boston for a given project type.

    Instead of: "What are my odds on this parcel?"
    This answers: "Which parcels should I look at for a 10-unit residential project?"

    Returns parcels ranked by predicted approval probability, filtered by criteria.
    """
    if model_package is None or gdf is None:
        raise HTTPException(status_code=503, detail="Model or parcel data not loaded")

    model = model_package.get('model')
    feature_cols = model_package.get('feature_cols', [])
    if model is None or not feature_cols:
        raise HTTPException(status_code=503, detail="Model not available")

    variance_list = [v.strip() for v in variances.split(",") if v.strip()] if variances else []

    # Get candidate parcels — optionally filter by ward
    candidates = gdf.copy()
    if ward:
        # Filter parcels by ward if we can match
        if zba_df is not None and 'ward' in zba_df.columns:
            ward_parcels = zba_df[zba_df['ward'] == float(ward)]['pa_parcel_id'].dropna().unique()
            ward_parcel_strs = set(str(int(p)).zfill(10) for p in ward_parcels if not np.isnan(p))
            candidates = candidates[candidates.index.isin(ward_parcel_strs)]

    # Sample if too many candidates (for speed)
    if len(candidates) > 2000:
        candidates = candidates.sample(2000, random_state=42)

    results = []
    for parcel_id in candidates.index:
        try:
            parcel_row = candidates.loc[parcel_id]
            if isinstance(parcel_row, pd.DataFrame):
                parcel_row = parcel_row.iloc[0]

            features = build_features(
                parcel_row=parcel_row,
                proposed_use=use_type,
                variances=variance_list,
                has_attorney=True,  # Assume best case
                ward=ward or str(parcel_row.get('ward', '')),
                proposed_units=0,
                proposed_stories=0,
                project_type=project_type,
            )

            feature_vector = pd.DataFrame([features])[feature_cols].fillna(0)
            prob = float(model.predict_proba(feature_vector)[:, 1][0])

            if prob >= min_approval_rate:
                result = {
                    "parcel_id": str(parcel_id),
                    "approval_probability": round(prob, 3),
                    "zoning_code": str(parcel_row.get("primary_zoning") or ""),
                    "district": str(parcel_row.get("districts") or ""),
                }

                # Add centroid for map
                if hasattr(parcel_row, 'geometry') and parcel_row.geometry:
                    centroid = parcel_row.geometry.centroid
                    result["lat"] = round(centroid.y, 6)
                    result["lon"] = round(centroid.x, 6)

                results.append(result)
        except Exception:
            continue

    results.sort(key=lambda x: -x['approval_probability'])
    results = results[:limit]

    return {
        "query": {
            "use_type": use_type,
            "project_type": project_type,
            "variances": variance_list,
            "ward": ward,
            "min_approval_rate": min_approval_rate,
        },
        "total_candidates": len(candidates),
        "results_found": len(results),
        "parcels": results,
        "disclaimer": "Probabilities are model estimates based on historical ZBA decisions. "
                      "Actual outcomes depend on many factors not captured in the model. "
                      "This is a risk assessment tool, not legal advice.",
    }


# =========================
# GLOBAL ERROR HANDLER
# =========================

from fastapi.responses import JSONResponse
from starlette.requests import Request

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Catch unhandled exceptions and return a clean JSON error."""
    logger.error("Unhandled error on %s: %s", request.url, traceback.format_exc())
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc),
            "path": str(request.url.path)
        }
    )
