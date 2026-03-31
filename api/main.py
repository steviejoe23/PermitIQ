"""
PermitIQ API v2 — Full Feature Set
FastAPI backend for Boston zoning intelligence
"""

from fastapi import FastAPI, HTTPException, Depends, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field
from typing import List, Optional
from services.zoning_code import get_zoning_requirements, check_compliance, ZONING_REQUIREMENTS
import geopandas as gpd
import pandas as pd
import numpy as np
import joblib
import re
import os
import logging
import traceback
from functools import lru_cache
# model_classes must be importable as top-level module for pickle deserialization
import sys
_services_dir = os.path.join(os.path.dirname(__file__), 'services')
if _services_dir not in sys.path:
    sys.path.insert(0, _services_dir)
try:
    from model_classes import StackingEnsemble, ManualCalibratedModel
except ImportError:
    StackingEnsemble = ManualCalibratedModel = None
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
TRACKER_PATH = os.path.join(BASE_DIR, "../zba_tracker.csv")

gdf = None
zba_df = None
model_package = None
parcel_addr_df = None  # address→parcel lookup
timeline_stats = None  # Pre-computed timeline stats from ZBA tracker


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


def _format_date(val) -> str:
    """Format a date value to a clean string. Handles NaN, None, various formats."""
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return ''
    s = str(val).strip()
    if s.lower() in ('nan', 'none', 'nat', ''):
        return ''
    try:
        dt = pd.to_datetime(s)
        return dt.strftime('%b %d, %Y')  # e.g. "Jul 21, 2023"
    except Exception:
        return s  # return raw string, don't truncate


def _precompute_timeline_stats(tracker_path: str) -> dict:
    """Pre-compute timeline statistics from ZBA tracker CSV.

    Returns a dict with:
      - overall: {filing_to_hearing, filing_to_decision, hearing_to_decision, filing_to_closed}
      - by_ward: {ward_str: same structure}
      - by_appeal_type: {appeal_type: same structure}
    Each phase dict has: median_days, p25_days, p75_days, cases_used
    """
    try:
        tk = pd.read_csv(tracker_path, low_memory=False)
    except Exception as e:
        logger.error("Failed to load tracker CSV for timeline stats: %s", e)
        return None

    for col in ['submitted_date', 'hearing_date', 'final_decision_date', 'closed_date']:
        tk[col] = pd.to_datetime(tk[col], errors='coerce')

    # Compute phase durations in days
    tk['filing_to_hearing'] = (tk['hearing_date'] - tk['submitted_date']).dt.days
    tk['filing_to_decision'] = (tk['final_decision_date'] - tk['submitted_date']).dt.days
    tk['hearing_to_decision'] = (tk['final_decision_date'] - tk['hearing_date']).dt.days
    tk['filing_to_closed'] = (tk['closed_date'] - tk['submitted_date']).dt.days

    phases = ['filing_to_hearing', 'filing_to_decision', 'hearing_to_decision', 'filing_to_closed']

    def _phase_stats(df_subset, phase_col):
        """Compute percentile stats for a single phase from a DataFrame subset."""
        valid = df_subset[(df_subset[phase_col] >= 0) & (df_subset[phase_col] < 730)][phase_col].dropna()
        if len(valid) < 5:
            return None
        return {
            "median_days": int(valid.median()),
            "p25_days": int(valid.quantile(0.25)),
            "p75_days": int(valid.quantile(0.75)),
            "cases_used": int(len(valid)),
        }

    def _all_phases(df_subset):
        """Compute stats for all phases from a DataFrame subset."""
        result = {}
        for phase in phases:
            stats = _phase_stats(df_subset, phase)
            if stats:
                result[phase] = stats
        return result

    stats = {"overall": _all_phases(tk), "by_ward": {}, "by_appeal_type": {}}

    # By ward
    if 'ward' in tk.columns:
        for ward_val, group in tk.groupby('ward'):
            ward_str = str(int(ward_val)) if pd.notna(ward_val) else None
            if ward_str and len(group) >= 10:
                ward_stats = _all_phases(group)
                if ward_stats:
                    stats["by_ward"][ward_str] = ward_stats

    # By appeal type (Zoning vs Building)
    if 'appeal_type' in tk.columns:
        for atype, group in tk.groupby('appeal_type'):
            if pd.notna(atype) and len(group) >= 10:
                atype_stats = _all_phases(group)
                if atype_stats:
                    stats["by_appeal_type"][str(atype)] = atype_stats

    total_cases = sum(
        stats["overall"].get(p, {}).get("cases_used", 0) for p in phases
    )
    logger.info("Timeline stats pre-computed from tracker (%d ward groups, %d total phase-observations)",
                len(stats["by_ward"]), total_cases)
    return stats


@app.on_event("startup")
def load_data():
    global gdf, zba_df, model_package, parcel_addr_df, timeline_stats

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
        _pa = pd.read_csv(PROPERTY_PATH, usecols=['PID', 'ST_NUM', 'ST_NAME', 'LAND_SF'], low_memory=False)
        _pa = _pa.dropna(subset=['PID', 'ST_NUM', 'ST_NAME'])
        # Clean street numbers (remove .0 from float conversion)
        _pa['ST_NUM'] = _pa['ST_NUM'].astype(str).str.strip().str.replace(r'\.0$', '', regex=True)
        _pa['ST_NAME'] = _pa['ST_NAME'].astype(str).str.strip()
        _pa['address'] = _pa['ST_NUM'] + ' ' + _pa['ST_NAME']
        _pa['_addr_norm'] = _pa['address'].apply(normalize_address)
        # Pad PID to 10 digits to match GeoJSON parcel_id
        _pa['parcel_id'] = _pa['PID'].astype(str).str.zfill(10)
        # Clean LAND_SF (remove commas, convert to float)
        _pa['lot_size'] = pd.to_numeric(_pa['LAND_SF'].astype(str).str.replace(',', ''), errors='coerce').fillna(0)
        parcel_addr_df = _pa[['parcel_id', 'address', '_addr_norm', 'lot_size']].drop_duplicates('parcel_id')
        logger.info("Property address lookup loaded (%d parcels)", len(parcel_addr_df))
    except Exception as e:
        logger.warning("Could not load property assessment for geocoding: %s", e)

    # Pre-compute timeline stats from ZBA tracker
    timeline_stats = _precompute_timeline_stats(TRACKER_PATH)

    # Initialize market intel router with loaded data
    if market_init is not None and zba_df is not None:
        market_init(zba_df, VARIANCE_TYPES, PROJECT_TYPES, timeline_stats=timeline_stats)
        logger.info("Market intel router initialized with %d cases", len(zba_df))

    # Build geographic index for nearby case search
    _build_case_coords()


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


def _haversine_m(lat1, lon1, lat2, lon2):
    """Haversine distance in meters between two (lat, lon) points."""
    R = 6371000  # Earth radius in meters
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlam = np.radians(lon2 - lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlam/2)**2
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


# Pre-computed case coordinates (built at startup after data loads)
_case_coords = None  # DataFrame with case_number, lat, lon, address, decision, ward, date


def _build_case_coords():
    """Match ZBA case addresses to parcel centroids for geographic search. Vectorized for speed."""
    global _case_coords
    if gdf is None or zba_df is None or parcel_addr_df is None:
        return

    logger.info("Building case coordinate index for geographic search...")

    # Vectorized centroid extraction from GeoJSON (fast — no row-by-row loop)
    centroids = gdf.geometry.centroid
    centroid_df = pd.DataFrame({
        'parcel_id': gdf.index.values,
        'lat': centroids.y.values,
        'lon': centroids.x.values,
    })

    # Build address → parcel_id lookup as a dict (fast)
    addr_to_parcel = dict(zip(parcel_addr_df['_addr_norm'], parcel_addr_df['parcel_id']))

    # Get ZBA cases with addresses and decisions
    zba_with_addr = zba_df[
        zba_df['address_clean'].notna() &
        zba_df['decision_clean'].notna() &
        zba_df['address_clean'].str.match(r'^\d', na=False)
    ].drop_duplicates('case_number').copy()

    if '_addr_norm' not in zba_with_addr.columns:
        zba_with_addr['_addr_norm'] = zba_with_addr['address_clean'].apply(normalize_address)

    # Direct match: join ZBA normalized addresses to parcel address lookup
    zba_with_addr['_matched_pid'] = zba_with_addr['_addr_norm'].map(addr_to_parcel)

    # Merge with centroids to get lat/lon
    merged = zba_with_addr.merge(
        centroid_df, left_on='_matched_pid', right_on='parcel_id', how='left'
    )

    # Keep only geocoded cases
    geocoded = merged[merged['lat'].notna()].copy()

    # Clean ward
    def _clean_ward(w):
        try:
            if pd.notna(w) and str(w).replace('.','',1).isdigit():
                return str(int(float(w)))
        except (ValueError, TypeError):
            pass
        return ''

    records = []
    for _, row in geocoded.iterrows():
        records.append({
            'case_number': str(row['case_number']),
            'address': str(row['address_clean']),
            'lat': float(row['lat']),
            'lon': float(row['lon']),
            'decision': str(row.get('decision_clean', '')),
            'ward': _clean_ward(row.get('ward', '')),
            'date': _clean_case_date(row),
            'applicant': safe_str(row.get('applicant_name', '')),
            'variances': safe_str(row.get('variance_types', '')),
        })

    _case_coords = pd.DataFrame(records)
    total_cases = len(zba_with_addr)
    logger.info("Case coordinate index built: %d of %d cases geocoded (%.0f%%)",
                len(_case_coords), total_cases, 100 * len(_case_coords) / max(total_cases, 1))


@app.get("/parcels/{parcel_id}/nearby_cases", tags=["Parcels"])
def nearby_cases(parcel_id: str, radius_m: int = 800, limit: int = 20, ward_only: bool = False):
    """
    Find ZBA cases near a parcel using real geographic distance.

    Parameters:
        parcel_id: Parcel to search around
        radius_m: Search radius in meters (default 800 = ~0.5 miles)
        limit: Max results (default 20)
        ward_only: If true, only return cases in the same ward
    """
    if gdf is None:
        raise HTTPException(status_code=500, detail="GeoJSON not loaded")
    if zba_df is None:
        raise HTTPException(status_code=500, detail="ZBA data not loaded")

    row = gdf.loc[[parcel_id]] if parcel_id in gdf.index else gdf.iloc[0:0]
    if row.empty:
        raise HTTPException(status_code=404, detail="Parcel not found")

    centroid = row.iloc[0].geometry.centroid
    parcel_lat, parcel_lon = centroid.y, centroid.x
    district = str(row.iloc[0].get("districts") or "")

    cases = []
    has_geo = False
    df = None

    # Use pre-computed geographic index if available
    if _case_coords is not None and len(_case_coords) > 0:
        # Vectorized haversine — fast
        df = _case_coords.copy()
        df['dist_m'] = _haversine_m(parcel_lat, parcel_lon, df['lat'].values, df['lon'].values)
        df = df[df['dist_m'] <= radius_m].sort_values('dist_m')
        has_geo = True

    # Detect ward from nearest cases (much more accurate than district-wide mode)
    parcel_ward = ''
    if has_geo and df is not None and not df.empty:
        nearby_wards = df[df['ward'] != ''].head(10)['ward']
        if not nearby_wards.empty:
            try:
                parcel_ward = str(nearby_wards.mode().iloc[0])
            except Exception:
                pass
    # Fallback: district-wide mode
    if not parcel_ward and zba_df is not None and district:
        z_col = 'zoning_district' if 'zoning_district' in zba_df.columns else None
        if z_col:
            ward_lookup = zba_df[(zba_df[z_col] == district) & zba_df['ward'].notna()]['ward']
            if not ward_lookup.empty:
                try:
                    parcel_ward = str(int(ward_lookup.mode().iloc[0]))
                except Exception:
                    pass

    # Geographic results: apply ward filter and build case list
    if has_geo and df is not None and not df.empty:
        if ward_only and parcel_ward:
            df = df[df['ward'] == parcel_ward]

        seen = set()
        for _, c in df.head(limit * 2).iterrows():
            cn = c['case_number']
            if cn in seen:
                continue
            addr = c['address']
            if len(addr) > 60 or not addr or addr in ('nan', 'None'):
                continue
            seen.add(cn)
            cases.append({
                "case_number": cn,
                "address": addr,
                "decision": c['decision'],
                "ward": c['ward'],
                "date": c['date'],
                "distance_m": int(c['dist_m']),
                "distance_ft": int(c['dist_m'] * 3.281),
                "applicant": c['applicant'] if c['applicant'] else None,
                "variances": c['variances'] if c['variances'] else None,
            })
            if len(cases) >= limit:
                break
    else:
        # Fallback: district-based matching (no coordinates available)
        z_col = 'zoning_district' if 'zoning_district' in zba_df.columns else None
        if z_col and district:
            nearby = zba_df[
                (zba_df[z_col] == district) &
                (zba_df['decision_clean'].notna())
            ].sort_values('case_number', ascending=False).head(limit)

            seen = set()
            for _, c in nearby.iterrows():
                cn = str(c.get('case_number', ''))
                if cn in seen:
                    continue
                addr = str(c.get('address_clean', ''))
                if len(addr) > 60 or not addr or addr in ('', 'nan', 'None'):
                    continue
                seen.add(cn)
                _w = c.get('ward', '')
                _ward_clean = str(int(float(_w))) if pd.notna(_w) and str(_w).replace('.','',1).isdigit() else safe_str(_w)
                cases.append({
                    "case_number": cn,
                    "address": addr,
                    "decision": str(c.get('decision_clean', '')),
                    "ward": _ward_clean,
                    "date": _clean_case_date(c),
                    "distance_m": None,
                    "distance_ft": None,
                })

    # Summary stats
    total_nearby = len(cases)
    approved_nearby = sum(1 for c in cases if c['decision'] == 'APPROVED')
    denied_nearby = sum(1 for c in cases if c['decision'] == 'DENIED')

    return {
        "parcel_id": parcel_id,
        "district": district,
        "ward": parcel_ward,
        "parcel_lat": parcel_lat,
        "parcel_lon": parcel_lon,
        "radius_m": radius_m,
        "radius_ft": int(radius_m * 3.281),
        "search_type": "geographic" if has_geo else "district",
        "cases": cases,
        "total": total_nearby,
        "approved": approved_nearby,
        "denied": denied_nearby,
        "approval_rate": round(approved_nearby / total_nearby, 3) if total_nearby > 0 else None,
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

    # Handle range addresses: "55 - 57 centre st" → try "57 centre st" and "55 centre st"
    range_match = re.match(r'^(\d+)\s*[-–]\s*(\d+)\s+(.*)', q_norm)
    if range_match:
        # Try the end number first (more likely to be the actual address), then start
        queries_to_try = [
            f"{range_match.group(2)} {range_match.group(3)}",
            f"{range_match.group(1)} {range_match.group(3)}",
            q_norm,  # also try the original
        ]
    else:
        queries_to_try = [q_norm]

    matches = pd.DataFrame()
    for q_try in queries_to_try:
        # Word-boundary matching: each word in query must appear as a whole word
        words = q_try.split()
        mask = pd.Series(True, index=parcel_addr_df.index)
        for word in words:
            if len(word) > 1:
                # Use word boundary for numbers to avoid "75" matching "175"
                if word.isdigit():
                    mask = mask & parcel_addr_df['_addr_norm'].str.contains(r'(?:^|\s)' + re.escape(word) + r'(?:\s|$)', na=False, regex=True)
                else:
                    mask = mask & parcel_addr_df['_addr_norm'].str.contains(word, na=False, regex=False)
        matches = parcel_addr_df[mask].head(10)
        if not matches.empty:
            break

    # Fallback: if no exact number match, find nearest address on the same street
    if matches.empty and queries_to_try:
        q_try = queries_to_try[0]
        num_match = re.match(r'^(\d+)\s+(.+)', q_try)
        if num_match:
            target_num = int(num_match.group(1))
            street_part = num_match.group(2)
            # Find all addresses on this street
            street_mask = parcel_addr_df['_addr_norm'].str.contains(street_part, na=False, regex=False)
            street_matches = parcel_addr_df[street_mask].copy()
            if not street_matches.empty:
                # Extract leading number and find closest
                street_matches['_num'] = street_matches['_addr_norm'].str.extract(r'^(\d+)', expand=False)
                street_matches['_num'] = pd.to_numeric(street_matches['_num'], errors='coerce')
                street_matches = street_matches.dropna(subset=['_num'])
                if not street_matches.empty:
                    street_matches['_dist'] = (street_matches['_num'] - target_num).abs()
                    matches = street_matches.sort_values('_dist').head(5)

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
    addr = re.sub(r'\bavenue\b', 'av', addr)
    addr = re.sub(r'\bave\b', 'av', addr)
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

    # Try exact substring match first — use word boundary on leading number to prevent
    # "75 tremont" from matching "1575 tremont"
    q_pattern = re.escape(q_norm)
    if re.match(r'^\d', q_norm):
        q_pattern = r'(?:^|\s|-)' + q_pattern
    mask = addr_df['_addr_norm'].str.contains(q_pattern, na=False, regex=True)

    # Also match range addresses: "57 centre st" should match "55 - 57 centre st"
    # Extract leading number from query
    q_num_match = re.match(r'^(\d+)\s+(.+)', q_norm)
    if q_num_match:
        q_num = q_num_match.group(1)
        q_street = q_num_match.group(2)
        # Match addresses like "55 - 57 <street>" or "55-57 <street>" where query number is in the range
        range_mask = addr_df['_addr_norm'].str.contains(
            r'\d+\s*[-–]\s*' + re.escape(q_num) + r'\s+' + re.escape(q_street),
            na=False, regex=True
        )
        mask = mask | range_mask

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
    if 'applicant_name' in matches.columns:
        agg_dict['applicant'] = ('applicant_name', lambda x: next((v for v in x if pd.notna(v) and str(v).strip()), ''))
    if zoning_col:
        agg_dict['zoning'] = (zoning_col, 'first')
    if date_col:
        agg_dict['latest_date'] = (date_col, 'last')  # 'max' fails on mixed-type date strings

    grouped = matches.groupby('_addr_norm').agg(**agg_dict)

    # Score results: exact start match > range match > partial match
    _q_num = q_num_match.group(1) if q_num_match else None
    def _relevance(addr_norm_key):
        if addr_norm_key.startswith(q_norm):
            return 0  # best: exact match from start
        if _q_num and re.search(r'\d+\s*[-–]\s*' + re.escape(_q_num) + r'\b', addr_norm_key):
            return 1  # good: range address match
        return 2  # ok: substring match
    grouped['_relevance'] = [_relevance(idx) for idx in grouped.index]
    grouped = grouped.sort_values(['_relevance', 'total_cases'], ascending=[True, False]).drop(columns=['_relevance']).head(10)

    results = []
    for _, row in grouped.iterrows():
        total = row['approved'] + row['denied']
        result_item = {
            "address": str(row['address']),
            "ward": str(safe_int(row['ward'])) if pd.notna(row['ward']) and safe_int(row['ward']) != 0 else "",
            "zoning": str(row.get('zoning', '')) if pd.notna(row.get('zoning', '')) else "",
            "total_cases": int(row['total_cases']),
            "approved": int(row['approved']),
            "denied": int(row['denied']),
            "approval_rate": round(row['approved'] / total, 2) if total > 0 else None,
            "latest_date": _format_date(row.get('latest_date', '')),
            "latest_case": str(row['latest_case']),
        }
        _applicant = safe_str(row.get('applicant', '')).strip()
        if _applicant and _applicant.lower() not in ('nan', 'none'):
            result_item["applicant"] = _applicant
        results.append(result_item)

    return results


# =========================
# ZONING ANALYSIS
# =========================

def _get_parcel_zoning(parcel_id: str) -> dict:
    """Single source of truth for parcel zoning data.

    Returns the REAL subdistrict requirements from BPDA official data,
    falling back to the lookup table only where subdistrict data is missing.
    Used by /zoning/{parcel_id}, /zoning/check_compliance, and /zoning/full_analysis.
    """
    district, article, all_codes, multi_zoning = '', '', '', False
    subdistrict = ''
    subdistrict_type = ''
    subdistrict_use = ''
    neighborhood = ''
    in_gcod = False
    in_coastal_flood = False

    # PostGIS first
    if db_available() and query_parcel is not None:
        db_row = query_parcel(parcel_id)
        if db_row is not None:
            district = db_row.get('primary_zoning', '')
            article = db_row.get('article', '')
            all_codes = db_row.get('all_zoning_codes', '')
            multi_zoning = db_row.get('multi_zoning', False)

    # GeoJSON — always read for subdistrict data even if PostGIS had district
    sub_max_far = None
    sub_max_height = None
    sub_max_floors = None
    sub_front_setback = None
    sub_side_setback = None
    sub_rear_setback = None

    if gdf is not None:
        matches = gdf[gdf['parcel_id'] == parcel_id]
        if not matches.empty:
            row = matches.iloc[0]
            if not district:
                district = str(row.get('primary_zoning', ''))
                article = str(row.get('article', ''))
                all_codes = str(row.get('all_zoning_codes', district))
                multi_zoning = bool(row.get('multi_zoning', False))
            subdistrict = str(row.get('zoning_subdistrict') or '')
            subdistrict_type = str(row.get('subdistrict_type') or '')
            subdistrict_use = str(row.get('subdistrict_use') or '')
            neighborhood = str(row.get('neighborhood_district') or row.get('districts') or '')
            in_gcod = bool(row.get('in_gcod', False))
            in_coastal_flood = bool(row.get('in_coastal_flood', False))
            sub_max_far = row.get('max_far')
            sub_max_height = row.get('max_height_ft')
            sub_max_floors = row.get('max_floors')
            sub_front_setback = row.get('front_setback_ft')
            sub_side_setback = row.get('side_setback_ft')
            sub_rear_setback = row.get('rear_setback_ft')
            sub_article = str(row.get('article_sub') or '')
            if sub_article:
                article = sub_article

    if not district:
        return None

    # Lookup table as fallback for fields not in subdistrict data
    reqs_fallback = get_zoning_requirements(district)

    def _pick(sub_val, fallback_key):
        if sub_val is not None and not (isinstance(sub_val, float) and np.isnan(sub_val)):
            return sub_val
        return reqs_fallback.get(fallback_key)

    # Build the single authoritative requirements dict
    requirements = {
        "max_far": _pick(sub_max_far, 'max_far'),
        "max_height_ft": _pick(sub_max_height, 'max_height_ft'),
        "max_stories": _pick(sub_max_floors, 'max_stories'),
        "min_lot_sf": reqs_fallback.get('min_lot_sf'),
        "min_frontage_ft": reqs_fallback.get('min_frontage_ft'),
        "min_front_yard_ft": _pick(sub_front_setback, 'min_front_yard_ft'),
        "min_side_yard_ft": _pick(sub_side_setback, 'min_side_yard_ft'),
        "min_rear_yard_ft": _pick(sub_rear_setback, 'min_rear_yard_ft'),
        "max_lot_coverage_pct": reqs_fallback.get('max_lot_coverage_pct'),
        "parking_per_unit": reqs_fallback.get('parking_per_unit'),
    }

    # Build overlay districts list
    overlay_districts = []
    if in_gcod:
        overlay_districts.append({
            "name": "Groundwater Conservation Overlay District (GCOD)",
            "code": "GCOD",
            "article": "32",
            "note": "This parcel is in the Groundwater Conservation Overlay District (GCOD). Additional permits and review may be required under Article 32.",
        })
    if in_coastal_flood:
        overlay_districts.append({
            "name": "Coastal Flood Resilience Overlay District",
            "code": "CFROD",
            "article": "25A",
            "note": "This parcel is in the Coastal Flood Resilience Overlay District. Flood-resistant design and elevated construction may be required.",
        })

    return {
        "parcel_id": parcel_id,
        "district": district,
        "article": article,
        "all_codes": all_codes,
        "multi_zoning": multi_zoning,
        "subdistrict": subdistrict,
        "subdistrict_type": subdistrict_type,
        "subdistrict_use": subdistrict_use,
        "neighborhood": neighborhood,
        "requirements": requirements,
        "allowed_uses": reqs_fallback.get('allowed_uses', []),
        "overlay_districts": overlay_districts,
        "data_source": "BPDA Zoning Subdistricts (official)" if subdistrict else "Lookup table (approximate)",
    }


@app.get("/zoning/districts", tags=["Zoning Analysis"])
def list_zoning_districts():
    """List all zoning districts with their dimensional requirements."""
    districts = []
    for code, reqs in ZONING_REQUIREMENTS.items():
        districts.append({
            "code": code,
            "name": reqs['district_name'],
            "article": reqs['article'],
            "description": reqs['description'],
            "max_far": reqs['max_far'],
            "max_height_ft": reqs['max_height_ft'],
            "max_stories": reqs.get('max_stories'),
            "allowed_uses": reqs['allowed_uses'],
        })
    return {"districts": districts, "total": len(districts)}


@app.get("/zoning/{parcel_id}", tags=["Zoning Analysis"])
def zoning_analysis(parcel_id: str):
    """
    Full zoning analysis for a parcel — answers:
    1. What zoning district is this parcel in?
    2. What article governs this property?
    3. What are the dimensional requirements?
    4. What's the zoning summary?
    """
    # Guard against static route names caught by path param
    if parcel_id in ('check_compliance', 'full_analysis', 'districts'):
        raise HTTPException(status_code=400, detail=f"Use POST /zoning/{parcel_id} instead")
    z = _get_parcel_zoning(parcel_id)
    if z is None:
        raise HTTPException(status_code=404, detail=f"Parcel {parcel_id} not found")

    # ZBA case history for this area
    case_count = 0
    area_approval_rate = 0
    if zba_df is not None:
        area_cases = zba_df[zba_df['decision_clean'].notna()]
        district = z['district']
        if district:
            zone_cases = area_cases[area_cases.get('zoning', pd.Series(dtype=str)).fillna('').str.contains(district[:2], na=False)]
            if len(zone_cases) > 10:
                area_cases = zone_cases
        case_count = len(area_cases)
        area_approval_rate = float((area_cases['decision_clean'] == 'APPROVED').mean()) if case_count > 0 else 0

    return {
        "parcel_id": parcel_id,
        "zoning_subdistrict": z['subdistrict'],
        "subdistrict_type": z['subdistrict_type'],
        "subdistrict_use": z['subdistrict_use'],
        "neighborhood": z['neighborhood'],
        "zoning_district": z['district'],
        "article": z['article'],
        "all_zoning_codes": z['all_codes'],
        "multi_zoning": z['multi_zoning'],
        "dimensional_requirements": z['requirements'],
        "allowed_uses": z['allowed_uses'],
        "overlay_districts": z['overlay_districts'],
        "area_zba_cases": case_count,
        "area_approval_rate": round(area_approval_rate, 3),
        "data_source": z['data_source'],
    }


@app.post("/zoning/check_compliance", tags=["Zoning Analysis"])
def zoning_compliance_check(payload: dict):
    """
    Check if a proposed project needs zoning relief.

    Answers: "Do I need a variance?" and "How complex is this case?"
    Uses the SAME subdistrict requirements as /zoning/{parcel_id}.

    Input:
        parcel_id: str (required) — parcel to check against
        proposed_far: float — floor area ratio of proposed project
        proposed_height_ft: int — proposed building height in feet
        proposed_stories: int — proposed number of stories
        proposed_units: int — proposed residential units
        parking_spaces: int — proposed parking spaces
        proposed_use: str — intended use (residential, commercial, mixed-use)
        lot_size_sf: float — lot size (auto-filled from parcel data if available)
        lot_frontage_ft: float — lot frontage (if known)
    """
    parcel_id = payload.get('parcel_id')
    if not parcel_id:
        raise HTTPException(status_code=400, detail="parcel_id is required")

    z = _get_parcel_zoning(parcel_id)
    if z is None:
        raise HTTPException(status_code=404, detail=f"Parcel {parcel_id} not found")

    reqs = z['requirements']

    # Support both flat format and nested {"parcel_id": ..., "proposal": {...}} format
    if 'proposal' in payload and isinstance(payload['proposal'], dict):
        proposal = dict(payload['proposal'])
        proposal['parcel_id'] = parcel_id  # keep parcel_id accessible
    else:
        proposal = dict(payload)
    if 'proposed_height_ft' not in proposal and 'proposed_height' in proposal:
        proposal['proposed_height_ft'] = proposal.pop('proposed_height')
    if 'proposed_stories' not in proposal and 'proposed_stories_count' in proposal:
        proposal['proposed_stories'] = proposal.pop('proposed_stories_count')
    if 'parking_spaces' not in proposal and 'proposed_parking_spaces' in proposal:
        proposal['parking_spaces'] = proposal.pop('proposed_parking_spaces')
    if 'parking_spaces' not in proposal and 'parking' in proposal:
        proposal['parking_spaces'] = proposal.pop('parking')
    if 'lot_frontage_ft' not in proposal and 'lot_frontage' in proposal:
        proposal['lot_frontage_ft'] = proposal.pop('lot_frontage')
    if 'proposed_far' not in proposal and 'far' in proposal:
        proposal['proposed_far'] = proposal.pop('far')

    # Auto-fill lot size and frontage from property data if available
    auto_filled = []
    # Try property assessment parcel database first (175K records)
    if parcel_addr_df is not None and ('lot_size_sf' not in proposal or 'lot_frontage_ft' not in proposal):
        pa_match = parcel_addr_df[parcel_addr_df['parcel_id'] == parcel_id] if 'parcel_id' in parcel_addr_df.columns else pd.DataFrame()
        if not pa_match.empty:
            pa_row = pa_match.iloc[0]
            if 'lot_size_sf' not in proposal:
                for col in ['lot_size', 'lot_size_sf', 'land_sf', 'lot_area']:
                    if col in pa_row.index:
                        ls = pa_row.get(col, 0)
                        if ls and pd.notna(ls) and float(ls) > 0:
                            proposal['lot_size_sf'] = float(ls)
                            auto_filled.append('lot_size_sf')
                            break
            if 'lot_frontage_ft' not in proposal:
                for col in ['lot_frontage', 'lot_frontage_ft', 'front']:
                    if col in pa_row.index:
                        lf = pa_row.get(col, 0)
                        if lf and pd.notna(lf) and float(lf) > 0:
                            proposal['lot_frontage_ft'] = float(lf)
                            auto_filled.append('lot_frontage_ft')
                            break

    # Fallback: try ZBA dataset for lot data
    if 'lot_size_sf' not in proposal and zba_df is not None:
        pa_match = zba_df[zba_df['pa_parcel_id'].astype(str) == parcel_id] if 'pa_parcel_id' in zba_df.columns else pd.DataFrame()
        if not pa_match.empty:
            ls = pa_match.iloc[0].get('lot_size_sf', 0)
            if ls and pd.notna(ls) and float(ls) > 0:
                proposal['lot_size_sf'] = float(ls)
                auto_filled.append('lot_size_sf')

    # Run compliance check against REAL subdistrict requirements
    violations = []
    variances_needed = []
    compliant = True

    # FAR check
    proposed_far = proposal.get('proposed_far', 0)
    max_far = reqs.get('max_far')
    if proposed_far and max_far and proposed_far > max_far:
        compliant = False
        violations.append({
            "type": "far",
            "requirement": f"Max FAR: {max_far}",
            "proposed": f"Proposed FAR: {proposed_far}",
            "excess": f"{((proposed_far / max_far) - 1) * 100:.0f}% over limit",
        })
        variances_needed.append("far")

    # Height check
    proposed_height = proposal.get('proposed_height_ft', 0)
    max_height = reqs.get('max_height_ft')
    if proposed_height and max_height and proposed_height > max_height:
        compliant = False
        violations.append({
            "type": "height",
            "requirement": f"Max height: {max_height} ft",
            "proposed": f"Proposed height: {proposed_height} ft",
            "excess": f"{proposed_height - max_height:.0f} ft over limit",
        })
        variances_needed.append("height")

    # Stories check
    proposed_stories = proposal.get('proposed_stories', 0)
    max_stories = reqs.get('max_stories')
    if proposed_stories and max_stories and proposed_stories > max_stories:
        compliant = False
        violations.append({
            "type": "height",
            "requirement": f"Max stories: {max_stories}",
            "proposed": f"Proposed stories: {proposed_stories}",
            "excess": f"{proposed_stories - max_stories:.1f} stories over limit",
        })
        if "height" not in variances_needed:
            variances_needed.append("height")

    # Lot size check
    lot_size = proposal.get('lot_size_sf', 0)
    min_lot = reqs.get('min_lot_sf')
    if lot_size and min_lot and lot_size < min_lot:
        compliant = False
        violations.append({
            "type": "lot_area",
            "requirement": f"Min lot: {min_lot:,} sf",
            "proposed": f"Lot size: {lot_size:,.0f} sf",
            "deficit": f"{min_lot - lot_size:,.0f} sf under minimum",
        })
        variances_needed.append("lot_area")

    # Frontage check
    frontage = proposal.get('lot_frontage_ft', 0)
    min_frontage = reqs.get('min_frontage_ft')
    if frontage and min_frontage and frontage < min_frontage:
        compliant = False
        violations.append({
            "type": "lot_frontage",
            "requirement": f"Min frontage: {min_frontage} ft",
            "proposed": f"Lot frontage: {frontage} ft",
            "deficit": f"{min_frontage - frontage:.0f} ft under minimum",
        })
        variances_needed.append("lot_frontage")

    # Front setback check
    front_setback = proposal.get('front_setback_ft')
    min_front = reqs.get('min_front_yard_ft')
    if front_setback is not None and min_front and front_setback < min_front:
        compliant = False
        violations.append({
            "type": "front_setback",
            "requirement": f"Min front setback: {min_front} ft",
            "proposed": f"Proposed front setback: {front_setback} ft",
            "deficit": f"{min_front - front_setback:.0f} ft under minimum",
        })
        variances_needed.append("front_setback")

    # Side setback check
    side_setback = proposal.get('side_setback_ft')
    min_side = reqs.get('min_side_yard_ft')
    if side_setback is not None and min_side and side_setback < min_side:
        compliant = False
        violations.append({
            "type": "side_setback",
            "requirement": f"Min side setback: {min_side} ft",
            "proposed": f"Proposed side setback: {side_setback} ft",
            "deficit": f"{min_side - side_setback:.0f} ft under minimum",
        })
        variances_needed.append("side_setback")

    # Rear setback check
    rear_setback = proposal.get('rear_setback_ft')
    min_rear = reqs.get('min_rear_yard_ft')
    if rear_setback is not None and min_rear and rear_setback < min_rear:
        compliant = False
        violations.append({
            "type": "rear_setback",
            "requirement": f"Min rear setback: {min_rear} ft",
            "proposed": f"Proposed rear setback: {rear_setback} ft",
            "deficit": f"{min_rear - rear_setback:.0f} ft under minimum",
        })
        variances_needed.append("rear_setback")

    # Parking check
    proposed_units = proposal.get('proposed_units', 0)
    parking_spaces = proposal.get('parking_spaces')
    parking_per_unit = reqs.get('parking_per_unit')
    if proposed_units and parking_spaces is not None and parking_per_unit:
        required_parking = int(proposed_units * parking_per_unit)
        if parking_spaces < required_parking:
            compliant = False
            violations.append({
                "type": "parking",
                "requirement": f"Required: {required_parking} spaces ({parking_per_unit} per unit)",
                "proposed": f"Provided: {parking_spaces} spaces",
                "deficit": f"{required_parking - parking_spaces} spaces short",
            })
            variances_needed.append("parking")

    # Lot coverage / open space check
    lot_coverage = proposal.get('lot_coverage_pct', 0)
    max_coverage = reqs.get('max_lot_coverage_pct')
    if lot_coverage and max_coverage and lot_coverage > max_coverage:
        compliant = False
        violations.append({
            "type": "open_space",
            "requirement": f"Max lot coverage: {max_coverage}%",
            "proposed": f"Proposed coverage: {lot_coverage}%",
            "excess": f"{lot_coverage - max_coverage:.0f}% over limit (insufficient open space)",
        })
        variances_needed.append("open_space")

    # Conditional use check — is the proposed use allowed in this district?
    proposed_use = proposal.get('proposed_use', '').lower()
    allowed_uses = z.get('allowed_uses', [])
    if proposed_use and allowed_uses:
        use_allowed = any(proposed_use in u.lower() for u in allowed_uses)
        if not use_allowed:
            compliant = False
            violations.append({
                "type": "conditional_use",
                "requirement": f"Allowed uses: {', '.join(allowed_uses[:5])}",
                "proposed": f"Proposed use: {proposed_use}",
                "excess": "Use not permitted as-of-right — conditional use permit or variance required",
            })
            variances_needed.append("conditional_use")

    # Complexity assessment
    num_violations = len(variances_needed)
    if num_violations == 0:
        complexity = "low"
        complexity_note = "Project appears to comply with zoning requirements. May not need ZBA relief."
    elif num_violations <= 2:
        complexity = "moderate"
        complexity_note = f"Project needs {num_violations} variance(s). Common for Boston ZBA — most projects with 1-2 variances are approved."
    else:
        complexity = "high"
        complexity_note = f"Project needs {num_violations} variances. More complex ZBA case — consider reducing scope or hiring experienced zoning attorney."

    # Overlay district warnings
    overlay_warnings = []
    for overlay in z.get('overlay_districts', []):
        overlay_warnings.append(overlay['note'])

    result = {
        "compliant": compliant,
        "parcel_id": parcel_id,
        "zoning_subdistrict": z['subdistrict'],
        "subdistrict_type": z['subdistrict_type'],
        "neighborhood": z['neighborhood'],
        "zoning_district": z['district'],
        "article": z['article'],
        "requirements": reqs,
        "violations": violations,
        "variances_needed": variances_needed,
        "num_variances_needed": len(variances_needed),
        "complexity": complexity,
        "complexity_note": complexity_note,
        "overlay_districts": z.get('overlay_districts', []),
        "overlay_warnings": overlay_warnings,
        "data_source": z['data_source'],
        "auto_filled": auto_filled,
        "lot_size_sf": proposal.get('lot_size_sf'),
        "lot_frontage_ft": proposal.get('lot_frontage_ft'),
    }

    # Add historical context
    if zba_df is not None and variances_needed:
        var_approval_rates = {}
        df_with_dec = zba_df[zba_df['decision_clean'].notna()]
        for var_type in variances_needed:
            col = f'var_{var_type}'
            if col in df_with_dec.columns:
                var_cases = df_with_dec[df_with_dec[col] == 1]
                if len(var_cases) > 0:
                    rate = float((var_cases['decision_clean'] == 'APPROVED').mean())
                    var_approval_rates[var_type] = {
                        "approval_rate": round(rate, 3),
                        "total_cases": len(var_cases),
                        "note": f"Based on {len(var_cases)} ZBA cases involving {var_type} variances"
                    }
        result['variance_historical_rates'] = var_approval_rates

    return result


@app.post("/zoning/full_analysis", tags=["Zoning Analysis"])
def full_zoning_analysis(payload: dict):
    """
    Complete zoning analysis workflow — single endpoint that answers ALL questions:
    1. What zoning district is this parcel in?
    2. What are the dimensional requirements?
    3. Does my project need variances?
    4. How complex is this case?
    5. What's the predicted approval probability?

    This is the developer's complete pre-filing analysis in one API call.
    """
    parcel_id = payload.get('parcel_id')
    if not parcel_id:
        raise HTTPException(status_code=400, detail="parcel_id is required")

    # Step 1: Zoning lookup
    try:
        zoning_info = zoning_analysis(parcel_id)
    except HTTPException:
        raise HTTPException(status_code=404, detail=f"Parcel {parcel_id} not found")

    # Step 2: Compliance check — uses the SAME subdistrict requirements
    compliance = zoning_compliance_check(payload)

    # Step 3: Run prediction (reuse analyze_proposal)
    prediction_input = dict(payload)
    prediction_input['variances'] = compliance.get('variances_needed', [])
    if 'ward' not in prediction_input:
        if gdf is not None:
            matches = gdf[gdf['parcel_id'] == parcel_id]
            if not matches.empty and 'ward' in matches.columns:
                prediction_input['ward'] = str(matches.iloc[0].get('ward', ''))

    try:
        prediction = analyze_proposal(prediction_input)
    except Exception as e:
        prediction = {"error": str(e), "probability": None}

    # Step 4: Compile full report
    return {
        "parcel_id": parcel_id,

        # Q1: What zoning district?
        "zoning": {
            "subdistrict": zoning_info.get('zoning_subdistrict', ''),
            "subdistrict_type": zoning_info.get('subdistrict_type', ''),
            "neighborhood": zoning_info.get('neighborhood', ''),
            "district": zoning_info['zoning_district'],
            "article": zoning_info['article'],
            "allowed_uses": zoning_info['allowed_uses'],
        },

        # Q2: Dimensional requirements
        "requirements": zoning_info['dimensional_requirements'],

        # Q3: Do I need variances?
        "compliance": {
            "compliant": compliance['compliant'],
            "variances_needed": compliance['variances_needed'],
            "num_variances": compliance['num_variances_needed'],
            "violations": compliance['violations'],
        },

        # Q4: How complex is this?
        "complexity": {
            "level": compliance['complexity'],
            "note": compliance['complexity_note'],
            "area_cases": zoning_info['area_zba_cases'],
            "area_approval_rate": zoning_info['area_approval_rate'],
        },

        # Overlay district warnings
        "overlay_districts": zoning_info.get('overlay_districts', []),

        # Q5: What's the predicted outcome?
        "prediction": prediction,

        "data_source": zoning_info.get('data_source', ''),
        "disclaimer": "This analysis is for informational purposes only. Always verify zoning requirements with the Boston Inspectional Services Department and consult a licensed zoning attorney before filing with the ZBA.",
    }


# NOTE: /zoning/districts moved above /zoning/{parcel_id} to avoid route conflict


@app.post("/variance_analysis", tags=["Zoning Analysis"])
def variance_analysis(payload: dict):
    """
    THE KEY QUESTION: "Based on recent ZBA decisions, how likely will my
    proposal requiring X, Y, Z zoning variances pass?"

    Returns a data-driven answer using actual historical outcomes —
    not a model prediction, but real approval rates for the exact
    variance combination requested, broken down by ward, attorney
    representation, and number of variances.
    """
    if zba_df is None:
        raise HTTPException(status_code=500, detail="ZBA data not loaded")

    variances = payload.get('variances', [])
    ward = payload.get('ward')
    has_attorney = payload.get('has_attorney', False)
    num_proposed_variances = len(variances) if variances else payload.get('num_variances', 0)

    df = zba_df[zba_df['decision_clean'].notna()].copy()
    vt = df['variance_types'].fillna('')

    # --- 1. Overall approval rate ---
    overall_rate = float((df['decision_clean'] == 'APPROVED').mean())
    overall_count = len(df)

    # --- 2. Approval rate for EXACT variance combination ---
    if variances:
        mask = pd.Series(True, index=df.index)
        for v in variances:
            mask = mask & vt.str.contains(v.lower(), na=False)
        combo_cases = df[mask]
    else:
        combo_cases = df

    combo_count = len(combo_cases)
    combo_approved = int((combo_cases['decision_clean'] == 'APPROVED').sum())
    combo_denied = combo_count - combo_approved
    combo_rate = float(combo_approved / combo_count) if combo_count > 0 else overall_rate

    # --- 3. Approval rate by ward for this combo ---
    ward_rate = None
    ward_count = 0
    ward_note = ""
    if ward and combo_count > 0:
        try:
            ward_float = float(ward)
            ward_cases = combo_cases[combo_cases['ward'] == ward_float]
            ward_count = len(ward_cases)
            if ward_count >= 3:
                ward_rate = float((ward_cases['decision_clean'] == 'APPROVED').mean())
                ward_note = f"In Ward {int(ward_float)}, {ward_count} cases with this variance combination had a {ward_rate:.0%} approval rate"
            else:
                ward_note = f"Only {ward_count} case(s) with this combination in Ward {int(ward_float)} — insufficient data for ward-specific rate"
        except (ValueError, TypeError):
            pass

    # --- 4. Attorney effect for this combo ---
    attorney_effect = {}
    if combo_count > 0 and 'has_attorney' in combo_cases.columns:
        with_atty = combo_cases[combo_cases['has_attorney'] == 1]
        without_atty = combo_cases[combo_cases['has_attorney'] == 0]
        if len(with_atty) >= 3 and len(without_atty) >= 3:
            rate_with = float((with_atty['decision_clean'] == 'APPROVED').mean())
            rate_without = float((without_atty['decision_clean'] == 'APPROVED').mean())
            attorney_effect = {
                "with_attorney": {"rate": round(rate_with, 3), "cases": len(with_atty)},
                "without_attorney": {"rate": round(rate_without, 3), "cases": len(without_atty)},
                "difference": round(rate_with - rate_without, 3),
            }

    # --- 5. Approval rate by number of variances ---
    variance_count_rates = []
    if 'num_variances' in df.columns:
        for nv in range(1, 8):
            nv_cases = df[df['num_variances'] == nv]
            if len(nv_cases) >= 10:
                rate = float((nv_cases['decision_clean'] == 'APPROVED').mean())
                variance_count_rates.append({
                    "num_variances": nv,
                    "approval_rate": round(rate, 3),
                    "cases": len(nv_cases),
                })

    # --- 6. Per-variance approval rates ---
    per_variance_rates = {}
    if variances:
        for v in variances:
            v_cases = df[vt.str.contains(v.lower(), na=False)]
            if len(v_cases) > 0:
                rate = float((v_cases['decision_clean'] == 'APPROVED').mean())
                per_variance_rates[v] = {
                    "approval_rate": round(rate, 3),
                    "cases": len(v_cases),
                }

    # --- 7. What distinguishes denied cases in this combo? ---
    denial_factors = []
    if combo_count > 0 and combo_denied > 0:
        denied = combo_cases[combo_cases['decision_clean'] == 'DENIED']
        approved = combo_cases[combo_cases['decision_clean'] == 'APPROVED']

        # Compare averages
        if 'num_variances' in df.columns and len(denied) >= 3:
            d_avg = denied['num_variances'].mean()
            a_avg = approved['num_variances'].mean()
            if d_avg > a_avg + 0.5:
                denial_factors.append(f"Denied cases averaged {d_avg:.1f} variances vs {a_avg:.1f} for approved — more variances = higher risk")

        if 'has_attorney' in df.columns and len(denied) >= 3:
            d_atty = denied['has_attorney'].mean()
            a_atty = approved['has_attorney'].mean()
            if a_atty > d_atty + 0.1:
                denial_factors.append(f"Only {d_atty:.0%} of denied cases had attorney representation vs {a_atty:.0%} of approved cases")

    # --- 8. Build the narrative answer ---
    variance_list = ', '.join(variances) if variances else 'unspecified variances'

    if combo_count >= 10:
        headline = f"Based on {combo_count} recent ZBA cases requesting {variance_list}, the approval rate is {combo_rate:.0%}."
    elif combo_count >= 3:
        headline = f"Based on {combo_count} cases with {variance_list} (limited data), the approval rate is {combo_rate:.0%}."
    else:
        headline = f"Very few cases ({combo_count}) match your exact combination of {variance_list}. Using the overall rate of {overall_rate:.0%} across {overall_count} cases."

    details = []
    if ward_note:
        details.append(ward_note)
    if attorney_effect:
        diff = attorney_effect['difference']
        if diff > 0.05:
            atty_word = "increases" if has_attorney else "would increase"
            details.append(f"Attorney representation {atty_word} approval odds by {diff:.0%} for this variance combination ({attorney_effect['with_attorney']['rate']:.0%} with vs {attorney_effect['without_attorney']['rate']:.0%} without)")
    if num_proposed_variances > 0:
        matching_nv = [r for r in variance_count_rates if r['num_variances'] == num_proposed_variances]
        if matching_nv:
            details.append(f"Cases with exactly {num_proposed_variances} variance(s) have a {matching_nv[0]['approval_rate']:.0%} approval rate ({matching_nv[0]['cases']} cases)")
    for factor in denial_factors:
        details.append(factor)

    return {
        "question": f"How likely will a proposal requiring {variance_list} pass?",
        "headline": headline,
        "details": details,
        "data": {
            "overall": {"rate": round(overall_rate, 3), "cases": overall_count},
            "your_combination": {"rate": round(combo_rate, 3), "cases": combo_count, "approved": combo_approved, "denied": combo_denied},
            "ward_specific": {"rate": round(ward_rate, 3) if ward_rate is not None else None, "cases": ward_count, "ward": ward} if ward else None,
            "attorney_effect": attorney_effect or None,
            "per_variance": per_variance_rates,
            "by_variance_count": variance_count_rates,
        },
        "recommendation": (
            "Strong historical precedent for approval." if combo_rate >= 0.85 else
            "Moderate risk — consider strengthening your application." if combo_rate >= 0.65 else
            "Higher risk — consult an experienced zoning attorney before filing."
        ),
        "disclaimer": "Based on historical ZBA decisions 2020-2026. Past decisions do not guarantee future outcomes. Consult a qualified zoning attorney.",
    }


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

    # Use pre-computed normalized addresses — word-boundary on leading number
    addr_pattern = re.escape(addr_norm)
    if re.match(r'^\d', addr_norm):
        addr_pattern = r'(?:^|\s|-)' + addr_pattern
    if '_addr_norm' in addr_df.columns:
        matches = addr_df[addr_df['_addr_norm'].str.contains(addr_pattern, na=False, regex=True)]
    else:
        addr_df = addr_df.copy()
        addr_df['_addr_norm'] = addr_df['address_clean'].apply(normalize_address)
        matches = addr_df[addr_df['_addr_norm'].str.contains(addr_pattern, na=False, regex=True)]
    sort_col = 'hearing_date' if 'hearing_date' in matches.columns else ('filing_date' if 'filing_date' in matches.columns else 'case_number')
    matches = matches.sort_values(sort_col, ascending=False, na_position='last').head(limit)

    # Determine best zoning column
    z_col = 'zoning_clean' if 'zoning_clean' in matches.columns else ('zoning_district' if 'zoning_district' in matches.columns else 'zoning')
    d_col = 'hearing_date' if 'hearing_date' in matches.columns else ('filing_date' if 'filing_date' in matches.columns else None)

    cases = []
    seen_cases = set()
    for _, row in matches.iterrows():
        cn = str(row.get('case_number') or '')
        if cn in seen_cases:
            continue
        seen_cases.add(cn)
        _ward_val = row.get('ward', '')
        try:
            _ward_str = str(int(float(_ward_val))) if pd.notna(_ward_val) and str(_ward_val).replace('.','',1).isdigit() else str(_ward_val or '')
        except (ValueError, TypeError):
            _ward_str = str(_ward_val or '')
        _case_item = {
            "case_number": cn,
            "address": str(row.get('address_clean') or ''),
            "decision": str(row.get('decision_clean') or ''),
            "ward": _ward_str,
            "zoning": str(row.get(z_col) or ''),
            "date": _format_date(row.get(d_col) or '') if d_col else '',
            "variances": str(row.get('variance_types') or ''),
            "has_attorney": bool(row.get('has_attorney', 0)),
            "project_type": ', '.join([
                pt.replace('proj_', '') for pt in [
                    'proj_addition', 'proj_new_construction', 'proj_renovation',
                    'proj_conversion', 'proj_demolition', 'proj_multi_family',
                    'proj_single_family', 'proj_mixed_use', 'proj_adu', 'proj_roof_deck'
                ] if row.get(pt, 0) == 1
            ]) or 'unknown',
        }
        _app_name = safe_str(row.get('applicant_name'))
        _contact = safe_str(row.get('contact'))
        if _app_name:
            _case_item["applicant"] = _app_name
        if _contact:
            _case_item["contact"] = _contact
        cases.append(_case_item)

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

    # --- Contact win rate lookup ---
    contact_win_rate = overall_rate
    if model_package:
        contact_rates = model_package.get('contact_win_rates', {})
        if contact_rates:
            contact_win_rate = overall_rate  # Default for unknown contacts

    # --- Ward x Zoning interaction rate ---
    ward_zoning_rate = overall_rate
    if model_package:
        wz_rates = model_package.get('ward_zoning_rates', {})
        wz_key = f"{ward or 'unknown'}_{zoning}"
        ward_zoning_rate = wz_rates.get(wz_key, overall_rate)

    # --- Year x Ward interaction rate ---
    year_ward_rate = overall_rate
    if model_package:
        yw_rates = model_package.get('year_ward_rates', {})
        yw_key = f"2026_{ward or 'unknown'}"
        year_ward_rate = yw_rates.get(yw_key, overall_rate)

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
        # Representation (4) — known at filing
        'has_attorney': int(has_attorney),
        'bpda_involved': 0,
        'is_building_appeal': 0,  # Default to zoning (most common)
        'is_refusal_appeal': 0,   # Would need to be specified by user
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
        # Location (6) — historical rates from training data
        'ward_approval_rate': ward_rate,
        'zoning_approval_rate': zoning_rate,
        'attorney_win_rate': attorney_win_rate,
        'contact_win_rate': contact_win_rate,
        'ward_zoning_rate': ward_zoning_rate,
        'year_ward_rate': year_ward_rate,
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
        # Log transforms (3)
        'lot_size_log': float(np.log1p(lot_size)),
        'total_value_log': float(np.log1p(total_value)),
        'prior_permits_log': float(np.log1p(prior_permits)),
        # Additional interactions (4)
        'contact_x_appeal': contact_win_rate * 0,  # is_building_appeal=0 by default
        'attorney_x_building': int(has_attorney) * 0,  # is_building_appeal=0
        'many_variances': int(num_variances >= 3),
        'has_property_data': int(total_value > 0),
        # Meta-features (3)
        'project_complexity': sum(v for v in proj_features.values()),
        'total_violations': excessive_far + insufficient_lot + insufficient_frontage + insufficient_yard + insufficient_parking,
        'num_features_active': sum([
            is_residential, is_commercial, int(has_attorney), 0,  # bpda
            excessive_far, insufficient_lot, insufficient_frontage,
            insufficient_yard, insufficient_parking,
            *proj_features.values(),
            int(num_variances > 0),  # is_variance
            int(prior_permits > 0),  # has_prior_permits
            is_high_value,
        ]),
    }

    return features


def _clean_case_address(row):
    """Clean OCR garbage from case addresses, trying multiple fallback fields."""
    # Try multiple address fields in order of quality
    for field in ('address_clean', 'address', 'property_address'):
        addr = str(row.get(field) or '')
        if not addr or addr in ('', 'nan', 'None', 'Unknown'):
            continue
        # Skip addresses that are clearly not real street addresses
        if len(addr) > 60 or '\n' in addr:
            continue
        if any(w in addr.lower() for w in ('record', 'conformity', 'hearing', 'board', 'appeal')):
            continue
        # Skip addresses that are just numbers (parcel IDs or case number fragments)
        stripped = addr.replace(' ', '').replace('-', '')
        if stripped.isdigit():
            continue
        # Must contain at least one letter (real addresses have street names)
        if not any(c.isalpha() for c in addr):
            continue
        return addr
    return 'Address not available'


def _clean_case_date(row):
    """Extract a clean date string from case data."""
    for field in ('hearing_date', 'filing_date', 'date'):
        val = row.get(field)
        if val is None or (isinstance(val, float) and np.isnan(val)):
            continue
        date_str = str(val).strip()
        if date_str in ('', 'nan', 'None', 'NaT'):
            continue
        # Try to parse and format cleanly
        try:
            dt = pd.to_datetime(date_str)
            return dt.strftime('%Y-%m-%d')
        except Exception:
            # Return raw string (don't truncate — named months get destroyed)
            return date_str.strip()
    return ''


def get_similar_cases(ward, variances, project_type=None, limit=5):
    """Find similar historical ZBA cases using relevance scoring.

    Returns a mix of approved AND denied cases for contrast.
    """
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

    # --- Stratified sampling: include both approved AND denied cases ---
    # This gives the developer a realistic picture, not just success stories
    approved_pool = relevant[relevant['decision_clean'] == 'APPROVED']
    denied_pool = relevant[relevant['decision_clean'] == 'DENIED']

    cases = []
    # If we have denied cases, include 1-2 for contrast
    denied_to_show = min(2, len(denied_pool), max(1, limit // 3))
    approved_to_show = limit - denied_to_show

    if len(denied_pool) == 0:
        # No denied cases — show all approved
        sample = relevant.head(limit)
    else:
        # Mix: top approved + top denied
        approved_sample = approved_pool.head(approved_to_show)
        denied_sample = denied_pool.head(denied_to_show)
        sample = pd.concat([approved_sample, denied_sample]).sort_values('_relevance', ascending=False)

    for _, row in sample.iterrows():
        addr = _clean_case_address(row)
        # Skip cases with no real address — don't show "Address not available"
        if addr == 'Address not available' or not addr.strip():
            continue
        date_str = _clean_case_date(row)
        cases.append({
            "case_number": str(row.get('case_number') or ''),
            "address": addr,
            "decision": str(row.get('decision_clean') or ''),
            "ward": str(int(float(row['ward']))) if pd.notna(row.get('ward')) else '',
            "date": date_str,
            "relevance_score": round(float(row.get('_relevance', 0)), 1),
        })

    return cases, total, approval_rate


def _estimate_timeline(ward=None, appeal_type=None):
    """Estimate ZBA decision timeline using pre-computed tracker statistics.

    Returns a rich breakdown with per-phase timing (filing->hearing, hearing->decision)
    and ward-specific data when available.  Based on ~14K real tracker records with
    99.5% date coverage (vs the old 14% from OCR filing_date).
    """
    if timeline_stats is None:
        return None

    overall = timeline_stats.get("overall", {})
    if not overall:
        return None

    # Determine which source to use: ward-specific or overall
    ward_specific = False
    ward_str = None
    source = overall

    if ward:
        try:
            ward_str = str(int(float(ward)))
        except (ValueError, TypeError):
            ward_str = str(ward).strip()

        ward_data = timeline_stats.get("by_ward", {}).get(ward_str)
        if ward_data and "filing_to_decision" in ward_data:
            source = ward_data
            ward_specific = True

    # Build phase breakdown
    phases = {}
    for phase_key, label in [
        ("filing_to_hearing", "Filing to First Hearing"),
        ("hearing_to_decision", "Hearing to Decision"),
        ("filing_to_decision", "Filing to Decision (total)"),
        ("filing_to_closed", "Filing to Case Closed"),
    ]:
        phase_data = source.get(phase_key)
        if phase_data:
            phases[phase_key] = {
                "label": label,
                "median_days": phase_data["median_days"],
                "p25_days": phase_data["p25_days"],
                "p75_days": phase_data["p75_days"],
                "cases_used": phase_data["cases_used"],
            }

    # Primary metric: filing_to_decision
    primary = source.get("filing_to_decision", overall.get("filing_to_decision", {}))

    result = {
        "median_days": primary.get("median_days"),
        "p25_days": primary.get("p25_days"),
        "p75_days": primary.get("p75_days"),
        "cases_used": primary.get("cases_used", 0),
        "ward_specific": ward_specific,
        "phases": phases,
    }

    if ward_specific and ward_str:
        result["ward"] = ward_str
        # Also include overall for comparison
        overall_primary = overall.get("filing_to_decision")
        if overall_primary:
            result["overall_median_days"] = overall_primary["median_days"]

    # Include appeal type breakdown if requested
    if appeal_type:
        atype_data = timeline_stats.get("by_appeal_type", {}).get(appeal_type)
        if atype_data and "filing_to_decision" in atype_data:
            result["appeal_type_median_days"] = atype_data["filing_to_decision"]["median_days"]
            result["appeal_type"] = appeal_type

    return result


# Human-readable labels for model features (used by SHAP explainability)
FEATURE_LABELS = {
    # Variance features
    'num_variances': 'Number of variances requested',
    'var_height': 'Height variance requested',
    'var_far': 'Floor Area Ratio (FAR) variance',
    'var_lot_area': 'Lot area variance',
    'var_lot_frontage': 'Lot frontage variance',
    'var_front_setback': 'Front setback variance',
    'var_rear_setback': 'Rear setback variance',
    'var_side_setback': 'Side setback variance',
    'var_parking': 'Parking variance requested',
    'var_conditional_use': 'Conditional use permit',
    'var_open_space': 'Open space variance',
    'var_density': 'Density variance',
    'var_nonconforming': 'Nonconforming use extension',
    # Violation types
    'excessive_far': 'Exceeds allowed floor area ratio',
    'insufficient_lot': 'Lot smaller than required',
    'insufficient_frontage': 'Lot frontage below minimum',
    'insufficient_yard': 'Yard/setback below minimum',
    'insufficient_parking': 'Fewer parking spaces than required',
    # Use type
    'is_residential': 'Residential use',
    'is_commercial': 'Commercial use',
    # Representation
    'has_attorney': 'Attorney representation',
    'bpda_involved': 'BPDA review involved',
    'is_building_appeal': 'Building code appeal (vs. zoning)',
    'is_refusal_appeal': 'Appeal of a building permit refusal',
    # Project types
    'proj_demolition': 'Involves demolition',
    'proj_new_construction': 'New construction project',
    'proj_addition': 'Addition to existing building',
    'proj_conversion': 'Building conversion',
    'proj_renovation': 'Renovation project',
    'proj_subdivision': 'Land subdivision',
    'proj_adu': 'Accessory dwelling unit (ADU)',
    'proj_roof_deck': 'Roof deck addition',
    'proj_parking': 'Parking-related project',
    'proj_single_family': 'Single-family project',
    'proj_multi_family': 'Multi-family project',
    'proj_mixed_use': 'Mixed-use project',
    # Legal
    'article_80': 'Large project (Article 80 review)',
    'is_conditional_use': 'Conditional use application',
    'is_variance': 'Variance application',
    'num_articles': 'Number of zoning articles involved',
    # Scale
    'proposed_units': 'Number of proposed units',
    'proposed_stories': 'Number of proposed stories',
    # Location rates
    'ward_approval_rate': 'Historical approval rate in this ward',
    'zoning_approval_rate': 'Historical approval rate for this zoning district',
    'attorney_win_rate': 'Attorney track record at ZBA',
    'contact_win_rate': 'Applicant track record at ZBA',
    'ward_zoning_rate': 'Approval trend for this ward + zoning combination',
    'year_ward_rate': 'Recent approval trend in this ward',
    # Recency
    'year_recency': 'How recent the case is',
    # Property
    'lot_size_sf': 'Lot size (sq ft)',
    'total_value': 'Property assessed value',
    'property_age': 'Building age (years)',
    'living_area': 'Living area (sq ft)',
    'is_high_value': 'High-value property (>$1M)',
    'value_per_sqft': 'Property value per sq ft',
    # Permits
    'prior_permits': 'Number of prior building permits',
    'has_prior_permits': 'Has prior building permit history',
    # Interactions
    'interact_height_stories': 'Height variance × stories (scale of height issue)',
    'interact_attorney_variances': 'Attorney × variance count (complex case with representation)',
    'interact_highvalue_permits': 'High-value property with permit history',
    # Log transforms
    'lot_size_log': 'Lot size (log scale)',
    'total_value_log': 'Property value (log scale)',
    'prior_permits_log': 'Prior permits (log scale)',
    # Additional
    'contact_x_appeal': 'Applicant track record × appeal type',
    'attorney_x_building': 'Attorney × building appeal',
    'many_variances': 'Requesting 3+ variances',
    'has_property_data': 'Property assessment data available',
    # Meta
    'project_complexity': 'Overall project complexity score',
    'total_violations': 'Total number of zoning violations',
    'num_features_active': 'Number of active risk factors',
}


def _get_variance_history(variances, ward=None, has_attorney=False):
    """Get real historical approval rates for a variance combination.

    This is the core data that answers the money question:
    'Based on recent ZBA decisions, how likely will my proposal pass?'
    """
    if zba_df is None:
        return None

    df = zba_df[zba_df['decision_clean'].notna()].copy()
    vt = df['variance_types'].fillna('')

    # Overall approval rate
    overall_rate = float((df['decision_clean'] == 'APPROVED').mean())

    # Exact combo match
    if variances:
        mask = pd.Series(True, index=df.index)
        for v in variances:
            mask = mask & vt.str.contains(v.lower(), na=False)
        combo_cases = df[mask]
    else:
        combo_cases = df

    combo_count = len(combo_cases)
    combo_approved = int((combo_cases['decision_clean'] == 'APPROVED').sum())
    combo_denied = combo_count - combo_approved
    combo_rate = float(combo_approved / combo_count) if combo_count > 0 else overall_rate

    # Ward-specific rate
    ward_rate = None
    ward_count = 0
    if ward and combo_count > 0:
        try:
            ward_float = float(ward)
            ward_cases = combo_cases[combo_cases['ward'] == ward_float]
            ward_count = len(ward_cases)
            if ward_count >= 3:
                ward_rate = float((ward_cases['decision_clean'] == 'APPROVED').mean())
        except (ValueError, TypeError):
            pass

    # Attorney effect
    attorney_effect = None
    if combo_count > 0 and 'has_attorney' in combo_cases.columns:
        with_atty = combo_cases[combo_cases['has_attorney'] == 1]
        without_atty = combo_cases[combo_cases['has_attorney'] == 0]
        if len(with_atty) >= 3 and len(without_atty) >= 3:
            rate_with = float((with_atty['decision_clean'] == 'APPROVED').mean())
            rate_without = float((without_atty['decision_clean'] == 'APPROVED').mean())
            attorney_effect = {
                "with_attorney": round(rate_with, 3),
                "without_attorney": round(rate_without, 3),
                "difference": round(rate_with - rate_without, 3),
                "cases_with": len(with_atty),
                "cases_without": len(without_atty),
            }

    # Per-variance rates
    per_variance = {}
    if variances:
        for v in variances:
            v_cases = df[vt.str.contains(v.lower(), na=False)]
            if len(v_cases) > 0:
                per_variance[v] = {
                    "approval_rate": round(float((v_cases['decision_clean'] == 'APPROVED').mean()), 3),
                    "cases": len(v_cases),
                }

    return {
        "combo_rate": round(combo_rate, 3),
        "combo_cases": combo_count,
        "combo_approved": combo_approved,
        "combo_denied": combo_denied,
        "overall_rate": round(overall_rate, 3),
        "ward_rate": round(ward_rate, 3) if ward_rate is not None else None,
        "ward_cases": ward_count,
        "attorney_effect": attorney_effect,
        "per_variance": per_variance,
    }


def _build_key_factors(variances, ward, has_attorney, project_type, proposed_units, vh):
    """Build key_factors from REAL data, not hardcoded percentages.

    vh = variance history dict from _get_variance_history()
    """
    factors = []

    if vh is None:
        return ["Insufficient data for detailed analysis"]

    # 1. THE HEADLINE — real historical rate for this exact combination
    combo_rate = vh['combo_rate']
    combo_cases = vh['combo_cases']
    variance_list = ', '.join(variances) if variances else 'no specific variances'

    if combo_cases >= 10:
        factors.append(
            f"Based on {combo_cases} ZBA cases with {variance_list}: "
            f"{combo_rate:.0%} were approved"
        )
    elif combo_cases >= 3:
        factors.append(
            f"Limited data: {combo_cases} cases with {variance_list} — "
            f"{combo_rate:.0%} approved (small sample)"
        )
    else:
        factors.append(
            f"Very few cases ({combo_cases}) match your exact combination — "
            f"overall ZBA approval rate is {vh['overall_rate']:.0%}"
        )

    # 2. Attorney effect — computed from real data
    ae = vh.get('attorney_effect')
    if ae:
        diff = ae['difference']
        if has_attorney:
            if diff > 0.02:
                factors.append(
                    f"Attorney representation: {ae['with_attorney']:.0%} approval rate "
                    f"({ae['cases_with']} cases) vs {ae['without_attorney']:.0%} without "
                    f"({ae['cases_without']} cases) — a {diff:.0%} advantage"
                )
            else:
                factors.append(
                    f"Attorney representation: minimal effect for this combination "
                    f"({ae['with_attorney']:.0%} with vs {ae['without_attorney']:.0%} without)"
                )
        else:
            if diff > 0.05:
                factors.append(
                    f"No attorney: cases with representation have {diff:.0%} higher approval rate "
                    f"for this combination ({ae['with_attorney']:.0%} vs {ae['without_attorney']:.0%})"
                )

    # 3. Ward-specific rate — from real data
    if vh.get('ward_rate') is not None and vh['ward_cases'] >= 3:
        ward_rate = vh['ward_rate']
        ward_cases = vh['ward_cases']
        if abs(ward_rate - combo_rate) > 0.05:
            direction = "higher" if ward_rate > combo_rate else "lower"
            factors.append(
                f"Ward {ward}: {ward_rate:.0%} approval for this combination "
                f"({ward_cases} cases) — {direction} than citywide"
            )
        else:
            factors.append(
                f"Ward {ward}: {ward_rate:.0%} approval for this combination ({ward_cases} cases)"
            )

    # 4. Per-variance insight — actual rates
    pv = vh.get('per_variance', {})
    for v_name, v_data in pv.items():
        rate = v_data['approval_rate']
        cases = v_data['cases']
        if rate < vh['overall_rate'] - 0.05:
            factors.append(
                f"{v_name.title()} variance: {rate:.0%} approval ({cases} cases) — "
                f"below the overall {vh['overall_rate']:.0%} rate"
            )
        elif rate > vh['overall_rate'] + 0.05:
            factors.append(
                f"{v_name.title()} variance: {rate:.0%} approval ({cases} cases) — "
                f"above the overall {vh['overall_rate']:.0%} rate"
            )

    # 5. Project type context (if applicable)
    if project_type:
        pt_lower = project_type.lower()
        if 'new_construction' in pt_lower or 'new construction' in pt_lower:
            factors.append("New construction typically faces more Board scrutiny than renovations/additions")
        elif 'addition' in pt_lower:
            factors.append("Additions to existing buildings generally have higher approval rates")

    # 6. Scale warning
    if proposed_units > 5:
        factors.append(
            f"Large project ({proposed_units} units) — may attract community attention and Board questions"
        )

    return factors


def _build_recommendations(prob, variances, ward, has_attorney, project_type, proposed_units,
                           proposed_stories, variance_history, top_drivers):
    """Generate actionable recommendations to improve approval odds.

    Uses SHAP drivers and variance_history to produce specific, data-backed advice.
    Returns up to 4 recommendations sorted by estimated impact (high first).
    """
    recs = []

    vh = variance_history or {}
    per_variance = vh.get('per_variance', {})
    ae = vh.get('attorney_effect')
    overall_rate = vh.get('overall_rate', 0.88)

    # Build a quick lookup of SHAP drivers by feature_name
    shap_lookup = {}
    for d in (top_drivers or []):
        shap_lookup[d.get('feature_name', '')] = d.get('shap_value', 0)

    # ---------------------------------------------------------------
    # 1. Attorney recommendation
    # ---------------------------------------------------------------
    if not has_attorney:
        atty_diff = 0
        with_rate = 0
        without_rate = 0
        if ae and ae.get('difference', 0) > 0.05:
            atty_diff = ae['difference']
            with_rate = ae['with_attorney']
            without_rate = ae['without_attorney']
        elif ae:
            atty_diff = ae.get('difference', 0)
            with_rate = ae.get('with_attorney', 0)
            without_rate = ae.get('without_attorney', 0)

        if atty_diff > 0.05:
            recs.append({
                "action": "Hire a zoning attorney",
                "detail": (
                    f"Cases like yours have {with_rate:.0%} approval with representation "
                    f"vs {without_rate:.0%} without ({atty_diff:.0%} improvement). "
                    f"Attorney representation is one of the strongest predictors of ZBA success."
                ),
                "estimated_impact": "high",
            })
        elif shap_lookup.get('has_attorney', 0) < -0.01:
            recs.append({
                "action": "Hire a zoning attorney",
                "detail": (
                    "The model identifies lack of attorney representation as reducing "
                    "your approval odds. Experienced zoning attorneys know how to frame "
                    "applications for Board approval."
                ),
                "estimated_impact": "medium",
            })

    # ---------------------------------------------------------------
    # 2. Reduce variances — identify the weakest one
    # ---------------------------------------------------------------
    if len(variances) > 2 and per_variance:
        # Find the variance with the lowest approval rate
        worst_var = None
        worst_rate = 1.0
        for v_name, v_data in per_variance.items():
            rate = v_data.get('approval_rate', 1.0)
            if rate < worst_rate:
                worst_rate = rate
                worst_var = v_name
        if worst_var and worst_rate < overall_rate - 0.03:
            recs.append({
                "action": f"Eliminate the {worst_var} variance",
                "detail": (
                    f"Consider redesigning to eliminate the {worst_var} variance — it has "
                    f"the lowest approval rate ({worst_rate:.0%}) of your requested variances. "
                    f"Reducing from {len(variances)} to {len(variances) - 1} variances also "
                    f"lowers overall project complexity."
                ),
                "estimated_impact": "high",
            })

    # ---------------------------------------------------------------
    # 3. Scale down — units
    # ---------------------------------------------------------------
    units_shap = shap_lookup.get('proposed_units', 0)
    if proposed_units and proposed_units > 3 and units_shap < -0.005:
        suggested = max(1, proposed_units - 2)
        recs.append({
            "action": "Reduce unit count",
            "detail": (
                f"Reducing from {proposed_units} to {suggested} units could improve odds — "
                f"the model shows larger projects face more scrutiny. The number of proposed "
                f"units is currently working against your approval probability."
            ),
            "estimated_impact": "high" if units_shap < -0.02 else "medium",
        })

    # ---------------------------------------------------------------
    # 4. Height variance
    # ---------------------------------------------------------------
    height_shap = shap_lookup.get('var_height', 0)
    if 'height' in [v.lower() for v in variances] and height_shap < -0.005:
        recs.append({
            "action": "Reduce building height",
            "detail": (
                "Consider reducing height to comply with the zoning limit — this "
                "eliminates one variance and removes a negative factor from your "
                "application. Height variances draw extra Board attention."
            ),
            "estimated_impact": "high" if height_shap < -0.02 else "medium",
        })

    # ---------------------------------------------------------------
    # 5. Parking variance
    # ---------------------------------------------------------------
    if 'parking' in [v.lower() for v in variances]:
        parking_rate = per_variance.get('parking', {}).get('approval_rate')
        parking_shap = shap_lookup.get('var_parking', 0)
        detail = (
            "Adding more parking spaces to meet the zoning requirement would "
            "eliminate the parking variance entirely. "
        )
        if parking_rate is not None:
            detail += f"Parking variances have a {parking_rate:.0%} approval rate historically."
        recs.append({
            "action": "Add parking to eliminate variance",
            "detail": detail,
            "estimated_impact": "medium" if parking_shap >= -0.01 else "high",
        })

    # ---------------------------------------------------------------
    # 6. Project type — new construction penalty
    # ---------------------------------------------------------------
    nc_shap = shap_lookup.get('proj_new_construction', 0)
    if project_type and 'new_construction' in project_type.lower().replace(' ', '_') and nc_shap < -0.005:
        recs.append({
            "action": "Consider renovation over new construction",
            "detail": (
                "Renovations and additions have higher approval rates than new construction. "
                "If possible, framing the project as a renovation or adaptive reuse could "
                "improve your odds with the Board."
            ),
            "estimated_impact": "medium",
        })

    # ---------------------------------------------------------------
    # 7. Scale down — stories (only if not already suggesting unit reduction)
    # ---------------------------------------------------------------
    stories_shap = shap_lookup.get('proposed_stories', 0)
    if (proposed_stories and proposed_stories > 3 and stories_shap < -0.005
            and not any(r['action'] == 'Reduce unit count' for r in recs)):
        recs.append({
            "action": "Reduce building height/stories",
            "detail": (
                f"Reducing from {proposed_stories} stories could improve approval odds. "
                f"The model identifies building scale as a negative factor for your proposal."
            ),
            "estimated_impact": "medium",
        })

    # Sort: high > medium > low
    impact_order = {"high": 0, "medium": 1, "low": 2}
    recs.sort(key=lambda r: impact_order.get(r["estimated_impact"], 9))

    # Cap at 4 recommendations
    return recs[:4]


@app.post("/analyze_proposal", tags=["Prediction"], dependencies=[Depends(verify_api_key)])
def analyze_proposal(payload: dict):
    """
    Analyze a development proposal and predict ZBA approval likelihood.

    Returns:
    - ML model probability (if model loaded)
    - Historical variance analysis (real approval rates for your exact combination)
    - SHAP-powered feature drivers with human-readable labels
    - Actionable recommendations to improve approval odds (up to 4)
    - Similar cases (mix of approved + denied for contrast)
    - Data-driven key factors (not hardcoded)

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

    # Get similar cases (stratified: includes denied cases for contrast)
    similar, total_similar, approval_rate_similar = get_similar_cases(ward, variances, project_type)

    # Get REAL historical variance data — THE answer to the money question
    variance_history = _get_variance_history(variances, ward, has_attorney)

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

                # Features where "increases approval odds" sounds counterintuitive
                _violation_prefixes = ('excessive_', 'insufficient_')
                _violation_label_keywords = ('exceeds', 'insufficient', 'smaller than', 'below minimum', 'fewer')

                contributions = []
                for col, shap_val in zip(feature_cols, shap_row):
                    input_val = float(input_df[col].iloc[0])
                    # Use human-readable label instead of cryptic feature name
                    label = FEATURE_LABELS.get(col, col.replace("_", " ").title())

                    # Determine direction text
                    is_violation_feature = (
                        any(col.startswith(p) for p in _violation_prefixes) or
                        any(kw in label.lower() for kw in _violation_label_keywords)
                    )
                    if is_violation_feature:
                        direction = "common in approved cases" if shap_val > 0 else "risk factor — watch this one"
                    else:
                        direction = "increases approval odds" if shap_val > 0 else "decreases approval odds"

                    contributions.append({
                        "feature": label,
                        "feature_name": col,  # Keep raw name for debugging
                        "shap_value": round(float(shap_val), 4),
                        "input_value": round(input_val, 3) if abs(input_val) > 1 else int(input_val),
                        "direction": direction,
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
                            label = FEATURE_LABELS.get(col, col.replace("_", " ").title())
                            contributions.append({
                                "feature": label,
                                "feature_name": col,
                                "shap_value": round(float(imp), 4),
                                "input_value": round(val, 3) if abs(val) > 1 else int(val),
                                "direction": "important factor",
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

            # Key factors — computed from REAL historical data, not hardcoded
            key_factors = _build_key_factors(
                variances, ward, has_attorney, project_type, proposed_units, variance_history
            )

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
                "recommendations": _build_recommendations(
                    prob, variances, ward, has_attorney, project_type,
                    proposed_units, proposed_stories, variance_history, top_drivers
                ),
                "similar_cases": similar,
                "variance_history": variance_history,
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

    # Even without ML model, use real historical data for key factors
    key_factors_heuristic = _build_key_factors(
        variances, ward, has_attorney, project_type, proposed_units, variance_history
    )
    key_factors_heuristic.insert(0, f"Note: Using historical statistics ({len(df_decided):,} cases) — ML model not loaded")

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
        "recommendations": _build_recommendations(
            score, variances, ward, has_attorney, project_type,
            proposed_units, proposed_stories, variance_history, []
        ),
        "similar_cases": similar,
        "variance_history": variance_history,
        "estimated_timeline_days": _estimate_timeline(ward),
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
            scenario_name = "Without attorney"
            scenario_desc = f"Removing attorney representation would change probability by {diff:+.1%}"
        else:
            scenario_name = "With attorney"
            scenario_desc = f"Adding attorney representation would change probability by {diff:+.1%}"
        scenarios.append({
            "scenario": scenario_name,
            "name": scenario_name,
            "probability": round(alt_prob, 3),
            "difference": round(diff, 3),
            "delta": round(diff, 3),
            "description": scenario_desc,
        })

        # --- Fewer variances (remove one at a time) ---
        if len(variances) > 1:
            fewer = variances[:max(1, len(variances) - 1)]
            fewer_features = build_features(parcel_row, proposed_use, fewer, project_type,
                                            ward, has_attorney, proposed_units, proposed_stories)
            fewer_prob = predict_prob(fewer_features)
            diff = fewer_prob - base_prob
            scenario_name = f"With {len(fewer)} variance(s) instead of {len(variances)}"
            scenarios.append({
                "scenario": scenario_name,
                "name": scenario_name,
                "probability": round(fewer_prob, 3),
                "difference": round(diff, 3),
                "delta": round(diff, 3),
                "description": f"Reducing to {len(fewer)} variance(s) would change probability by {diff:+.1%}"
            })

        # --- Single variance only (if requesting 3+) ---
        if len(variances) >= 3:
            single = variances[:1]
            single_features = build_features(parcel_row, proposed_use, single, project_type,
                                             ward, has_attorney, proposed_units, proposed_stories)
            single_prob = predict_prob(single_features)
            diff = single_prob - base_prob
            scenario_name = f"With only 1 variance ({single[0]})"
            scenarios.append({
                "scenario": scenario_name,
                "name": scenario_name,
                "probability": round(single_prob, 3),
                "difference": round(diff, 3),
                "delta": round(diff, 3),
                "description": f"Requesting only 1 variance would change probability by {diff:+.1%}"
            })

        # --- Different project type: addition (usually higher approval) ---
        if project_type and 'addition' not in (project_type or '').lower():
            add_features = build_features(parcel_row, proposed_use, variances, 'addition',
                                          ward, has_attorney, proposed_units, proposed_stories)
            add_prob = predict_prob(add_features)
            diff = add_prob - base_prob
            if abs(diff) > 0.001:
                scenario_name = "As an Addition/Extension project"
                scenarios.append({
                    "scenario": scenario_name,
                    "name": scenario_name,
                    "probability": round(add_prob, 3),
                    "difference": round(diff, 3),
                    "delta": round(diff, 3),
                    "description": f"If framed as an addition/extension: probability changes by {diff:+.1%}"
                })

        # --- Fewer units (if > 1) ---
        if proposed_units > 1:
            fewer_units = max(1, proposed_units - 1)
            fewer_u_features = build_features(parcel_row, proposed_use, variances, project_type,
                                              ward, has_attorney, fewer_units, proposed_stories)
            fewer_u_prob = predict_prob(fewer_u_features)
            diff = fewer_u_prob - base_prob
            if abs(diff) > 0.001:
                scenario_name = f"With {fewer_units} unit(s) instead of {proposed_units}"
                scenarios.append({
                    "scenario": scenario_name, "name": scenario_name,
                    "probability": round(fewer_u_prob, 3),
                    "difference": round(diff, 3), "delta": round(diff, 3),
                    "description": f"Reducing to {fewer_units} unit(s) would change probability by {diff:+.1%}"
                })

        # --- Fewer stories (if > 2) ---
        if proposed_stories > 2:
            fewer_s = proposed_stories - 1
            fewer_s_features = build_features(parcel_row, proposed_use, variances, project_type,
                                              ward, has_attorney, proposed_units, fewer_s)
            fewer_s_prob = predict_prob(fewer_s_features)
            diff = fewer_s_prob - base_prob
            if abs(diff) > 0.001:
                scenario_name = f"With {fewer_s} stories instead of {proposed_stories}"
                scenarios.append({
                    "scenario": scenario_name, "name": scenario_name,
                    "probability": round(fewer_s_prob, 3),
                    "difference": round(diff, 3), "delta": round(diff, 3),
                    "description": f"Reducing to {fewer_s} stories would change probability by {diff:+.1%}"
                })

        # --- Best-case scenario: attorney + fewer variances ---
        best_attorney = True
        best_variances = variances[:1] if len(variances) > 1 else variances
        best_features = build_features(parcel_row, proposed_use, best_variances, project_type,
                                       ward, best_attorney, proposed_units, proposed_stories)
        best_prob = predict_prob(best_features)
        diff = best_prob - base_prob
        if abs(diff) > 0.001:
            scenario_name = "Best case (attorney + minimal variances)"
            scenarios.append({
                "scenario": scenario_name,
                "name": scenario_name,
                "probability": round(best_prob, 3),
                "difference": round(diff, 3),
                "delta": round(diff, 3),
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

# --- TTL cache for /recommend (expensive ML batch predictions) ---
_recommend_cache = {}
_RECOMMEND_CACHE_TTL = 300  # 5 minutes


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
    import time as _time

    if model_package is None or gdf is None:
        raise HTTPException(status_code=503, detail="Model or parcel data not loaded")

    model = model_package.get('model')
    feature_cols = model_package.get('feature_cols', [])
    if model is None or not feature_cols:
        raise HTTPException(status_code=503, detail="Model not available")

    variance_list = [v.strip() for v in variances.split(",") if v.strip()] if variances else []

    # Check TTL cache
    cache_key = (use_type, project_type, tuple(variance_list), min_approval_rate, ward or "")
    if cache_key in _recommend_cache:
        cached_result, cached_time = _recommend_cache[cache_key]
        if _time.time() - cached_time < _RECOMMEND_CACHE_TTL:
            # Return cached result, but apply limit
            result = dict(cached_result)
            result["parcels"] = result["parcels"][:limit]
            result["results_found"] = len(result["parcels"])
            return result

    # Get candidate parcels — optionally filter by ward
    candidates = gdf.copy()
    if ward:
        # Filter parcels by ward if we can match
        if zba_df is not None and 'ward' in zba_df.columns:
            ward_parcels = zba_df[zba_df['ward'] == float(ward)]['pa_parcel_id'].dropna().unique()
            ward_parcel_strs = set(str(int(p)).zfill(10) for p in ward_parcels if not np.isnan(p))
            candidates = candidates[candidates.index.isin(ward_parcel_strs)]

    # Sample if too many candidates (reduced from 2000 to 500 for speed)
    if len(candidates) > 500:
        candidates = candidates.sample(500, random_state=42)

    # Build all feature vectors at once for batch prediction (10x faster)
    features_list = []
    valid_indices = []
    for parcel_id in candidates.index:
        try:
            parcel_row = candidates.loc[parcel_id]
            if isinstance(parcel_row, pd.DataFrame):
                parcel_row = parcel_row.iloc[0]
            features = build_features(
                parcel_row=parcel_row,
                proposed_use=use_type,
                variances=variance_list,
                has_attorney=True,
                ward=ward or str(parcel_row.get('ward', '')),
                proposed_units=0,
                proposed_stories=0,
                project_type=project_type,
            )
            features_list.append(features)
            valid_indices.append(parcel_id)
        except Exception:
            continue

    if not features_list:
        return {"query": {"use_type": use_type, "project_type": project_type, "variances": variance_list, "ward": ward, "min_approval_rate": min_approval_rate}, "total_candidates": len(candidates), "results_found": 0, "parcels": []}

    # Batch prediction — single model call instead of N calls
    features_df = pd.DataFrame(features_list)[feature_cols].fillna(0)
    probs = model.predict_proba(features_df)[:, 1]

    results = []
    for i, (parcel_id, prob) in enumerate(zip(valid_indices, probs)):
        prob = float(prob)
        if prob >= min_approval_rate:
            parcel_row = candidates.loc[parcel_id]
            if isinstance(parcel_row, pd.DataFrame):
                parcel_row = parcel_row.iloc[0]
            result = {
                "parcel_id": str(parcel_id),
                "approval_probability": round(prob, 3),
                "zoning_code": str(parcel_row.get("primary_zoning") or ""),
                "district": str(parcel_row.get("districts") or ""),
            }
            if hasattr(parcel_row, 'geometry') and parcel_row.geometry:
                centroid = parcel_row.geometry.centroid
                result["lat"] = round(centroid.y, 6)
                result["lon"] = round(centroid.x, 6)
            results.append(result)

    results.sort(key=lambda x: -x['approval_probability'])

    # Cache the full result set (before applying limit) for reuse
    full_response = {
        "query": {
            "use_type": use_type,
            "project_type": project_type,
            "variances": variance_list,
            "ward": ward,
            "min_approval_rate": min_approval_rate,
        },
        "total_candidates": len(candidates),
        "results_found": len(results[:limit]),
        "parcels": results,
        "disclaimer": "Probabilities are model estimates based on historical ZBA decisions. "
                      "Actual outcomes depend on many factors not captured in the model. "
                      "This is a risk assessment tool, not legal advice.",
    }
    _recommend_cache[cache_key] = (full_response, _time.time())

    # Apply limit for this response
    full_response_copy = dict(full_response)
    full_response_copy["parcels"] = results[:limit]
    full_response_copy["results_found"] = len(full_response_copy["parcels"])
    return full_response_copy


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
