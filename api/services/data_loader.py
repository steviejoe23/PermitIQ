"""
Data loading and startup initialization for PermitIQ API.

Populates the shared state module (api.state) with all data needed by routers.
"""

import os
import logging
import geopandas as gpd
import pandas as pd
import numpy as np
import joblib

# model_classes must be importable as top-level module for pickle deserialization
import sys
_services_dir = os.path.dirname(os.path.abspath(__file__))
if _services_dir not in sys.path:
    sys.path.insert(0, _services_dir)
try:
    from model_classes import StackingEnsemble, ManualCalibratedModel
except ImportError:
    StackingEnsemble = ManualCalibratedModel = None

from api import state
from api.utils import normalize_address, safe_str, _clean_case_date

logger = logging.getLogger("permitiq")

BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
GEOJSON_PATH = os.path.join(BASE_DIR, "../boston_parcels_zoning.geojson")
ZBA_DATA_PATH = os.path.join(BASE_DIR, "../zba_cases_cleaned.csv")
MODEL_PATH = os.path.join(BASE_DIR, "zba_model.pkl")
PROPERTY_PATH = os.path.join(BASE_DIR, "../property_assessment_fy2026.csv")
TRACKER_PATH = os.path.join(BASE_DIR, "../zba_tracker.csv")


def _precompute_timeline_stats(tracker_path: str) -> dict:
    """Pre-compute timeline statistics from ZBA tracker CSV."""
    try:
        tk = pd.read_csv(tracker_path, low_memory=False)
    except Exception as e:
        logger.error("Failed to load tracker CSV for timeline stats: %s", e)
        return None

    for col in ['submitted_date', 'hearing_date', 'final_decision_date', 'closed_date']:
        tk[col] = pd.to_datetime(tk[col], errors='coerce')

    tk['filing_to_hearing'] = (tk['hearing_date'] - tk['submitted_date']).dt.days
    tk['filing_to_decision'] = (tk['final_decision_date'] - tk['submitted_date']).dt.days
    tk['hearing_to_decision'] = (tk['final_decision_date'] - tk['hearing_date']).dt.days
    tk['filing_to_closed'] = (tk['closed_date'] - tk['submitted_date']).dt.days

    phases = ['filing_to_hearing', 'filing_to_decision', 'hearing_to_decision', 'filing_to_closed']

    def _phase_stats(df_subset, phase_col):
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
        result = {}
        for phase in phases:
            stats = _phase_stats(df_subset, phase)
            if stats:
                result[phase] = stats
        return result

    stats = {"overall": _all_phases(tk), "by_ward": {}, "by_appeal_type": {}}

    if 'ward' in tk.columns:
        for ward_val, group in tk.groupby('ward'):
            ward_str = str(int(ward_val)) if pd.notna(ward_val) else None
            if ward_str and len(group) >= 10:
                ward_stats = _all_phases(group)
                if ward_stats:
                    stats["by_ward"][ward_str] = ward_stats

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


def _build_case_coords():
    """Match ZBA case addresses to parcel centroids for geographic search."""
    if state.gdf is None or state.zba_df is None or state.parcel_addr_df is None:
        return

    logger.info("Building case coordinate index for geographic search...")

    centroids = state.gdf.geometry.centroid
    centroid_df = pd.DataFrame({
        'parcel_id': state.gdf.index.values,
        'lat': centroids.y.values,
        'lon': centroids.x.values,
    })

    addr_to_parcel = dict(zip(state.parcel_addr_df['_addr_norm'], state.parcel_addr_df['parcel_id']))

    zba_with_addr = state.zba_df[
        state.zba_df['address_clean'].notna() &
        state.zba_df['decision_clean'].notna() &
        state.zba_df['address_clean'].str.match(r'^\d', na=False)
    ].drop_duplicates('case_number').copy()

    if '_addr_norm' not in zba_with_addr.columns:
        zba_with_addr['_addr_norm'] = zba_with_addr['address_clean'].apply(normalize_address)

    zba_with_addr['_matched_pid'] = zba_with_addr['_addr_norm'].map(addr_to_parcel)

    merged = zba_with_addr.merge(
        centroid_df, left_on='_matched_pid', right_on='parcel_id', how='left'
    )

    geocoded = merged[merged['lat'].notna()].copy()

    def _clean_ward(w):
        try:
            if pd.notna(w) and str(w).replace('.', '', 1).isdigit():
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

    state._case_coords = pd.DataFrame(records)
    total_cases = len(zba_with_addr)
    logger.info("Case coordinate index built: %d of %d cases geocoded (%.0f%%)",
                len(state._case_coords), total_cases, 100 * len(state._case_coords) / max(total_cases, 1))


def _log_memory(label: str):
    """Log current RSS memory usage."""
    try:
        # Linux: read from /proc (most reliable)
        with open('/proc/self/status') as f:
            for line in f:
                if line.startswith('VmRSS:'):
                    rss_kb = int(line.split()[1])
                    logger.info("Memory [%s]: %.0f MB RSS", label, rss_kb / 1024)
                    return
    except Exception:
        pass
    try:
        import resource, platform
        maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # macOS returns bytes, Linux returns KB
        if platform.system() == 'Darwin':
            rss_mb = maxrss / (1024 * 1024)
        else:
            rss_mb = maxrss / 1024
        logger.info("Memory [%s]: %.0f MB RSS", label, rss_mb)
    except Exception:
        pass


def load_all(market_init=None, attorney_init=None, variance_types=None, project_types=None):
    """Load all data into state module. Called once at startup."""
    import gc
    light_mode_val = os.environ.get('PERMITIQ_LIGHT_MODE', '')
    light_mode = light_mode_val.lower() in ('1', 'true', 'yes')
    logger.info("PERMITIQ_LIGHT_MODE=%r, light_mode=%s", light_mode_val, light_mode)
    if light_mode:
        logger.info("LIGHT MODE enabled — skipping GeoJSON and property assessment to conserve memory")

    _log_memory("baseline")

    if not light_mode:
        try:
            state.gdf = gpd.read_file(GEOJSON_PATH)
            state.gdf["parcel_id"] = state.gdf["parcel_id"].astype(str)
            state.gdf = state.gdf.set_index("parcel_id", drop=False)
            logger.info("GeoJSON loaded (%d parcels)", len(state.gdf))
        except Exception as e:
            logger.error("Failed to load GeoJSON: %s", e)
        gc.collect()
        _log_memory("after geojson")

    try:
        state.zba_df = pd.read_csv(ZBA_DATA_PATH, low_memory=False)
        if light_mode and 'raw_text' in state.zba_df.columns:
            state.zba_df = state.zba_df.drop(columns=['raw_text'])
            logger.info("Dropped raw_text column to save memory")
        if 'address_clean' in state.zba_df.columns:
            state.zba_df['_addr_norm'] = state.zba_df['address_clean'].apply(normalize_address)
        logger.info("ZBA dataset loaded (%d cases, %d cols)", len(state.zba_df), len(state.zba_df.columns))
    except Exception as e:
        logger.error("Failed to load ZBA dataset: %s", e)
    gc.collect()
    _log_memory("after zba")

    try:
        state.model_package = joblib.load(MODEL_PATH)
        model_name = state.model_package.get('model_name', 'unknown')
        auc = state.model_package.get('auc_score', 0)
        n_features = len(state.model_package.get('feature_cols', []))
        logger.info("ML model loaded (%s, AUC: %.4f, %d features)", model_name, auc, n_features)
    except Exception as e:
        logger.warning("No trained model found, using fallback logic: %s", e)
    gc.collect()
    _log_memory("after model")

    if not light_mode:
        try:
            _pa = pd.read_csv(PROPERTY_PATH, usecols=['PID', 'ST_NUM', 'ST_NAME', 'LAND_SF'], low_memory=False)
            _pa = _pa.dropna(subset=['PID', 'ST_NUM', 'ST_NAME'])
            _pa['ST_NUM'] = _pa['ST_NUM'].astype(str).str.strip().str.replace(r'\.0$', '', regex=True)
            _pa['ST_NAME'] = _pa['ST_NAME'].astype(str).str.strip()
            _pa['address'] = _pa['ST_NUM'] + ' ' + _pa['ST_NAME']
            _pa['_addr_norm'] = _pa['address'].apply(normalize_address)
            _pa['parcel_id'] = _pa['PID'].astype(str).str.zfill(10)
            _pa['lot_size'] = pd.to_numeric(_pa['LAND_SF'].astype(str).str.replace(',', ''), errors='coerce').fillna(0)
            state.parcel_addr_df = _pa[['parcel_id', 'address', '_addr_norm', 'lot_size']].drop_duplicates('parcel_id')
            logger.info("Property address lookup loaded (%d parcels)", len(state.parcel_addr_df))
        except Exception as e:
            logger.warning("Could not load property assessment for geocoding: %s", e)
        gc.collect()
        _log_memory("after property")

    if not light_mode:
        state.timeline_stats = _precompute_timeline_stats(TRACKER_PATH)

    if market_init is not None and state.zba_df is not None:
        market_init(state.zba_df, variance_types, project_types, timeline_stats=state.timeline_stats)
        logger.info("Market intel router initialized with %d cases", len(state.zba_df))

    if attorney_init is not None and state.zba_df is not None:
        attorney_init(state.zba_df, variance_types)
        logger.info("Attorney router initialized")

    if not light_mode:
        _build_case_coords()

    _log_memory("startup complete")
    logger.info("Startup complete — light_mode=%s", light_mode)
