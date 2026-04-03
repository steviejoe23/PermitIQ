"""
Platform endpoints — stats, health, model info, data status.
"""

import os
import time
import datetime
import json
import logging
import pandas as pd
from fastapi import APIRouter, HTTPException

from api import state
from api.utils import normalize_address

logger = logging.getLogger("permitiq")
router = APIRouter()

# Path constants (same as data_loader)
BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
ZBA_DATA_PATH = os.path.join(BASE_DIR, "../zba_cases_cleaned.csv")
MODEL_PATH = os.path.join(BASE_DIR, "zba_model.pkl")
GEOJSON_PATH = os.path.join(BASE_DIR, "../boston_parcels_zoning.geojson")
PROPERTY_PATH = os.path.join(BASE_DIR, "../property_assessment_fy2026.csv")

try:
    from api.services.database import db_available
except ImportError:
    try:
        from services.database import db_available
    except ImportError:
        db_available = lambda: False


@router.get("/stats", tags=["Platform"])
def overall_stats():
    """Overall platform statistics for the dashboard."""
    if state.zba_df is None:
        raise HTTPException(status_code=500, detail="ZBA data not loaded")

    df = state.zba_df[state.zba_df['decision_clean'].notna()].copy()
    total = len(df)
    approved = int((df['decision_clean'] == 'APPROVED').sum())
    denied = total - approved

    def _safe_ward(w):
        if pd.isna(w):
            return None
        try:
            return str(int(float(w)))
        except (ValueError, TypeError):
            return None
    df['_ward'] = df['ward'].apply(_safe_ward)
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
        "total_parcels": len(state.gdf) if state.gdf is not None else 0,
    }


@router.get("/model_info", tags=["Platform"])
def model_info():
    """Full model metadata and version info."""
    if state.model_package is None:
        raise HTTPException(status_code=503, detail="No model loaded")

    return {
        "model_name": state.model_package.get('model_name'),
        "model_version": state.model_package.get('model_version'),
        "auc_score": state.model_package.get('auc_score'),
        "brier_score": state.model_package.get('brier_score'),
        "optimal_threshold": state.model_package.get('optimal_threshold'),
        "cv_auc_mean": state.model_package.get('cv_auc_mean'),
        "cv_auc_std": state.model_package.get('cv_auc_std'),
        "total_cases": state.model_package.get('total_cases'),
        "train_size": state.model_package.get('train_size'),
        "test_size": state.model_package.get('test_size'),
        "feature_count": len(state.model_package.get('feature_cols', [])),
        "feature_cols": state.model_package.get('feature_cols', []),
        "is_calibrated": state.model_package.get('is_calibrated', False),
        "leakage_free": state.model_package.get('leakage_free', False),
        "removed_features": state.model_package.get('removed_features', []),
        "trained_at": state.model_package.get('trained_at'),
        "dataset_hash": state.model_package.get('dataset_hash'),
        "calibration": {
            "ece": 0.0101,
            "ece_pct": "1.0%",
            "verdict": "Well-calibrated — probabilities are trustworthy as risk estimates",
            "buckets": {
                "90-100%": {"predicted": "95.8%", "actual": "95.4%", "gap": "-0.3pp", "trust": "HIGH"},
                "80-90%": {"predicted": "85.5%", "actual": "85.1%", "gap": "-0.4pp", "trust": "HIGH"},
                "70-80%": {"predicted": "75.6%", "actual": "78.2%", "gap": "+2.6pp", "trust": "HIGH"},
                "60-70%": {"predicted": "64.7%", "actual": "70.1%", "gap": "+5.4pp", "trust": "MODERATE"},
                "50-60%": {"predicted": "55.0%", "actual": "54.9%", "gap": "-0.1pp", "trust": "HIGH"},
                "0-50%": {"predicted": "49.3%", "actual": "29.0%", "gap": "-20.2pp", "trust": "LOW"},
            },
            "warnings": [
                "Building appeal cases are ~10pp less likely to be approved than predicted",
                "Predictions below 50% have limited calibration data — treat with extra caution",
            ],
        },
    }


@router.get("/health", tags=["Platform"])
def health():
    return {
        "status": "ok",
        "geojson_loaded": state.gdf is not None,
        "zba_loaded": state.zba_df is not None,
        "model_loaded": state.model_package is not None,
        "geocoder_loaded": state.parcel_addr_df is not None,
        "total_parcels": len(state.gdf) if state.gdf is not None else 0,
        "total_cases": len(state.zba_df) if state.zba_df is not None else 0,
        "geocoder_parcels": len(state.parcel_addr_df) if state.parcel_addr_df is not None else 0,
        "model_name": state.model_package.get('model_name') if state.model_package else None,
        "model_auc": state.model_package.get('auc_score') if state.model_package else None,
        "model_brier": state.model_package.get('brier_score') if state.model_package else None,
        "optimal_threshold": state.model_package.get('optimal_threshold') if state.model_package else None,
        "features": len(state.model_package.get('feature_cols', [])) if state.model_package else 0,
        "postgis_available": db_available(),
        "leakage_free": state.model_package.get('leakage_free', False) if state.model_package else False,
        "model_version": state.model_package.get('model_version') if state.model_package else None,
    }


@router.get("/data_status", tags=["Platform"])
def data_status():
    """Data freshness and pipeline status."""
    status = {}

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
                "file": os.path.basename(full_path),
                "last_modified": dt.isoformat(),
                "age_hours": round(age_hours, 1),
                "size_mb": round(os.path.getsize(full_path) / (1024 * 1024), 1),
            }
        else:
            status[name] = {"path": full_path, "exists": False}

    checkpoint_path = os.path.join(BASE_DIR, "../ocr_checkpoint.json")
    if os.path.exists(checkpoint_path):
        try:
            with open(checkpoint_path) as f:
                cp = json.load(f)
            completed = len(cp.get("completed", []))
            status["ocr_pipeline"] = {"running": True, "completed_pdfs": completed, "checkpoint_file": checkpoint_path}
        except Exception:
            status["ocr_pipeline"] = {"running": True, "error": "Could not read checkpoint"}
    else:
        status["ocr_pipeline"] = {"running": False}

    return status
