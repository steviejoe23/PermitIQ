"""
Parcel Risk Score Routes — Pre-computed development difficulty scores.

Every parcel in Boston gets a 0-100 risk score based on ward denial rates,
zoning district patterns, case density, and lot characteristics.
"""

import logging
import os
import pandas as pd
from fastapi import APIRouter, HTTPException
from api import state

logger = logging.getLogger("permitiq")
router = APIRouter(prefix="/risk", tags=["Risk Score"])

_risk_df = None
DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data', 'parcel_risk_scores.csv')


def init():
    global _risk_df
    path = DATA_PATH
    if not os.path.exists(path):
        path = os.path.join('data', 'parcel_risk_scores.csv')
    if os.path.exists(path):
        _risk_df = pd.read_csv(path, low_memory=False)
        _risk_df['parcel_id'] = _risk_df['parcel_id'].astype(str).str.zfill(10)
        logger.info("Parcel risk scores loaded (%d parcels)", len(_risk_df))
    else:
        logger.warning("Parcel risk scores not found at %s", path)


@router.get("/parcels/{parcel_id}")
def parcel_risk_score(parcel_id: str):
    """Get the development difficulty score for a specific parcel."""
    if _risk_df is None:
        init()
    if _risk_df is None or _risk_df.empty:
        raise HTTPException(status_code=500, detail="Risk scores not loaded")

    pid = str(parcel_id).zfill(10)
    row = _risk_df[_risk_df['parcel_id'] == pid]

    if row.empty:
        raise HTTPException(status_code=404, detail=f"No risk score for parcel {parcel_id}")

    r = row.iloc[0]
    score = float(r['risk_score'])

    if score <= 25:
        level = "Easy"
        description = "Permissive zoning with low historical denial rates. Standard projects likely to pass."
    elif score <= 50:
        level = "Moderate"
        description = "Some zoning constraints. Variance requests face average scrutiny."
    elif score <= 75:
        level = "Difficult"
        description = "Restrictive zoning district with above-average denial rates. Legal counsel recommended."
    else:
        level = "Very Difficult"
        description = "High denial rate area with significant zoning constraints. Expect strong scrutiny."

    return {
        "parcel_id": parcel_id,
        "risk_score": round(score, 1),
        "risk_level": level,
        "description": description,
        "components": {
            "ward_denial_rate": round(float(r.get('ward_denial_rate', 0)), 3),
            "district_denial_rate": round(float(r.get('district_denial_rate', 0)), 3),
            "case_density_score": round(float(r.get('case_density_score', 0)), 1),
            "lot_size_score": round(float(r.get('lot_size_score', 0)), 1),
            "zoning_restrictiveness": round(float(r.get('zoning_restrictiveness', 0)), 1),
        },
    }


@router.get("/summary")
def risk_summary():
    """Summary statistics about risk scores across all parcels."""
    if _risk_df is None:
        init()
    if _risk_df is None or _risk_df.empty:
        return {"error": "Risk scores not loaded"}

    scores = _risk_df['risk_score']
    levels = _risk_df['risk_level'].value_counts().to_dict() if 'risk_level' in _risk_df.columns else {}

    return {
        "total_parcels": len(_risk_df),
        "mean_score": round(float(scores.mean()), 1),
        "median_score": round(float(scores.median()), 1),
        "min_score": round(float(scores.min()), 1),
        "max_score": round(float(scores.max()), 1),
        "distribution": {
            "easy": int(levels.get('Easy', 0)),
            "moderate": int(levels.get('Moderate', 0)),
            "difficult": int(levels.get('Difficult', 0)),
            "very_difficult": int(levels.get('Very Difficult', 0)),
        },
    }
