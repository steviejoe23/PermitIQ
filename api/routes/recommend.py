"""
Site selection / recommendation endpoint.
"""

import time as _time
import logging
import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException

from api import state
from api.routes.prediction import build_features

logger = logging.getLogger("permitiq")
router = APIRouter()

# TTL cache for /recommend (expensive ML batch predictions)
_recommend_cache = {}
_RECOMMEND_CACHE_TTL = 300  # 5 minutes


@router.get("/recommend", tags=["Recommendation"])
def recommend_parcels(
    use_type: str = "residential",
    project_type: str = "new_construction",
    variances: str = "",
    min_approval_rate: float = 0.5,
    ward: str = None,
    limit: int = 20,
):
    """Recommend the best parcels in Boston for a given project type."""
    if state.model_package is None or state.gdf is None:
        raise HTTPException(status_code=503, detail="Model or parcel data not loaded")

    model = state.model_package.get('model')
    feature_cols = state.model_package.get('feature_cols', [])
    if model is None or not feature_cols:
        raise HTTPException(status_code=503, detail="Model not available")

    variance_list = [v.strip() for v in variances.split(",") if v.strip()] if variances else []

    cache_key = (use_type, project_type, tuple(variance_list), min_approval_rate, ward or "")
    if cache_key in _recommend_cache:
        cached_result, cached_time = _recommend_cache[cache_key]
        if _time.time() - cached_time < _RECOMMEND_CACHE_TTL:
            result = dict(cached_result)
            result["parcels"] = result["parcels"][:limit]
            result["results_found"] = len(result["parcels"])
            return result

    candidates = state.gdf.copy()
    if ward:
        try:
            ward_float = float(ward)
        except (ValueError, TypeError):
            raise HTTPException(status_code=400, detail=f"Invalid ward: {ward}")
        if state.zba_df is not None and 'ward' in state.zba_df.columns:
            ward_parcels = state.zba_df[state.zba_df['ward'] == ward_float]['pa_parcel_id'].dropna().unique()
            ward_parcel_strs = set(str(int(p)).zfill(10) for p in ward_parcels if not np.isnan(p))
            candidates = candidates[candidates.index.isin(ward_parcel_strs)]

    if len(candidates) > 500:
        candidates = candidates.sample(500, random_state=42)

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
        except Exception as e:
            logger.debug("Failed to build features for parcel %s: %s", parcel_id, e)
            continue

    if not features_list:
        return {"query": {"use_type": use_type, "project_type": project_type, "variances": variance_list, "ward": ward, "min_approval_rate": min_approval_rate}, "total_candidates": len(candidates), "results_found": 0, "parcels": []}

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

    full_response = {
        "query": {"use_type": use_type, "project_type": project_type, "variances": variance_list, "ward": ward, "min_approval_rate": min_approval_rate},
        "total_candidates": len(candidates),
        "results_found": len(results[:limit]),
        "parcels": results,
        "disclaimer": "Probabilities are model estimates based on historical ZBA decisions. This is a risk assessment tool, not legal advice.",
    }
    _recommend_cache[cache_key] = (full_response, _time.time())

    full_response_copy = dict(full_response)
    full_response_copy["parcels"] = results[:limit]
    full_response_copy["results_found"] = len(full_response_copy["parcels"])
    return full_response_copy
