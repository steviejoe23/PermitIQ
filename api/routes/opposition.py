"""
Opposition Risk Routes — Neighborhood opposition intensity analysis.

Scores each neighborhood's opposition level by project/variance type,
extracted from hearing transcript sentiment data.
"""

import json
import logging
import os
from fastapi import APIRouter, HTTPException, Query
from typing import Optional

logger = logging.getLogger("permitiq")
router = APIRouter(prefix="/opposition", tags=["Opposition Risk"])

_index = None
DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data', 'opposition_index.json')


def init():
    global _index
    path = DATA_PATH
    if not os.path.exists(path):
        path = os.path.join('data', 'opposition_index.json')
    if os.path.exists(path):
        with open(path) as f:
            _index = json.load(f)
        logger.info("Opposition index loaded (%d neighborhoods)", len(_index.get('neighborhoods', {})))
    else:
        logger.warning("Opposition index not found at %s", path)
        _index = {"neighborhoods": {}}


@router.get("/by_neighborhood")
def opposition_by_neighborhood():
    """All neighborhoods ranked by opposition intensity."""
    if _index is None:
        init()
    neighborhoods = _index.get('neighborhoods', {})

    ranked = []
    for name, data in neighborhoods.items():
        ranked.append({
            "neighborhood": name,
            "opposition_ratio": data.get('avg_opposition_ratio', 0),
            "risk_level": data.get('risk_level', 'Unknown'),
            "hearings_analyzed": data.get('hearings_analyzed', 0),
            "denial_rate": data.get('denial_rate'),
            "trend": data.get('trend', 'stable'),
        })

    ranked.sort(key=lambda x: x['opposition_ratio'], reverse=True)
    return {
        "neighborhoods": ranked,
        "total": len(ranked),
        "generated": _index.get('generated'),
    }


@router.get("/score")
def opposition_score(
    neighborhood: Optional[str] = Query(None, description="Neighborhood name"),
    ward: Optional[str] = Query(None, description="Ward number"),
    variance_types: Optional[str] = Query(None, description="Comma-separated variance types"),
):
    """Get opposition risk score for a specific project profile."""
    if _index is None:
        init()
    neighborhoods = _index.get('neighborhoods', {})

    if not neighborhood and not ward:
        raise HTTPException(status_code=400, detail="Provide neighborhood or ward")

    # Find matching neighborhood
    match = None
    if neighborhood:
        nb_lower = neighborhood.lower().strip()
        for name, data in neighborhoods.items():
            if name.lower() == nb_lower or nb_lower in name.lower():
                match = (name, data)
                break

    if not match:
        return {
            "neighborhood": neighborhood or f"Ward {ward}",
            "opposition_ratio": None,
            "risk_level": "Unknown",
            "note": "No opposition data available for this area.",
            "variance_risks": [],
        }

    name, data = match

    # Get variance-specific opposition
    variance_risks = []
    if variance_types:
        vtypes = [v.strip().lower() for v in variance_types.split(',') if v.strip()]
        var_opp = data.get('variance_opposition', {})
        for vt in vtypes:
            for key, stats in var_opp.items():
                if vt in key.lower():
                    variance_risks.append({
                        "variance_type": key,
                        "opposition_ratio": stats.get('opposition_ratio', 0),
                        "hearings": stats.get('hearings', 0),
                        "risk_level": stats.get('risk_level', 'Unknown'),
                    })

    return {
        "neighborhood": name,
        "opposition_ratio": data.get('avg_opposition_ratio'),
        "risk_level": data.get('risk_level', 'Unknown'),
        "hearings_analyzed": data.get('hearings_analyzed', 0),
        "denial_rate": data.get('denial_rate'),
        "trend": data.get('trend', 'stable'),
        "variance_risks": variance_risks,
    }
