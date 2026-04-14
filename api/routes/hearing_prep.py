"""
Hearing Prep Report — Premium comprehensive report for upcoming ZBA hearings.

Aggregates prediction, similar cases, board member tendencies, opposition risk,
attorney recommendations, and filing strategy into a single tactical report.
"""

import logging
import requests
from fastapi import APIRouter, HTTPException, Query, Request
from typing import Optional

logger = logging.getLogger("permitiq")
router = APIRouter(prefix="/hearing_prep", tags=["Hearing Prep"])


def _internal_get(request: Request, path: str, params: dict = None):
    """Make an internal API call to another endpoint."""
    base = str(request.base_url).rstrip('/')
    try:
        resp = requests.get(f"{base}{path}", params=params, timeout=10)
        if resp.status_code == 200:
            return resp.json()
    except Exception as e:
        logger.warning("Internal call to %s failed: %s", path, e)
    return None


@router.get("/generate")
def generate_hearing_prep(
    request: Request,
    address: str = Query(..., description="Property address"),
    parcel_id: Optional[str] = Query(None, description="Parcel ID"),
    variance_types: Optional[str] = Query(None, description="Comma-separated variance types"),
    ward: Optional[str] = Query(None, description="Ward number"),
    neighborhood: Optional[str] = Query(None, description="Neighborhood name"),
    attorney: Optional[str] = Query(None, description="Your attorney's name"),
    project_type: Optional[str] = Query(None, description="Project type"),
):
    """Generate a comprehensive hearing prep report.

    Combines data from multiple endpoints into a single tactical report
    designed to help developers and attorneys prepare for ZBA hearings.
    """
    report = {
        "address": address,
        "parcel_id": parcel_id,
        "variance_types": variance_types,
        "ward": ward,
        "neighborhood": neighborhood,
    }

    # 1. Similar cases (from nearby cases if parcel_id available)
    if parcel_id:
        nearby = _internal_get(request, f"/parcels/{parcel_id}/nearby_cases", {"radius_m": 800, "limit": 15})
        if nearby:
            report["nearby_cases"] = {
                "cases": nearby.get("cases", []),
                "total": nearby.get("total", 0),
                "approval_rate": nearby.get("approval_rate"),
                "ward": nearby.get("ward"),
            }

        # Risk score
        risk = _internal_get(request, f"/risk/parcels/{parcel_id}")
        if risk:
            report["risk_score"] = risk

    # 2. Board member tendencies
    board = _internal_get(request, "/board_members/for_hearing",
                          {"variance_types": variance_types} if variance_types else {})
    if board:
        report["board_analysis"] = board

    # 3. Opposition risk
    if neighborhood or ward:
        opp_params = {}
        if neighborhood:
            opp_params["neighborhood"] = neighborhood
        if ward:
            opp_params["ward"] = ward
        if variance_types:
            opp_params["variance_types"] = variance_types
        opposition = _internal_get(request, "/opposition/score", opp_params)
        if opposition:
            report["opposition_risk"] = opposition

    # 4. Attorney recommendation or comparison
    if variance_types:
        rec_params = {"variance_types": variance_types, "limit": 5}
        if ward:
            rec_params["ward"] = ward
        attorney_rec = _internal_get(request, "/attorneys/recommend", rec_params)
        if attorney_rec:
            report["attorney_recommendations"] = attorney_rec

        # If user has an attorney, get their profile
        if attorney:
            profile = _internal_get(request, f"/attorneys/{attorney}/profile")
            if profile:
                report["your_attorney"] = {
                    "name": profile.get("name"),
                    "win_rate": profile.get("win_rate"),
                    "total_cases": profile.get("total_cases"),
                    "percentile_rank": profile.get("comparison", {}).get("percentile_rank"),
                    "variance_specialties": profile.get("variance_specialties", [])[:5],
                }

    # 5. Filing strategy
    strategy_params = {}
    if variance_types:
        strategy_params["variance_types"] = variance_types
    if ward:
        strategy_params["ward"] = ward
    strategy = _internal_get(request, "/filing_strategy/recommend", strategy_params)
    if strategy:
        report["filing_strategy"] = strategy

    # 6. Generate tactical advice
    advice = _generate_advice(report)
    report["tactical_advice"] = advice

    return report


def _generate_advice(report: dict) -> list:
    """Generate rule-based tactical advice from the aggregated data."""
    advice = []

    # Opposition-based advice
    opp = report.get("opposition_risk", {})
    if opp.get("risk_level") == "High":
        advice.append({
            "priority": "HIGH",
            "category": "Community Engagement",
            "text": f"High community opposition in {opp.get('neighborhood', 'this area')}. "
                    f"Hold a neighborhood meeting BEFORE your hearing to address concerns. "
                    f"Opposition ratio: {opp.get('opposition_ratio', 0):.0%}.",
        })
    elif opp.get("risk_level") == "Medium":
        advice.append({
            "priority": "MEDIUM",
            "category": "Community Engagement",
            "text": f"Moderate opposition levels in {opp.get('neighborhood', 'this area')}. "
                    f"Consider proactive outreach to abutters.",
        })

    # Variance-specific opposition
    for vr in opp.get("variance_risks", []):
        if vr.get("risk_level") == "High":
            advice.append({
                "priority": "HIGH",
                "category": "Variance Strategy",
                "text": f"{vr['variance_type']} variances face strong opposition in this neighborhood "
                        f"(opposition ratio: {vr.get('opposition_ratio', 0):.0%}). "
                        f"Prepare detailed justification and consider reducing scope.",
            })

    # Attorney advice
    recs = report.get("attorney_recommendations", {})
    your_atty = report.get("your_attorney", {})
    if not your_atty and recs.get("attorneys"):
        top = recs["attorneys"][0]
        advice.append({
            "priority": "HIGH",
            "category": "Legal Representation",
            "text": f"No attorney specified. Top-performing attorney for these variances: "
                    f"{top['name']} ({top['approval_rate']:.0%} approval rate, "
                    f"{top['cases_for_filter']} relevant cases).",
        })
    elif your_atty:
        rank = your_atty.get("percentile_rank", 50)
        if rank < 30:
            advice.append({
                "priority": "MEDIUM",
                "category": "Legal Representation",
                "text": f"Your attorney ranks in the bottom third of ZBA attorneys by win rate. "
                        f"Consider consulting a specialist for this variance type.",
            })

    # Filing timing
    strategy = report.get("filing_strategy", {})
    if strategy.get("best_month") and strategy.get("worst_month"):
        best = strategy["best_month"]
        worst = strategy["worst_month"]
        if strategy.get("seasonal_spread", 0) > 0.05:
            advice.append({
                "priority": "LOW",
                "category": "Timing",
                "text": f"Historical data suggests {best['name']} hearings have higher approval rates "
                        f"({best['approval_rate']:.0%}) vs {worst['name']} ({worst['approval_rate']:.0%}).",
            })

    # Risk score advice
    risk = report.get("risk_score", {})
    if risk.get("risk_score", 0) > 75:
        advice.append({
            "priority": "HIGH",
            "category": "Site Risk",
            "text": f"This parcel has a development difficulty score of {risk['risk_score']:.0f}/100. "
                    f"Expect elevated scrutiny. Strong legal representation essential.",
        })

    # Nearby case insights
    nearby = report.get("nearby_cases", {})
    if nearby.get("approval_rate") is not None and nearby["approval_rate"] < 0.8:
        advice.append({
            "priority": "MEDIUM",
            "category": "Area Patterns",
            "text": f"Nearby cases show a {nearby['approval_rate']:.0%} approval rate — "
                    f"below the city-wide average. The board may be more cautious in this area.",
        })

    if not advice:
        advice.append({
            "priority": "LOW",
            "category": "General",
            "text": "No specific risk factors identified. Standard preparation should suffice.",
        })

    # Sort by priority
    priority_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
    advice.sort(key=lambda x: priority_order.get(x["priority"], 3))

    return advice
