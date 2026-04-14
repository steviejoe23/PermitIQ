"""
Hearing Prep Report — Premium comprehensive report for upcoming ZBA hearings.

Aggregates prediction, similar cases, board member tendencies, opposition risk,
attorney recommendations, and filing strategy into a single tactical report.
"""

import logging
from fastapi import APIRouter, HTTPException, Query
from typing import Optional

logger = logging.getLogger("permitiq")
router = APIRouter(prefix="/hearing_prep", tags=["Hearing Prep"])


def _safe_call(fn, *args, **kwargs):
    """Call a route handler directly and return its result, or None on error."""
    try:
        return fn(*args, **kwargs)
    except HTTPException:
        return None
    except Exception as e:
        logger.warning("Hearing prep sub-call failed: %s", e)
        return None


@router.get("/generate")
def generate_hearing_prep(
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
        "warnings": [],
    }

    # 1. Risk score for parcel
    if parcel_id:
        try:
            from api.routes.risk_score import parcel_risk_score
            risk = _safe_call(parcel_risk_score, parcel_id)
            if risk:
                report["risk_score"] = risk
        except ImportError:
            report["warnings"].append("Risk score module not available")

    # 2. Board member tendencies
    try:
        from api.routes.board_members import members_for_hearing
        board = _safe_call(members_for_hearing, variance_types=variance_types)
        if board:
            report["board_analysis"] = board
        else:
            report["warnings"].append("Board member data unavailable")
    except ImportError:
        report["warnings"].append("Board members module not available")

    # 3. Opposition risk
    if neighborhood or ward:
        try:
            from api.routes.opposition import opposition_score
            opposition = _safe_call(opposition_score, neighborhood=neighborhood, ward=ward, variance_types=variance_types)
            if opposition:
                report["opposition_risk"] = opposition
            else:
                report["warnings"].append("Opposition data unavailable")
        except ImportError:
            report["warnings"].append("Opposition module not available")

    # 4. Attorney recommendation
    if variance_types:
        try:
            from api.routes.attorneys import recommend_attorney
            attorney_rec = _safe_call(recommend_attorney, variance_types=variance_types, ward=ward, limit=5, min_cases=3)
            if attorney_rec:
                report["attorney_recommendations"] = attorney_rec
        except ImportError:
            report["warnings"].append("Attorney module not available")

        # If user has an attorney, get their profile
        if attorney:
            try:
                from api.routes.attorneys import attorney_profile
                profile = _safe_call(attorney_profile, attorney)
                if profile:
                    report["your_attorney"] = {
                        "name": profile.get("name"),
                        "win_rate": profile.get("win_rate"),
                        "total_cases": profile.get("total_cases"),
                        "percentile_rank": profile.get("comparison", {}).get("percentile_rank"),
                        "variance_specialties": profile.get("variance_specialties", [])[:5],
                    }
            except ImportError:
                pass

    # 5. Filing strategy
    if variance_types or ward:
        try:
            from api.routes.filing_strategy import recommend_timing
            strategy = _safe_call(recommend_timing, variance_types=variance_types, ward=ward)
            if strategy:
                report["filing_strategy"] = strategy
        except ImportError:
            report["warnings"].append("Filing strategy module not available")

    # 6. Generate tactical advice
    advice = _generate_advice(report)
    report["tactical_advice"] = advice

    # Remove empty warnings list
    if not report["warnings"]:
        del report["warnings"]

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
