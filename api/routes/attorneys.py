"""
Attorney Routes — Attorney profiles, search, and similar case lookup.
Designed for zoning attorneys (the repeat buyer: 10-50 cases/year).

Usage: In main.py, add:
    from routes.attorneys import router as attorney_router, init as attorney_init
    app.include_router(attorney_router)
    attorney_init(zba_df, VARIANCE_TYPES)
"""

import logging
from fastapi import APIRouter, HTTPException, Query
import pandas as pd
import numpy as np
import re
import time
from difflib import SequenceMatcher
from typing import Optional, List

logger = logging.getLogger("permitiq")
router = APIRouter(prefix="/attorneys", tags=["Attorneys"])

# Data injected by main.py at startup
_zba_df = None
_VARIANCE_TYPES = []


def _safe_ward(w):
    """Convert ward value to clean string, handling NaN/float."""
    if pd.isna(w):
        return None
    try:
        return str(int(float(w)))
    except (ValueError, TypeError):
        return None

# Simple TTL cache
_cache = {}
CACHE_TTL = 3600


def _cached(key, compute_fn):
    if key in _cache:
        val, ts = _cache[key]
        if time.time() - ts < CACHE_TTL:
            return val
    result = compute_fn()
    _cache[key] = (result, time.time())
    return result


def init(zba_df, variance_types):
    """Called by main.py after data loads."""
    global _zba_df, _VARIANCE_TYPES
    _zba_df = zba_df
    _VARIANCE_TYPES = variance_types
    _cache.clear()


def _require_data():
    if _zba_df is None:
        raise HTTPException(status_code=500, detail="ZBA data not loaded")
    return _zba_df


def _normalize_name(name: str) -> str:
    """Normalize attorney name for matching: lowercase, strip titles, collapse whitespace."""
    if not name:
        return ""
    n = name.lower().strip()
    # Remove common prefixes/suffixes
    n = re.sub(r'\b(esq\.?|attorney|atty\.?|mr\.?|ms\.?|mrs\.?|dr\.?|jr\.?|sr\.?|ii|iii)\b', '', n)
    n = re.sub(r'c/o\s+', '', n)
    n = re.sub(r'[,.\-\'"]', '', n)
    n = re.sub(r'\s+', ' ', n).strip()
    return n


def _get_decided_df():
    """Return only cases with a decision."""
    df = _require_data()
    return df[df['decision_clean'].notna()].copy()


def _find_attorney_exact(name: str):
    """Find an attorney by exact contact match (case-insensitive)."""
    df = _get_decided_df()
    contacts = df['contact'].dropna()
    # Try exact match first (case-insensitive)
    mask = contacts.str.lower() == name.lower().strip()
    if mask.any():
        return df[df['contact'].str.lower() == name.lower().strip()]
    # Try normalized match
    norm = _normalize_name(name)
    df['_norm_contact'] = df['contact'].apply(lambda x: _normalize_name(str(x)) if pd.notna(x) else '')
    mask = df['_norm_contact'] == norm
    if mask.any():
        result = df[mask].copy()
        result.drop(columns=['_norm_contact'], inplace=True)
        return result
    return None


def _get_year(row):
    """Extract year from source_pdf or filing_date."""
    if 'source_pdf' in row.index and pd.notna(row.get('source_pdf')):
        m = re.search(r'(20\d{2})', str(row['source_pdf']))
        if m:
            return int(m.group(1))
    if 'filing_date' in row.index and pd.notna(row.get('filing_date')):
        try:
            return pd.to_datetime(row['filing_date']).year
        except Exception as e:
            logger.debug("Failed to parse filing_date '%s': %s", row.get('filing_date'), e)
    return None


def _extract_year_series(df):
    """Extract year for an entire DataFrame efficiently."""
    years = pd.Series(index=df.index, dtype='float64')
    if 'source_pdf' in df.columns:
        extracted = df['source_pdf'].str.extract(r'(20\d{2})', expand=False)
        years = pd.to_numeric(extracted, errors='coerce')
    if 'filing_date' in df.columns:
        filing_years = pd.to_datetime(df['filing_date'], errors='coerce').dt.year
        years = years.fillna(filing_years)
    return years


def _safe_int(val):
    try:
        if pd.isna(val):
            return 0
        return int(val)
    except (ValueError, TypeError):
        return 0


@router.get("/search")
def search_attorneys(q: str = Query(..., min_length=2, description="Attorney name to search for")):
    """Search for attorneys/contacts by name (fuzzy match).

    Returns matching attorneys with basic stats, sorted by relevance.
    """
    df = _get_decided_df()
    contacts = df['contact'].dropna().unique()

    q_norm = _normalize_name(q)
    q_lower = q.lower().strip()

    scored = []
    for contact in contacts:
        c_str = str(contact).strip()
        if len(c_str) < 2:
            continue
        c_lower = c_str.lower()
        c_norm = _normalize_name(c_str)

        # Exact substring match gets highest score
        if q_lower in c_lower:
            score = 0.95
        elif q_norm in c_norm:
            score = 0.90
        else:
            # Fuzzy match
            score = SequenceMatcher(None, q_norm, c_norm).ratio()

        if score >= 0.45:
            scored.append((c_str, score))

    scored.sort(key=lambda x: -x[1])
    top = scored[:20]

    results = []
    for name, score in top:
        cases = df[df['contact'] == name]
        total = len(cases)
        approved = int((cases['decision_clean'] == 'APPROVED').sum())
        denied = total - approved
        results.append({
            "name": name,
            "match_score": round(score, 3),
            "total_cases": total,
            "approved": approved,
            "denied": denied,
            "approval_rate": round(approved / total, 3) if total > 0 else 0,
        })

    return {"query": q, "results": results}


@router.get("/{attorney_name}/profile")
def attorney_profile(attorney_name: str):
    """Full attorney profile with win rate, ward breakdown, variance specialties,
    yearly trends, complexity metrics, recent cases, and comparison to average.

    This is the core endpoint for attorney-focused users.
    """
    cases = _find_attorney_exact(attorney_name)
    if cases is None or len(cases) == 0:
        raise HTTPException(status_code=404, detail=f"No cases found for '{attorney_name}'. Use /attorneys/search?q=... to find the correct name.")

    # Use the actual contact name from the data (for display)
    _contact_mode = cases['contact'].dropna().mode()
    display_name = _contact_mode.iloc[0] if not _contact_mode.empty else attorney_name

    total = len(cases)
    approved = int((cases['decision_clean'] == 'APPROVED').sum())
    denied = total - approved
    win_rate = round(approved / total, 3) if total > 0 else 0

    # --- Cases by ward ---
    ward_breakdown = []
    if 'ward' in cases.columns:
        cases['_ward'] = cases['ward'].apply(_safe_ward)
        ward_groups = cases[cases['_ward'].notna()].groupby('_ward')
        for ward, grp in ward_groups:
            w_total = len(grp)
            w_approved = int((grp['decision_clean'] == 'APPROVED').sum())
            ward_breakdown.append({
                "ward": ward,
                "total_cases": w_total,
                "approved": w_approved,
                "denied": w_total - w_approved,
                "approval_rate": round(w_approved / w_total, 3) if w_total > 0 else 0,
            })
        ward_breakdown.sort(key=lambda x: -x['total_cases'])

    # --- Cases by variance type ---
    variance_breakdown = []
    if 'variance_types' in cases.columns:
        vt_str = cases['variance_types'].fillna('')
        for vt in _VARIANCE_TYPES:
            mask = vt_str.str.contains(vt, na=False)
            matched = cases[mask]
            if len(matched) > 0:
                v_approved = int((matched['decision_clean'] == 'APPROVED').sum())
                v_total = len(matched)
                variance_breakdown.append({
                    "variance_type": vt,
                    "total_cases": v_total,
                    "approved": v_approved,
                    "denied": v_total - v_approved,
                    "approval_rate": round(v_approved / v_total, 3) if v_total > 0 else 0,
                })
        variance_breakdown.sort(key=lambda x: -x['total_cases'])

    # --- Cases by year ---
    year_breakdown = []
    years = _extract_year_series(cases)
    cases_with_year = cases.copy()
    cases_with_year['_year'] = years
    year_groups = cases_with_year[cases_with_year['_year'].notna()].groupby('_year')
    for year, grp in year_groups:
        y_total = len(grp)
        y_approved = int((grp['decision_clean'] == 'APPROVED').sum())
        year_breakdown.append({
            "year": int(year),
            "total_cases": y_total,
            "approved": y_approved,
            "denied": y_total - y_approved,
            "approval_rate": round(y_approved / y_total, 3) if y_total > 0 else 0,
        })
    year_breakdown.sort(key=lambda x: x['year'])

    # --- Complexity metrics ---
    avg_variances = float(cases['num_variances'].fillna(0).mean()) if 'num_variances' in cases.columns else 0

    # --- Recent cases (last 10) ---
    recent_cases = []
    # Sort by year descending, then by index
    sorted_cases = cases_with_year.sort_values('_year', ascending=False, na_position='last').head(10)
    for _, row in sorted_cases.iterrows():
        case_item = {
            "case_number": str(row.get('case_number', '')) if pd.notna(row.get('case_number')) else '',
            "address": str(row.get('address_clean', row.get('address', ''))) if pd.notna(row.get('address_clean', row.get('address'))) else '',
            "decision": str(row.get('decision_clean', '')),
            "year": int(row['_year']) if pd.notna(row.get('_year')) else None,
            "ward": _safe_ward(row.get('ward')),
        }
        if 'variance_types' in row.index and pd.notna(row.get('variance_types')):
            case_item["variance_types"] = str(row['variance_types'])
        if 'num_variances' in row.index and pd.notna(row.get('num_variances')):
            case_item["num_variances"] = _safe_int(row['num_variances'])
        recent_cases.append(case_item)

    # --- Comparison to average ---
    all_df = _get_decided_df()
    overall_rate = float((all_df['decision_clean'] == 'APPROVED').mean())
    # Average across all contacts with at least 3 cases
    contact_rates = all_df.groupby('contact')['decision_clean'].agg(
        total='count',
        approved=lambda x: (x == 'APPROVED').sum()
    )
    contact_rates = contact_rates[contact_rates['total'] >= 3]
    contact_rates['rate'] = contact_rates['approved'] / contact_rates['total']
    avg_attorney_rate = float(contact_rates['rate'].mean()) if len(contact_rates) > 0 else overall_rate

    # Percentile rank among attorneys with 3+ cases
    if len(contact_rates) > 0:
        percentile = float((contact_rates['rate'] < win_rate).mean() * 100)
    else:
        percentile = 50.0

    # Streaks
    recent_decisions = cases_with_year.sort_values('_year', ascending=False)['decision_clean'].tolist()
    current_streak = 0
    streak_type = recent_decisions[0] if recent_decisions else None
    for d in recent_decisions:
        if d == streak_type:
            current_streak += 1
        else:
            break

    return {
        "name": display_name,
        "total_cases": total,
        "approved": approved,
        "denied": denied,
        "win_rate": win_rate,
        "loss_rate": round(1 - win_rate, 3),
        "wards": ward_breakdown,
        "top_ward": ward_breakdown[0]["ward"] if ward_breakdown else None,
        "variance_specialties": variance_breakdown,
        "yearly_performance": year_breakdown,
        "avg_variances_per_case": round(avg_variances, 2),
        "recent_cases": recent_cases,
        "comparison": {
            "overall_zba_rate": round(overall_rate, 3),
            "avg_attorney_rate": round(avg_attorney_rate, 3),
            "vs_overall": round(win_rate - overall_rate, 3),
            "vs_avg_attorney": round(win_rate - avg_attorney_rate, 3),
            "percentile_rank": round(percentile, 1),
            "beats_average_by": f"{abs(win_rate - avg_attorney_rate):.1%}" if win_rate >= avg_attorney_rate else None,
            "below_average_by": f"{abs(win_rate - avg_attorney_rate):.1%}" if win_rate < avg_attorney_rate else None,
        },
        "current_streak": {
            "type": streak_type,
            "length": current_streak,
        },
    }


@router.get("/recommend")
def recommend_attorney(
    variance_types: str = Query(..., description="Comma-separated variance types (e.g. height,parking)"),
    ward: Optional[str] = Query(None, description="Ward number"),
    limit: int = Query(5, ge=1, le=20),
    min_cases: int = Query(3, ge=1, le=20),
):
    """Recommend top attorneys for a given variance type and ward combination.

    Returns attorneys ranked by win rate for the specified filters,
    with minimum case threshold to ensure statistical significance.
    """
    df = _get_decided_df()

    vtypes = [v.strip().lower() for v in variance_types.split(',') if v.strip()]
    if not vtypes:
        raise HTTPException(status_code=400, detail="At least one variance type required")

    # Filter to cases matching any of the requested variance types
    vt_col = df['variance_types'].fillna('')
    mask = pd.Series(False, index=df.index)
    for vt in vtypes:
        mask |= vt_col.str.contains(vt, na=False, case=False)
    filtered = df[mask].copy()

    # Filter by ward if provided
    if ward:
        try:
            ward_num = float(ward)
            filtered = filtered[filtered['ward'] == ward_num]
        except (ValueError, TypeError):
            pass

    if len(filtered) == 0:
        return {"variance_types": vtypes, "ward": ward, "attorneys": [], "note": "No cases found for this combination."}

    # Only include cases with a named contact (attorney)
    with_attorney = filtered[filtered['contact'].notna() & (filtered['contact'].str.len() > 2)]

    # Group by attorney
    attorney_stats = with_attorney.groupby('contact').agg(
        total=('decision_clean', 'count'),
        approved=('decision_clean', lambda x: (x == 'APPROVED').sum()),
    ).reset_index()
    attorney_stats = attorney_stats[attorney_stats['total'] >= min_cases]
    attorney_stats['approval_rate'] = (attorney_stats['approved'] / attorney_stats['total']).round(3)

    # Also compute each attorney's overall stats for context
    all_stats = df[df['contact'].notna()].groupby('contact').agg(
        overall_total=('decision_clean', 'count'),
        overall_approved=('decision_clean', lambda x: (x == 'APPROVED').sum()),
    ).reset_index()
    all_stats['overall_rate'] = (all_stats['overall_approved'] / all_stats['overall_total']).round(3)

    merged = attorney_stats.merge(all_stats, on='contact', how='left')

    # Specialist score: what fraction of this attorney's cases involve these variance types
    merged['specialist_pct'] = (merged['total'] / merged['overall_total']).round(3)

    # Sort by approval rate, then by case count
    merged = merged.sort_values(['approval_rate', 'total'], ascending=[False, False])

    attorneys = []
    for _, row in merged.head(limit).iterrows():
        attorneys.append({
            "name": row['contact'],
            "cases_for_filter": int(row['total']),
            "approved_for_filter": int(row['approved']),
            "approval_rate": float(row['approval_rate']),
            "overall_cases": int(row['overall_total']),
            "overall_rate": float(row['overall_rate']),
            "specialist_pct": float(row['specialist_pct']),
        })

    # Overall baseline for comparison
    baseline_rate = float((filtered['decision_clean'] == 'APPROVED').mean()) if len(filtered) > 0 else 0

    return {
        "variance_types": vtypes,
        "ward": ward,
        "baseline_approval_rate": round(baseline_rate, 3),
        "min_cases_threshold": min_cases,
        "attorneys": attorneys,
        "total_matching_cases": len(filtered),
    }


@router.get("/{attorney_name}/similar_cases")
def attorney_similar_cases(
    attorney_name: str,
    variance_type: Optional[List[str]] = Query(None, description="Filter by variance types (e.g. height, parking)"),
    ward: Optional[str] = Query(None, description="Filter by ward number"),
    limit: int = Query(20, ge=1, le=50, description="Max results"),
):
    """Find cases similar to a potential filing that this attorney has won.

    Attorneys use this to show clients: 'Here are cases like yours that I won.'
    Optionally filter by variance types and/or ward to narrow results.
    """
    cases = _find_attorney_exact(attorney_name)
    if cases is None or len(cases) == 0:
        raise HTTPException(status_code=404, detail=f"No cases found for '{attorney_name}'. Use /attorneys/search?q=... to find the correct name.")

    _contact_mode = cases['contact'].dropna().mode()
    display_name = _contact_mode.iloc[0] if not _contact_mode.empty else attorney_name
    filtered = cases.copy()

    # Filter by variance types
    if variance_type:
        for vt in variance_type:
            vt_clean = vt.strip().lower()
            if vt_clean:
                filtered = filtered[filtered['variance_types'].fillna('').str.contains(vt_clean, na=False)]

    # Filter by ward
    if ward:
        try:
            ward_num = float(ward)
            filtered = filtered[filtered['ward'] == ward_num]
        except (ValueError, TypeError):
            logger.warning("Invalid ward value in attorney similar_cases filter: %s", ward)

    if len(filtered) == 0:
        return {
            "attorney": display_name,
            "filters": {"variance_types": variance_type, "ward": ward},
            "total_matching": 0,
            "cases": [],
            "note": "No matching cases found. Try broadening your search."
        }

    # Sort: approved first, then by year descending
    years = _extract_year_series(filtered)
    filtered = filtered.copy()
    filtered['_year'] = years
    filtered['_is_approved'] = (filtered['decision_clean'] == 'APPROVED').astype(int)
    filtered = filtered.sort_values(['_is_approved', '_year'], ascending=[False, False])

    result_cases = []
    for _, row in filtered.head(limit).iterrows():
        case_item = {
            "case_number": str(row.get('case_number', '')) if pd.notna(row.get('case_number')) else '',
            "address": str(row.get('address_clean', row.get('address', ''))) if pd.notna(row.get('address_clean', row.get('address'))) else '',
            "decision": str(row.get('decision_clean', '')),
            "year": int(row['_year']) if pd.notna(row.get('_year')) else None,
            "ward": _safe_ward(row.get('ward')),
        }
        if 'variance_types' in row.index and pd.notna(row.get('variance_types')):
            case_item["variance_types"] = str(row['variance_types'])
        if 'num_variances' in row.index and pd.notna(row.get('num_variances')):
            case_item["num_variances"] = _safe_int(row['num_variances'])
        if 'tracker_description' in row.index and pd.notna(row.get('tracker_description')):
            desc = str(row['tracker_description'])[:200]
            case_item["description"] = desc
        result_cases.append(case_item)

    won = sum(1 for c in result_cases if c['decision'] == 'APPROVED')
    lost = len(result_cases) - won

    return {
        "attorney": display_name,
        "filters": {"variance_types": variance_type, "ward": ward},
        "total_matching": len(filtered),
        "showing": len(result_cases),
        "won": won,
        "lost": lost,
        "win_rate": round(won / len(result_cases), 3) if result_cases else 0,
        "cases": result_cases,
    }
