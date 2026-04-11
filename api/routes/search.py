"""
Search endpoints — address search, case history, autocomplete.
"""

import re
import logging
import pandas as pd
from functools import lru_cache
from fastapi import APIRouter, HTTPException

from api import state
from api.utils import safe_int, safe_str, _format_date, normalize_address

logger = logging.getLogger("permitiq")
router = APIRouter()


@lru_cache(maxsize=256)
def _cached_search(q_norm: str) -> list:
    """Cache search results by normalized query."""
    return _do_search(q_norm)


def _do_search(q_norm: str) -> list:
    """Core search logic, separated for caching."""
    addr_df = state.zba_df[
        state.zba_df['address_clean'].notna() &
        state.zba_df['address_clean'].str.match(r'^\d', na=False) &
        (state.zba_df['address_clean'].str.len() > 5)
    ]

    addr_norm_col = '_addr_norm' if '_addr_norm' in addr_df.columns else None
    if addr_norm_col is None:
        addr_df = addr_df.copy()
        addr_df['_addr_norm'] = addr_df['address_clean'].apply(normalize_address)

    q_pattern = re.escape(q_norm)
    if re.match(r'^\d', q_norm):
        q_pattern = r'(?:^|\s|-)' + q_pattern
    mask = addr_df['_addr_norm'].str.contains(q_pattern, na=False, regex=True)

    q_num_match = re.match(r'^(\d+)\s+(.+)', q_norm)
    if q_num_match:
        q_num = q_num_match.group(1)
        q_street = q_num_match.group(2)
        range_mask = addr_df['_addr_norm'].str.contains(
            r'\d+\s*[-\u2013]\s*' + re.escape(q_num) + r'\s+' + re.escape(q_street),
            na=False, regex=True
        )
        mask = mask | range_mask

    if mask.sum() == 0:
        words = q_norm.split()
        mask = pd.Series(True, index=addr_df.index)
        for word in words:
            if len(word) > 2:
                mask = mask & addr_df['_addr_norm'].str.contains(word, na=False, regex=False)

    matches = addr_df[mask]

    zoning_col = 'zoning_clean' if 'zoning_clean' in matches.columns else (
        'zoning_district' if 'zoning_district' in matches.columns else (
        'zoning' if 'zoning' in matches.columns else None))
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
        agg_dict['latest_date'] = (date_col, 'max')
        agg_dict['earliest_date'] = (date_col, 'min')

    grouped = matches.groupby('_addr_norm').agg(**agg_dict)

    _q_num = q_num_match.group(1) if q_num_match else None

    def _relevance(addr_norm_key):
        if addr_norm_key.startswith(q_norm):
            return 0
        if _q_num and re.search(r'\d+\s*[-\u2013]\s*' + re.escape(_q_num) + r'\b', addr_norm_key):
            return 1
        return 2

    grouped['_relevance'] = [_relevance(idx) for idx in grouped.index]
    grouped = grouped.sort_values(['_relevance', 'total_cases'], ascending=[True, False]).drop(columns=['_relevance']).head(10)

    # Build address→parcel_id lookup for enrichment
    addr_to_parcel = {}
    if state.parcel_addr_df is not None and '_addr_norm' in state.parcel_addr_df.columns:
        addr_to_parcel = dict(zip(state.parcel_addr_df['_addr_norm'], state.parcel_addr_df['parcel_id']))

    results = []
    for addr_norm_key, row in grouped.iterrows():
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
            "earliest_date": _format_date(row.get('earliest_date', '')),
            "latest_case": str(row['latest_case']),
        }
        # Enrich with parcel_id from geocoder (handle range addresses like "55-57 centre st")
        pid = addr_to_parcel.get(addr_norm_key)
        if not pid:
            # Try extracting individual numbers from range addresses (e.g., "55-57 centre st" → "57 centre st")
            range_match = re.match(r'^(\d+)\s*[-\u2013]\s*(\d+)\s+(.+)', addr_norm_key)
            if range_match:
                for num in [range_match.group(2), range_match.group(1)]:
                    pid = addr_to_parcel.get(f"{num} {range_match.group(3)}")
                    if pid:
                        break
        if pid:
            result_item["parcel_id"] = str(pid)
        _applicant = safe_str(row.get('applicant', '')).strip()
        if _applicant and _applicant.lower() not in ('nan', 'none'):
            result_item["applicant"] = _applicant
        results.append(result_item)

    return results


@router.get("/search", tags=["Search"])
def search_address(q: str):
    """Search for addresses with ZBA history."""
    if state.zba_df is None:
        raise HTTPException(status_code=500, detail="ZBA data not loaded")

    q_norm = normalize_address(q)
    if len(q_norm) < 2:
        return {"query": q, "results": [], "total_results": 0}

    results = _cached_search(q_norm)
    return {"query": q, "results": results, "total_results": len(results)}


@router.get("/address/{address}/cases", tags=["Search"])
def get_address_cases(address: str, limit: int = 20):
    """Get all ZBA cases for a specific address with full details."""
    if state.zba_df is None:
        raise HTTPException(status_code=500, detail="ZBA data not loaded")

    addr_norm = normalize_address(address)
    addr_df = state.zba_df[state.zba_df['address_clean'].notna()]

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
            _ward_str = str(int(float(_ward_val))) if pd.notna(_ward_val) and str(_ward_val).replace('.', '', 1).isdigit() else str(_ward_val or '')
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

    return {"address": address, "cases": cases, "total": len(cases)}


@router.get("/case/{case_number}", tags=["Search"])
def get_case(case_number: str):
    """Look up a single ZBA case by BOA case number. Returns full case details."""
    if state.zba_df is None:
        raise HTTPException(status_code=503, detail="Dataset not loaded")

    # Normalize case number format (handle BOA-1234567 or just 1234567)
    q = case_number.strip().upper()
    mask = state.zba_df['case_number'].astype(str).str.upper().str.strip() == q
    if mask.sum() == 0 and not q.startswith('BOA-'):
        mask = state.zba_df['case_number'].astype(str).str.upper().str.strip() == f"BOA-{q}"
    if mask.sum() == 0:
        # Try partial match
        q_escaped = re.escape(q)
        mask = state.zba_df['case_number'].astype(str).str.upper().str.contains(q_escaped, na=False)

    if mask.sum() == 0:
        raise HTTPException(status_code=404, detail=f"Case {case_number} not found")

    row = state.zba_df[mask].iloc[0]

    variance_types = ['height', 'far', 'lot_area', 'lot_frontage',
                      'front_setback', 'rear_setback', 'side_setback',
                      'parking', 'conditional_use', 'open_space', 'density', 'nonconforming']
    project_types = ['demolition', 'new_construction', 'addition', 'conversion',
                     'renovation', 'subdivision', 'adu', 'roof_deck', 'parking',
                     'single_family', 'multi_family', 'mixed_use']

    variances = [vt for vt in variance_types if row.get(f'var_{vt}', 0) == 1]
    projects = [pt for pt in project_types if row.get(f'proj_{pt}', 0) == 1]

    result = {
        "case_number": safe_str(row.get('case_number')),
        "address": safe_str(row.get('address_clean')),
        "decision": safe_str(row.get('decision_clean')),
        "date": _format_date(row.get('date')),
        "ward": safe_str(row.get('ward')),
        "zoning_district": safe_str(row.get('zoning_clean') or row.get('zoning_district')),
        "applicant": safe_str(row.get('applicant_name')),
        "contact": safe_str(row.get('contact')),
        "has_attorney": bool(row.get('has_attorney', False)),
        "variances": variances,
        "project_types": projects,
        "is_building_appeal": bool(row.get('is_building_appeal', False)),
        "proposed_units": safe_int(row.get('proposed_units')),
        "proposed_stories": safe_int(row.get('proposed_stories')),
        "is_residential": bool(row.get('is_residential', False)),
        "is_commercial": bool(row.get('is_commercial', False)),
        "lot_size_sf": row.get('lot_size_sf') if pd.notna(row.get('lot_size_sf')) else None,
        "total_value": row.get('total_value') if pd.notna(row.get('total_value')) else None,
    }
    return result


@router.get("/autocomplete", tags=["Search"])
def autocomplete(q: str = "", limit: int = 10):
    """Address autocomplete from 175K property records."""
    if not q or len(q) < 3:
        return {"suggestions": []}
    if state.parcel_addr_df is None:
        return {"suggestions": []}

    q_norm = normalize_address(q)
    q_escaped = re.escape(q_norm)
    matches = state.parcel_addr_df[state.parcel_addr_df['_addr_norm'].str.contains(q_escaped, na=False)].head(limit)

    return {
        "suggestions": [
            {"address": row['address'], "parcel_id": row['parcel_id']}
            for _, row in matches.iterrows()
        ]
    }
