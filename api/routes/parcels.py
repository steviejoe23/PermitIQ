"""
Parcel endpoints — lookup, geocode, nearby cases.
"""

import re
import logging
import pandas as pd
import numpy as np
from fastapi import APIRouter, HTTPException

from api import state
from api.utils import safe_float, safe_str, safe_int, _clean_case_date, _haversine_m, normalize_address

logger = logging.getLogger("permitiq")
router = APIRouter()


def _enrich_parcel_result(result: dict, parcel_id: str):
    """Add ZBA case history to a parcel result."""
    district = result.get("district", "")
    if state.zba_df is not None and 'zoning_district' in state.zba_df.columns:
        ward_lookup = state.zba_df[
            (state.zba_df['zoning_district'] == district) & state.zba_df['ward'].notna()
        ]['ward']
        if not ward_lookup.empty:
            ward_mode = ward_lookup.mode()
            if not ward_mode.empty and pd.notna(ward_mode.iloc[0]):
                try:
                    result["ward"] = str(int(ward_mode.iloc[0]))
                except (ValueError, TypeError):
                    pass

    if state.zba_df is not None and 'pa_parcel_id' in state.zba_df.columns:
        pid_float = safe_float(parcel_id, default=None)
        if pid_float is not None:
            matches = state.zba_df[state.zba_df['pa_parcel_id'] == pid_float]
            if not matches.empty:
                result["zba_cases"] = len(matches)
                addr = matches['address_clean'].dropna().iloc[0] if 'address_clean' in matches.columns and matches['address_clean'].notna().any() else None
                if addr:
                    result["address"] = str(addr)

    # Fallback: look up address from property assessment data if still missing
    if not result.get("address") and state.parcel_addr_df is not None:
        _pid_str = str(parcel_id).zfill(10)
        _addr_match = state.parcel_addr_df[state.parcel_addr_df['parcel_id'] == _pid_str]
        if not _addr_match.empty:
            _pa_addr = _addr_match.iloc[0].get('address', '')
            if _pa_addr and str(_pa_addr).lower() not in ('', 'nan', 'none'):
                result["address"] = str(_pa_addr)


@router.get("/parcels/{parcel_id}", tags=["Parcels"])
def get_parcel(parcel_id: str):
    """Look up zoning details and geometry for a Boston parcel by its 10-digit ID."""
    from api.services.database import query_parcel, db_available

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
            _enrich_parcel_result(result, parcel_id)
            return result

    if state.gdf is None:
        raise HTTPException(status_code=500, detail="No parcel data available (PostGIS and GeoJSON both unavailable)")

    row = state.gdf.loc[[parcel_id]] if parcel_id in state.gdf.index else state.gdf.iloc[0:0]
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


@router.get("/parcels/{parcel_id}/nearby_cases", tags=["Parcels"])
def nearby_cases(parcel_id: str, radius_m: int = 800, limit: int = 20, ward_only: bool = False):
    """
    Find ZBA cases near a parcel using real geographic distance.
    """
    if radius_m < 0:
        raise HTTPException(status_code=400, detail="radius_m must be non-negative")
    if limit < 1:
        limit = 1
    if state.gdf is None:
        raise HTTPException(status_code=500, detail="GeoJSON not loaded")
    if state.zba_df is None:
        raise HTTPException(status_code=500, detail="ZBA data not loaded")

    row = state.gdf.loc[[parcel_id]] if parcel_id in state.gdf.index else state.gdf.iloc[0:0]
    if row.empty:
        raise HTTPException(status_code=404, detail="Parcel not found")

    centroid = row.iloc[0].geometry.centroid
    parcel_lat, parcel_lon = centroid.y, centroid.x
    district = str(row.iloc[0].get("districts") or "")

    cases = []
    has_geo = False
    df = None

    if state._case_coords is not None and len(state._case_coords) > 0:
        df = state._case_coords.copy()
        df['dist_m'] = _haversine_m(parcel_lat, parcel_lon, df['lat'].values, df['lon'].values)
        df = df[df['dist_m'] <= radius_m].sort_values('dist_m')
        has_geo = True

    # Detect ward
    parcel_ward = ''
    if has_geo and df is not None and not df.empty:
        nearby_wards = df[df['ward'] != ''].head(10)['ward']
        if not nearby_wards.empty:
            try:
                parcel_ward = str(nearby_wards.mode().iloc[0])
            except Exception as e:
                logger.debug("Failed to compute ward mode from nearby cases: %s", e)
    if not parcel_ward and state.zba_df is not None and district:
        z_col = 'zoning_district' if 'zoning_district' in state.zba_df.columns else None
        if z_col:
            ward_lookup = state.zba_df[(state.zba_df[z_col] == district) & state.zba_df['ward'].notna()]['ward']
            if not ward_lookup.empty:
                try:
                    parcel_ward = str(int(ward_lookup.mode().iloc[0]))
                except Exception as e:
                    logger.debug("Failed to compute ward mode from district lookup: %s", e)

    # Geographic results
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
        # Fallback: district-based matching
        z_col = 'zoning_district' if 'zoning_district' in state.zba_df.columns else None
        if z_col and district:
            nearby = state.zba_df[
                (state.zba_df[z_col] == district) &
                (state.zba_df['decision_clean'].notna())
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
                _ward_clean = str(int(float(_w))) if pd.notna(_w) and str(_w).replace('.', '', 1).isdigit() else safe_str(_w)
                cases.append({
                    "case_number": cn,
                    "address": addr,
                    "decision": str(c.get('decision_clean', '')),
                    "ward": _ward_clean,
                    "date": _clean_case_date(c),
                    "distance_m": None,
                    "distance_ft": None,
                })

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


@router.get("/geocode", tags=["Parcels"])
def geocode_address(q: str):
    """Find a parcel ID from a street address."""
    if state.parcel_addr_df is None:
        raise HTTPException(status_code=503, detail="Property address data not loaded")
    if state.gdf is None:
        raise HTTPException(status_code=500, detail="GeoJSON not loaded")

    q_norm = normalize_address(q)
    if len(q_norm) < 3:
        return {"query": q, "results": []}

    range_match = re.match(r'^(\d+)\s*[-\u2013]\s*(\d+)\s+(.*)', q_norm)
    if range_match:
        queries_to_try = [
            f"{range_match.group(2)} {range_match.group(3)}",
            f"{range_match.group(1)} {range_match.group(3)}",
            q_norm,
        ]
    else:
        queries_to_try = [q_norm]

    matches = pd.DataFrame()
    for q_try in queries_to_try:
        words = q_try.split()
        mask = pd.Series(True, index=state.parcel_addr_df.index)
        for word in words:
            if len(word) > 1:
                if word.isdigit():
                    mask = mask & state.parcel_addr_df['_addr_norm'].str.contains(
                        r'(?:^|\s)' + re.escape(word) + r'(?:\s|$)', na=False, regex=True)
                else:
                    mask = mask & state.parcel_addr_df['_addr_norm'].str.contains(word, na=False, regex=False)
        matches = state.parcel_addr_df[mask].head(10)
        if not matches.empty:
            break

    # Fallback: nearest address on same street
    if matches.empty and queries_to_try:
        q_try = queries_to_try[0]
        num_match = re.match(r'^(\d+)\s+(.+)', q_try)
        if num_match:
            target_num = int(num_match.group(1))
            street_part = num_match.group(2)
            street_mask = state.parcel_addr_df['_addr_norm'].str.contains(street_part, na=False, regex=False)
            street_matches = state.parcel_addr_df[street_mask].copy()
            if not street_matches.empty:
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
        geo_match = state.gdf.loc[[pid]] if pid in state.gdf.index else state.gdf.iloc[0:0]
        if not geo_match.empty:
            geo_row = geo_match.iloc[0]
            entry["zoning_code"] = str(geo_row.get("primary_zoning") or "")
            entry["district"] = str(geo_row.get("districts") or "")
        results.append(entry)

    return {"query": q, "results": results, "total": len(results)}
