"""
Zoning analysis endpoints — district lookup, compliance check, variance analysis.
"""

import logging
import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException

from api import state
from api.utils import safe_float
from api.services.zoning_code import get_zoning_requirements, ZONING_REQUIREMENTS

logger = logging.getLogger("permitiq")
router = APIRouter()

# Cache for parcel zoning lookups (same parcel_id always returns same data)
_parcel_zoning_cache = {}

# Import db functions
try:
    from api.services.database import query_parcel, db_available
except ImportError:
    try:
        from services.database import query_parcel, db_available
    except ImportError:
        query_parcel = None
        db_available = lambda: False


def _get_parcel_zoning(parcel_id: str) -> dict:
    """Single source of truth for parcel zoning data. Cached per parcel_id."""
    if parcel_id in _parcel_zoning_cache:
        return _parcel_zoning_cache[parcel_id]
    result = _compute_parcel_zoning(parcel_id)
    if len(_parcel_zoning_cache) < 1000:  # Bound cache size
        _parcel_zoning_cache[parcel_id] = result
    return result


def _compute_parcel_zoning(parcel_id: str) -> dict:
    """Compute parcel zoning data (uncached)."""
    district, article, all_codes, multi_zoning = '', '', '', False
    subdistrict = ''
    subdistrict_type = ''
    subdistrict_use = ''
    neighborhood = ''
    in_gcod = False
    in_coastal_flood = False

    if db_available() and query_parcel is not None:
        db_row = query_parcel(parcel_id)
        if db_row is not None:
            district = db_row.get('primary_zoning', '')
            article = db_row.get('article', '')
            all_codes = db_row.get('all_zoning_codes', '')
            multi_zoning = db_row.get('multi_zoning', False)

    sub_max_far = None
    sub_max_height = None
    sub_max_floors = None
    sub_front_setback = None
    sub_side_setback = None
    sub_rear_setback = None

    if state.gdf is not None:
        matches = state.gdf[state.gdf['parcel_id'] == parcel_id]
        if not matches.empty:
            row = matches.iloc[0]
            if not district:
                district = str(row.get('primary_zoning', ''))
                article = str(row.get('article', ''))
                all_codes = str(row.get('all_zoning_codes', district))
                multi_zoning = bool(row.get('multi_zoning', False))
            subdistrict = str(row.get('zoning_subdistrict') or '')
            subdistrict_type = str(row.get('subdistrict_type') or '')
            subdistrict_use = str(row.get('subdistrict_use') or '')
            neighborhood = str(row.get('neighborhood_district') or row.get('districts') or '')
            in_gcod = bool(row.get('in_gcod', False))
            in_coastal_flood = bool(row.get('in_coastal_flood', False))
            sub_max_far = row.get('max_far')
            sub_max_height = row.get('max_height_ft')
            sub_max_floors = row.get('max_floors')
            sub_front_setback = row.get('front_setback_ft')
            sub_side_setback = row.get('side_setback_ft')
            sub_rear_setback = row.get('rear_setback_ft')
            sub_article = str(row.get('article_sub') or '')
            if sub_article:
                article = sub_article

    if not district:
        return None

    reqs_fallback = get_zoning_requirements(district)

    def _pick(sub_val, fallback_key):
        if sub_val is not None and not (isinstance(sub_val, float) and np.isnan(sub_val)):
            return sub_val
        return reqs_fallback.get(fallback_key)

    requirements = {
        "max_far": _pick(sub_max_far, 'max_far'),
        "max_height_ft": _pick(sub_max_height, 'max_height_ft'),
        "max_stories": _pick(sub_max_floors, 'max_stories'),
        "min_lot_sf": reqs_fallback.get('min_lot_sf'),
        "min_frontage_ft": reqs_fallback.get('min_frontage_ft'),
        "min_front_yard_ft": _pick(sub_front_setback, 'min_front_yard_ft'),
        "min_side_yard_ft": _pick(sub_side_setback, 'min_side_yard_ft'),
        "min_rear_yard_ft": _pick(sub_rear_setback, 'min_rear_yard_ft'),
        "max_lot_coverage_pct": reqs_fallback.get('max_lot_coverage_pct'),
        "parking_per_unit": reqs_fallback.get('parking_per_unit'),
    }

    overlay_districts = []
    if in_gcod:
        overlay_districts.append({
            "name": "Groundwater Conservation Overlay District (GCOD)",
            "code": "GCOD",
            "article": "32",
            "note": "This parcel is in the Groundwater Conservation Overlay District (GCOD). Additional permits and review may be required under Article 32.",
        })
    if in_coastal_flood:
        overlay_districts.append({
            "name": "Coastal Flood Resilience Overlay District",
            "code": "CFROD",
            "article": "25A",
            "note": "This parcel is in the Coastal Flood Resilience Overlay District. Flood-resistant design and elevated construction may be required.",
        })

    return {
        "parcel_id": parcel_id,
        "district": district,
        "article": article,
        "all_codes": all_codes,
        "multi_zoning": multi_zoning,
        "subdistrict": subdistrict,
        "subdistrict_type": subdistrict_type,
        "subdistrict_use": subdistrict_use,
        "neighborhood": neighborhood,
        "requirements": requirements,
        "allowed_uses": reqs_fallback.get('allowed_uses', []),
        "overlay_districts": overlay_districts,
        "data_source": "BPDA Zoning Subdistricts (official)" if subdistrict else "Lookup table (approximate)",
    }


@router.get("/zoning/districts", tags=["Zoning Analysis"])
def list_zoning_districts():
    """List all zoning districts with their dimensional requirements."""
    districts = []
    for code, reqs in ZONING_REQUIREMENTS.items():
        districts.append({
            "code": code,
            "name": reqs['district_name'],
            "article": reqs['article'],
            "description": reqs['description'],
            "max_far": reqs['max_far'],
            "max_height_ft": reqs['max_height_ft'],
            "max_stories": reqs.get('max_stories'),
            "allowed_uses": reqs['allowed_uses'],
        })
    return {"districts": districts, "total": len(districts)}


def _detect_parcel_issues(parcel_id: str, z: dict) -> dict:
    """
    Auto-detect zoning issues that exist at the PARCEL level — problems that
    apply regardless of what the developer proposes to build. Uses property
    assessment records (lot size) and zoning requirements.

    Returns dict with:
      - parcel_variances: list of variances this parcel will always need
      - parcel_violations: detailed violation descriptions
      - proposal_dependent: list of variance types that depend on the proposal
      - auto_filled_data: dict of data auto-filled from public records
    """
    reqs = z['requirements']
    parcel_variances = []
    parcel_violations = []
    auto_filled_data = {}

    # Try to get lot data from property assessment
    lot_size = None
    if state.parcel_addr_df is not None and 'parcel_id' in state.parcel_addr_df.columns:
        pa_match = state.parcel_addr_df[state.parcel_addr_df['parcel_id'] == parcel_id]
        if not pa_match.empty:
            pa_row = pa_match.iloc[0]
            for col in ['lot_size', 'lot_size_sf', 'land_sf', 'lot_area']:
                if col in pa_row.index:
                    ls = pa_row.get(col, 0)
                    if ls and pd.notna(ls) and float(ls) > 0:
                        lot_size = float(ls)
                        auto_filled_data['lot_size_sf'] = lot_size
                        break

    # Fallback: try ZBA dataset
    if lot_size is None and state.zba_df is not None and 'pa_parcel_id' in state.zba_df.columns:
        pa_match = state.zba_df[state.zba_df['pa_parcel_id'].astype(str) == parcel_id]
        if not pa_match.empty:
            ls = pa_match.iloc[0].get('lot_size_sf', 0)
            if ls and pd.notna(ls) and float(ls) > 0:
                lot_size = float(ls)
                auto_filled_data['lot_size_sf'] = lot_size

    # Check lot size vs minimum
    min_lot = reqs.get('min_lot_sf')
    if lot_size and min_lot and lot_size < min_lot:
        parcel_variances.append("lot_area")
        parcel_violations.append({
            "type": "lot_area",
            "requirement": f"Min lot: {min_lot:,} sf",
            "actual": f"Lot size: {lot_size:,.0f} sf",
            "deficit": f"{min_lot - lot_size:,.0f} sf under minimum",
            "source": "Boston property assessment records",
            "note": "This parcel's lot size is below the zoning minimum — a lot area variance will be required regardless of what you build.",
        })

    # Check lot frontage if data available
    lot_frontage = auto_filled_data.get('lot_frontage_ft')
    min_frontage = reqs.get('min_frontage_ft')
    if lot_frontage and min_frontage and lot_frontage < min_frontage:
        parcel_variances.append("lot_frontage")
        parcel_violations.append({
            "type": "lot_frontage",
            "requirement": f"Min frontage: {min_frontage} ft",
            "actual": f"Lot frontage: {lot_frontage} ft",
            "deficit": f"{min_frontage - lot_frontage:.0f} ft under minimum",
            "source": "Boston property assessment records",
            "note": "This parcel's frontage is below the zoning minimum — a lot frontage variance will be required regardless of what you build.",
        })

    # Identify proposal-dependent variance types (can't auto-detect)
    proposal_dependent = []
    proposal_dependent.append({"type": "far", "depends_on": "Your proposed floor area ratio (building size relative to lot)", "input_needed": "proposed_far"})
    proposal_dependent.append({"type": "height", "depends_on": "Your proposed building height and number of stories", "input_needed": "proposed_height_ft, proposed_stories"})
    proposal_dependent.append({"type": "parking", "depends_on": "Number of units and parking spaces provided", "input_needed": "proposed_units, parking_spaces"})
    proposal_dependent.append({"type": "setbacks", "depends_on": "Building position on the lot (front, side, rear distances)", "input_needed": "front_setback_ft, side_setback_ft, rear_setback_ft"})
    proposal_dependent.append({"type": "open_space", "depends_on": "Building footprint relative to lot area", "input_needed": "lot_coverage_pct"})
    proposal_dependent.append({"type": "conditional_use", "depends_on": "Whether your proposed use is allowed as-of-right", "input_needed": "proposed_use"})

    # If we couldn't auto-fill lot frontage, flag it
    if lot_frontage is None and min_frontage:
        proposal_dependent.append({"type": "lot_frontage", "depends_on": "Lot frontage measurement (not available in public records for this parcel)", "input_needed": "lot_frontage_ft"})

    return {
        "parcel_variances": parcel_variances,
        "parcel_violations": parcel_violations,
        "proposal_dependent": proposal_dependent,
        "auto_filled_data": auto_filled_data,
    }


@router.get("/zoning/{parcel_id}", tags=["Zoning Analysis"])
def zoning_analysis(parcel_id: str):
    """Full zoning analysis for a parcel — includes auto-detected parcel-level issues."""
    if parcel_id in ('check_compliance', 'full_analysis', 'districts'):
        raise HTTPException(status_code=400, detail=f"Use POST /zoning/{parcel_id} instead")
    z = _get_parcel_zoning(parcel_id)
    if z is None:
        raise HTTPException(status_code=404, detail=f"Parcel {parcel_id} not found")

    case_count = 0
    area_approval_rate = 0
    if state.zba_df is not None:
        area_cases = state.zba_df[state.zba_df['decision_clean'].notna()]
        district = z['district']
        if district:
            zone_cases = area_cases[area_cases.get('zoning', pd.Series(dtype=str)).fillna('').str.contains(district[:2], na=False)]
            if len(zone_cases) > 10:
                area_cases = zone_cases
        case_count = len(area_cases)
        area_approval_rate = float((area_cases['decision_clean'] == 'APPROVED').mean()) if case_count > 0 else 0

    # Auto-detect parcel-level zoning issues from public records
    parcel_issues = _detect_parcel_issues(parcel_id, z)

    return {
        "parcel_id": parcel_id,
        "zoning_subdistrict": z['subdistrict'],
        "subdistrict_type": z['subdistrict_type'],
        "subdistrict_use": z['subdistrict_use'],
        "neighborhood": z['neighborhood'],
        "zoning_district": z['district'],
        "article": z['article'],
        "all_zoning_codes": z['all_codes'],
        "multi_zoning": z['multi_zoning'],
        "dimensional_requirements": z['requirements'],
        "allowed_uses": z['allowed_uses'],
        "overlay_districts": z['overlay_districts'],
        "parcel_issues": {
            "auto_detected_variances": parcel_issues['parcel_variances'],
            "auto_detected_violations": parcel_issues['parcel_violations'],
            "proposal_dependent_checks": parcel_issues['proposal_dependent'],
            "data_sources": parcel_issues['auto_filled_data'],
            "summary": (
                f"This parcel has {len(parcel_issues['parcel_variances'])} variance(s) that will be required regardless of your proposal: {', '.join(parcel_issues['parcel_variances'])}. "
                f"An additional {len(parcel_issues['proposal_dependent'])} variance types depend on your specific proposal details."
                if parcel_issues['parcel_variances'] else
                f"No parcel-level issues detected from public records. {len(parcel_issues['proposal_dependent'])} variance types depend on your specific proposal details."
            ),
        },
        "area_zba_cases": case_count,
        "area_approval_rate": round(area_approval_rate, 3),
        "data_source": z['data_source'],
    }


@router.post("/zoning/check_compliance", tags=["Zoning Analysis"])
def zoning_compliance_check(payload: dict):
    """Check if a proposed project needs zoning relief."""
    parcel_id = payload.get('parcel_id')
    if not parcel_id:
        raise HTTPException(status_code=400, detail="parcel_id is required")

    z = _get_parcel_zoning(parcel_id)
    if z is None:
        # Parcel not in GIS shapefile — return graceful response instead of 404
        # Try to get zoning district from ZBA case history
        _fallback_district = ""
        if state.zba_df is not None and 'pa_parcel_id' in state.zba_df.columns:
            _fb_match = state.zba_df[state.zba_df['pa_parcel_id'].astype(str) == parcel_id]
            if not _fb_match.empty and 'zoning_district' in _fb_match.columns:
                _fb_zd = _fb_match.iloc[0].get('zoning_district', '')
                if pd.notna(_fb_zd):
                    _fallback_district = str(_fb_zd)
        return {
            "compliant": None,
            "violations": [],
            "variances_needed": [],
            "complexity": "unknown",
            "complexity_note": "Zoning dimensional data not available for this parcel. Compliance cannot be auto-checked — enter your variances manually below.",
            "auto_filled": [],
            "zoning_district": _fallback_district,
            "parcel_level_variances": {"types": [], "count": 0},
            "proposal_level_variances": {"types": [], "count": 0},
            "variance_historical_rates": {},
            "data_limited": True,
        }

    reqs = z['requirements']

    # Support both flat and nested format
    if 'proposal' in payload and isinstance(payload['proposal'], dict):
        proposal = dict(payload['proposal'])
        proposal['parcel_id'] = parcel_id
    else:
        proposal = dict(payload)

    # Field aliases
    if 'proposed_height_ft' not in proposal and 'proposed_height' in proposal:
        proposal['proposed_height_ft'] = proposal.pop('proposed_height')
    if 'proposed_stories' not in proposal and 'proposed_stories_count' in proposal:
        proposal['proposed_stories'] = proposal.pop('proposed_stories_count')
    if 'parking_spaces' not in proposal and 'proposed_parking_spaces' in proposal:
        proposal['parking_spaces'] = proposal.pop('proposed_parking_spaces')
    if 'parking_spaces' not in proposal and 'parking' in proposal:
        proposal['parking_spaces'] = proposal.pop('parking')
    if 'lot_frontage_ft' not in proposal and 'lot_frontage' in proposal:
        proposal['lot_frontage_ft'] = proposal.pop('lot_frontage')
    if 'proposed_far' not in proposal and 'far' in proposal:
        proposal['proposed_far'] = proposal.pop('far')

    # Auto-fill lot data
    auto_filled = []
    if state.parcel_addr_df is not None and ('lot_size_sf' not in proposal or 'lot_frontage_ft' not in proposal):
        pa_match = state.parcel_addr_df[state.parcel_addr_df['parcel_id'] == parcel_id] if 'parcel_id' in state.parcel_addr_df.columns else pd.DataFrame()
        if not pa_match.empty:
            pa_row = pa_match.iloc[0]
            if 'lot_size_sf' not in proposal:
                for col in ['lot_size', 'lot_size_sf', 'land_sf', 'lot_area']:
                    if col in pa_row.index:
                        ls = pa_row.get(col, 0)
                        if ls and pd.notna(ls) and float(ls) > 0:
                            proposal['lot_size_sf'] = float(ls)
                            auto_filled.append('lot_size_sf')
                            break
            if 'lot_frontage_ft' not in proposal:
                for col in ['lot_frontage', 'lot_frontage_ft', 'front']:
                    if col in pa_row.index:
                        lf = pa_row.get(col, 0)
                        if lf and pd.notna(lf) and float(lf) > 0:
                            proposal['lot_frontage_ft'] = float(lf)
                            auto_filled.append('lot_frontage_ft')
                            break

    if 'lot_size_sf' not in proposal and state.zba_df is not None:
        pa_match = state.zba_df[state.zba_df['pa_parcel_id'].astype(str) == parcel_id] if 'pa_parcel_id' in state.zba_df.columns else pd.DataFrame()
        if not pa_match.empty:
            ls = pa_match.iloc[0].get('lot_size_sf', 0)
            if ls and pd.notna(ls) and float(ls) > 0:
                proposal['lot_size_sf'] = float(ls)
                auto_filled.append('lot_size_sf')

    violations = []
    variances_needed = []
    compliant = True

    # Coerce all proposal values to floats (handles string input from JSON)
    proposed_far = safe_float(proposal.get('proposed_far', 0))
    proposed_height = safe_float(proposal.get('proposed_height_ft', 0))
    proposed_stories = safe_float(proposal.get('proposed_stories', 0))
    proposed_units = safe_float(proposal.get('proposed_units', 0))
    lot_size_input = safe_float(proposal.get('lot_size_sf', 0))
    frontage_input = safe_float(proposal.get('lot_frontage_ft', 0))
    lot_coverage = safe_float(proposal.get('lot_coverage_pct', 0))
    front_setback = safe_float(proposal.get('front_setback_ft', 0))
    side_setback = safe_float(proposal.get('side_setback_ft', 0))
    rear_setback = safe_float(proposal.get('rear_setback_ft', 0))

    # FAR check
    max_far = reqs.get('max_far')
    if proposed_far and max_far and proposed_far > max_far:
        compliant = False
        violations.append({"type": "far", "requirement": f"Max FAR: {max_far}", "proposed": f"Proposed FAR: {proposed_far}", "excess": f"{((proposed_far / max_far) - 1) * 100:.0f}% over limit"})
        variances_needed.append("far")

    # Height check
    max_height = reqs.get('max_height_ft')
    if proposed_height and max_height and proposed_height > max_height:
        compliant = False
        violations.append({"type": "height", "requirement": f"Max height: {max_height} ft", "proposed": f"Proposed height: {proposed_height} ft", "excess": f"{proposed_height - max_height:.0f} ft over limit"})
        variances_needed.append("height")

    # Stories check
    max_stories = reqs.get('max_stories')
    if proposed_stories and max_stories and proposed_stories > max_stories:
        compliant = False
        violations.append({"type": "height", "requirement": f"Max stories: {max_stories}", "proposed": f"Proposed stories: {proposed_stories}", "excess": f"{proposed_stories - max_stories:.1f} stories over limit"})
        if "height" not in variances_needed:
            variances_needed.append("height")

    # Lot size check — auto-filled from public records OR user-provided
    lot_size = lot_size_input
    min_lot = reqs.get('min_lot_sf')
    if lot_size and min_lot and lot_size < min_lot:
        compliant = False
        violation = {"type": "lot_area", "requirement": f"Min lot: {min_lot:,} sf", "proposed": f"Lot size: {lot_size:,.0f} sf", "deficit": f"{min_lot - lot_size:,.0f} sf under minimum"}
        if 'lot_size_sf' in auto_filled:
            violation["source"] = "Boston property assessment records"
            violation["note"] = "This parcel's lot size is below the zoning minimum — a lot area variance will be required regardless of what you build."
        violations.append(violation)
        variances_needed.append("lot_area")

    # Frontage check — auto-filled from public records OR user-provided
    frontage = frontage_input
    min_frontage = reqs.get('min_frontage_ft')
    if frontage and min_frontage and frontage < min_frontage:
        compliant = False
        violation = {"type": "lot_frontage", "requirement": f"Min frontage: {min_frontage} ft", "proposed": f"Lot frontage: {frontage} ft", "deficit": f"{min_frontage - frontage:.0f} ft under minimum"}
        if 'lot_frontage_ft' in auto_filled:
            violation["source"] = "Boston property assessment records"
            violation["note"] = "This parcel's frontage is below the zoning minimum — a lot frontage variance will be required regardless of what you build."
        violations.append(violation)
        variances_needed.append("lot_frontage")

    # Setback checks
    _setback_vals = {'front_setback_ft': front_setback, 'side_setback_ft': side_setback, 'rear_setback_ft': rear_setback}
    for setback_type, req_key in [('front_setback_ft', 'min_front_yard_ft'), ('side_setback_ft', 'min_side_yard_ft'), ('rear_setback_ft', 'min_rear_yard_ft')]:
        setback_val = _setback_vals[setback_type]
        min_val = reqs.get(req_key)
        if setback_val is not None and min_val and setback_val < min_val:
            compliant = False
            stype = setback_type.replace('_ft', '').replace('_', ' ')
            violations.append({"type": stype, "requirement": f"Min {stype}: {min_val} ft", "proposed": f"Proposed {stype}: {setback_val} ft", "deficit": f"{min_val - setback_val:.0f} ft under minimum"})
            variances_needed.append(stype.replace(' ', '_'))

    # Parking check
    parking_spaces = safe_float(proposal.get('parking_spaces'), default=None)
    parking_per_unit = reqs.get('parking_per_unit')
    if proposed_units and parking_spaces is not None and parking_per_unit:
        required_parking = int(proposed_units * parking_per_unit)
        if parking_spaces < required_parking:
            compliant = False
            violations.append({"type": "parking", "requirement": f"Required: {required_parking} spaces ({parking_per_unit} per unit)", "proposed": f"Provided: {parking_spaces} spaces", "deficit": f"{required_parking - parking_spaces} spaces short"})
            variances_needed.append("parking")

    # Lot coverage (use safe_float value from line 384)
    max_coverage = reqs.get('max_lot_coverage_pct')
    if lot_coverage and max_coverage and lot_coverage > max_coverage:
        compliant = False
        violations.append({"type": "open_space", "requirement": f"Max lot coverage: {max_coverage}%", "proposed": f"Proposed coverage: {lot_coverage}%", "excess": f"{lot_coverage - max_coverage:.0f}% over limit (insufficient open space)"})
        variances_needed.append("open_space")

    # Conditional use check
    proposed_use = proposal.get('proposed_use', '').lower()
    allowed_uses = z.get('allowed_uses', [])
    if proposed_use and allowed_uses:
        use_allowed = any(proposed_use in u.lower() for u in allowed_uses)
        if not use_allowed:
            compliant = False
            violations.append({"type": "conditional_use", "requirement": f"Allowed uses: {', '.join(allowed_uses[:5])}", "proposed": f"Proposed use: {proposed_use}", "excess": "Use not permitted as-of-right"})
            variances_needed.append("conditional_use")

    # Complexity
    num_violations = len(variances_needed)
    if num_violations == 0:
        complexity = "low"
        complexity_note = "Project appears to comply with zoning requirements. May not need ZBA relief."
    elif num_violations <= 2:
        complexity = "moderate"
        complexity_note = f"Project needs {num_violations} variance(s). Common for Boston ZBA."
    else:
        complexity = "high"
        complexity_note = f"Project needs {num_violations} variances. More complex ZBA case."

    overlay_warnings = [overlay['note'] for overlay in z.get('overlay_districts', [])]

    # Classify variances into parcel-level (auto-detected from records) vs proposal-level
    parcel_level_variances = []
    proposal_level_variances = []
    parcel_level_types = set()
    for v in violations:
        is_parcel_level = v.get('source') == 'Boston property assessment records'
        if is_parcel_level:
            parcel_level_variances.append(v)
            parcel_level_types.add(v['type'])
        else:
            proposal_level_variances.append(v)

    result = {
        "compliant": compliant,
        "parcel_id": parcel_id,
        "zoning_subdistrict": z['subdistrict'],
        "subdistrict_type": z['subdistrict_type'],
        "neighborhood": z['neighborhood'],
        "zoning_district": z['district'],
        "article": z['article'],
        "requirements": reqs,
        "violations": violations,
        "variances_needed": variances_needed,
        "num_variances_needed": len(variances_needed),
        "parcel_level_variances": {
            "types": [v for v in variances_needed if v in parcel_level_types],
            "violations": parcel_level_variances,
            "note": "These variances are required regardless of what you propose — they stem from the parcel's physical characteristics vs zoning minimums." if parcel_level_variances else None,
        },
        "proposal_level_variances": {
            "types": [v for v in variances_needed if v not in parcel_level_types],
            "violations": proposal_level_variances,
            "note": "These variances are triggered by your specific proposal exceeding zoning limits." if proposal_level_variances else None,
        },
        "complexity": complexity,
        "complexity_note": complexity_note,
        "overlay_districts": z.get('overlay_districts', []),
        "overlay_warnings": overlay_warnings,
        "data_source": z['data_source'],
        "auto_filled": auto_filled,
        "lot_size_sf": proposal.get('lot_size_sf'),
        "lot_frontage_ft": proposal.get('lot_frontage_ft'),
    }

    # Historical variance rates — ward-specific when possible, city-wide fallback
    if state.zba_df is not None and variances_needed:
        var_approval_rates = {}
        df_with_dec = state.zba_df[state.zba_df['decision_clean'].notna()].copy()

        # Determine ward for this parcel
        _parcel_ward = None
        if 'ward' in df_with_dec.columns and 'pa_parcel_id' in df_with_dec.columns:
            _ward_match = df_with_dec[df_with_dec['pa_parcel_id'].astype(str) == parcel_id]
            if not _ward_match.empty:
                _w = _ward_match.iloc[0].get('ward')
                if pd.notna(_w):
                    _parcel_ward = _w

        if 'variance_types' in df_with_dec.columns:
            vt_series = df_with_dec['variance_types'].fillna('')

            # Filter to ward if known
            ward_df = None
            if _parcel_ward is not None and 'ward' in df_with_dec.columns:
                ward_df = df_with_dec[df_with_dec['ward'] == _parcel_ward]
                ward_vt_series = ward_df['variance_types'].fillna('') if not ward_df.empty else None
            else:
                ward_vt_series = None

            for var_type in variances_needed:
                pattern = rf'(?:^|,)\s*{var_type}\s*(?:,|$)'
                # Try ward-specific first
                if ward_df is not None and ward_vt_series is not None and not ward_df.empty:
                    w_mask = ward_vt_series.str.contains(pattern, na=False, regex=True)
                    w_cases = ward_df[w_mask]
                    if len(w_cases) >= 3:  # Require minimum cases for ward-level stats
                        rate = float((w_cases['decision_clean'] == 'APPROVED').mean())
                        ward_str = str(int(_parcel_ward)) if float(_parcel_ward) == int(_parcel_ward) else str(_parcel_ward)
                        var_approval_rates[var_type] = {
                            "approval_rate": round(rate, 3),
                            "total_cases": len(w_cases),
                            "source": f"Ward {ward_str}",
                            "note": f"Based on {len(w_cases)} Ward {ward_str} ZBA cases involving {var_type} variances"
                        }
                        continue
                # Fall back to city-wide
                mask = vt_series.str.contains(pattern, na=False, regex=True)
                var_cases = df_with_dec[mask]
                if len(var_cases) > 0:
                    rate = float((var_cases['decision_clean'] == 'APPROVED').mean())
                    var_approval_rates[var_type] = {
                        "approval_rate": round(rate, 3),
                        "total_cases": len(var_cases),
                        "source": "city-wide",
                        "note": f"Based on {len(var_cases)} city-wide ZBA cases involving {var_type} variances"
                    }
        result['variance_historical_rates'] = var_approval_rates

    return result


@router.post("/zoning/full_analysis", tags=["Zoning Analysis", "Prediction"])
def full_zoning_analysis(payload: dict):
    """Complete zoning analysis workflow — all questions in one API call."""
    from api.routes.prediction import analyze_proposal

    parcel_id = payload.get('parcel_id')
    if not parcel_id:
        raise HTTPException(status_code=400, detail="parcel_id is required")

    try:
        zoning_info = zoning_analysis(parcel_id)
    except HTTPException:
        raise HTTPException(status_code=404, detail=f"Parcel {parcel_id} not found")

    compliance = zoning_compliance_check(payload)

    prediction_input = dict(payload)
    prediction_input['variances'] = compliance.get('variances_needed', [])
    if 'ward' not in prediction_input:
        if state.gdf is not None:
            matches = state.gdf[state.gdf['parcel_id'] == parcel_id]
            if not matches.empty and 'ward' in matches.columns:
                prediction_input['ward'] = str(matches.iloc[0].get('ward', ''))

    try:
        prediction = analyze_proposal(prediction_input)
    except Exception as e:
        prediction = {"error": str(e), "probability": None}

    return {
        "parcel_id": parcel_id,
        "zoning": {
            "subdistrict": zoning_info.get('zoning_subdistrict', ''),
            "subdistrict_type": zoning_info.get('subdistrict_type', ''),
            "neighborhood": zoning_info.get('neighborhood', ''),
            "district": zoning_info['zoning_district'],
            "article": zoning_info['article'],
            "allowed_uses": zoning_info['allowed_uses'],
        },
        "requirements": zoning_info['dimensional_requirements'],
        "compliance": {
            "compliant": compliance['compliant'],
            "variances_needed": compliance['variances_needed'],
            "num_variances": compliance['num_variances_needed'],
            "violations": compliance['violations'],
            "parcel_level_variances": compliance.get('parcel_level_variances'),
            "proposal_level_variances": compliance.get('proposal_level_variances'),
            "variance_historical_rates": compliance.get('variance_historical_rates'),
            "overlay_warnings": compliance.get('overlay_warnings', []),
            "auto_filled": compliance.get('auto_filled'),
        },
        "complexity": {
            "level": compliance['complexity'],
            "note": compliance['complexity_note'],
            "area_cases": zoning_info['area_zba_cases'],
            "area_approval_rate": zoning_info['area_approval_rate'],
        },
        "parcel_issues": zoning_info.get('parcel_issues'),
        "overlay_districts": zoning_info.get('overlay_districts', []),
        "prediction": prediction,
        "data_source": zoning_info.get('data_source', ''),
        "disclaimer": "This analysis is for informational purposes only. Always verify zoning requirements with the Boston Inspectional Services Department and consult a licensed zoning attorney before filing with the ZBA.",
    }


@router.post("/variance_analysis", tags=["Zoning Analysis"])
def variance_analysis(payload: dict):
    """Historical variance approval rates — the money question."""
    if state.zba_df is None:
        raise HTTPException(status_code=500, detail="ZBA data not loaded")

    variances = payload.get('variances', [])
    ward = payload.get('ward')
    has_attorney = payload.get('has_attorney', False)
    num_proposed_variances = len(variances) if variances else payload.get('num_variances', 0)

    df = state.zba_df[state.zba_df['decision_clean'].notna()].copy()
    vt = df['variance_types'].fillna('')

    overall_rate = float((df['decision_clean'] == 'APPROVED').mean())
    overall_count = len(df)

    if variances:
        mask = pd.Series(True, index=df.index)
        for v in variances:
            mask = mask & vt.str.contains(v.lower(), na=False)
        combo_cases = df[mask]
    else:
        combo_cases = df

    combo_count = len(combo_cases)
    combo_approved = int((combo_cases['decision_clean'] == 'APPROVED').sum())
    combo_denied = combo_count - combo_approved
    combo_rate = float(combo_approved / combo_count) if combo_count > 0 else overall_rate

    ward_rate = None
    ward_count = 0
    ward_note = ""
    if ward and combo_count > 0:
        try:
            ward_float = float(ward)
            ward_cases = combo_cases[combo_cases['ward'] == ward_float]
            ward_count = len(ward_cases)
            if ward_count >= 3:
                ward_rate = float((ward_cases['decision_clean'] == 'APPROVED').mean())
                ward_note = f"In Ward {int(ward_float)}, {ward_count} cases had a {ward_rate:.0%} approval rate"
            else:
                ward_note = f"Only {ward_count} case(s) in Ward {int(ward_float)} — insufficient data"
        except (ValueError, TypeError):
            logger.debug("Invalid ward value in variance_analysis: %s", ward)

    attorney_effect = {}
    if combo_count > 0 and 'has_attorney' in combo_cases.columns:
        with_atty = combo_cases[combo_cases['has_attorney'] == 1]
        without_atty = combo_cases[combo_cases['has_attorney'] == 0]
        if len(with_atty) >= 3 and len(without_atty) >= 3:
            rate_with = float((with_atty['decision_clean'] == 'APPROVED').mean())
            rate_without = float((without_atty['decision_clean'] == 'APPROVED').mean())
            attorney_effect = {
                "with_attorney": {"rate": round(rate_with, 3), "cases": len(with_atty)},
                "without_attorney": {"rate": round(rate_without, 3), "cases": len(without_atty)},
                "difference": round(rate_with - rate_without, 3),
            }

    variance_count_rates = []
    if 'num_variances' in df.columns:
        for nv in range(1, 8):
            nv_cases = df[df['num_variances'] == nv]
            if len(nv_cases) >= 10:
                rate = float((nv_cases['decision_clean'] == 'APPROVED').mean())
                variance_count_rates.append({"num_variances": nv, "approval_rate": round(rate, 3), "cases": len(nv_cases)})

    per_variance_rates = {}
    if variances:
        for v in variances:
            v_cases = df[vt.str.contains(v.lower(), na=False)]
            if len(v_cases) > 0:
                rate = float((v_cases['decision_clean'] == 'APPROVED').mean())
                per_variance_rates[v] = {"approval_rate": round(rate, 3), "cases": len(v_cases)}

    denial_factors = []
    if combo_count > 0 and combo_denied > 0:
        denied = combo_cases[combo_cases['decision_clean'] == 'DENIED']
        approved = combo_cases[combo_cases['decision_clean'] == 'APPROVED']
        if 'num_variances' in df.columns and len(denied) >= 3:
            d_avg = denied['num_variances'].mean()
            a_avg = approved['num_variances'].mean()
            if d_avg > a_avg + 0.5:
                denial_factors.append(f"Denied cases averaged {d_avg:.1f} variances vs {a_avg:.1f} for approved")
        if 'has_attorney' in df.columns and len(denied) >= 3:
            d_atty = denied['has_attorney'].mean()
            a_atty = approved['has_attorney'].mean()
            if a_atty > d_atty + 0.1:
                denial_factors.append(f"Only {d_atty:.0%} of denied had attorney vs {a_atty:.0%} of approved")

    variance_list = ', '.join(variances) if variances else 'unspecified variances'
    if combo_count >= 10:
        headline = f"Based on {combo_count} recent ZBA cases requesting {variance_list}, the approval rate is {combo_rate:.0%}."
    elif combo_count >= 3:
        headline = f"Based on {combo_count} cases with {variance_list} (limited data), the approval rate is {combo_rate:.0%}."
    else:
        headline = f"Very few cases ({combo_count}) match your exact combination. Using overall rate of {overall_rate:.0%}."

    details = []
    if ward_note:
        details.append(ward_note)
    if attorney_effect:
        diff = attorney_effect['difference']
        if diff > 0.05:
            atty_word = "increases" if has_attorney else "would increase"
            details.append(f"Attorney representation {atty_word} approval odds by {diff:.0%}")
    if num_proposed_variances > 0:
        matching_nv = [r for r in variance_count_rates if r['num_variances'] == num_proposed_variances]
        if matching_nv:
            details.append(f"Cases with exactly {num_proposed_variances} variance(s) have a {matching_nv[0]['approval_rate']:.0%} approval rate")
    for factor in denial_factors:
        details.append(factor)

    return {
        "question": f"How likely will a proposal requiring {variance_list} pass?",
        "headline": headline,
        "details": details,
        "data": {
            "overall": {"rate": round(overall_rate, 3), "cases": overall_count},
            "your_combination": {"rate": round(combo_rate, 3), "cases": combo_count, "approved": combo_approved, "denied": combo_denied},
            "ward_specific": {"rate": round(ward_rate, 3) if ward_rate is not None else None, "cases": ward_count, "ward": ward} if ward else None,
            "attorney_effect": attorney_effect or None,
            "per_variance": per_variance_rates,
            "by_variance_count": variance_count_rates,
        },
        "recommendation": (
            "Strong historical precedent for approval." if combo_rate >= 0.85 else
            "Moderate risk — consider strengthening your application." if combo_rate >= 0.65 else
            "Higher risk — consult an experienced zoning attorney."
        ),
        "disclaimer": "Based on historical ZBA decisions 2020-2026. Past decisions do not guarantee future outcomes.",
    }
