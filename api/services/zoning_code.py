"""
Boston Zoning Code Reference — Dimensional Requirements by District

Source: Boston Zoning Code Articles 13, 50-69+
Reference: https://library.municode.com/ma/boston/codes/redevelopment_authority
           https://www.bostonplans.org/planning-zoning/zoning-code

This maps zoning district codes to their dimensional requirements.
Districts are organized by their governing article.

IMPORTANT: This is a reference implementation. The actual zoning code
has hundreds of subdistrict variations and overlay districts. This covers
the major residential and commercial districts that appear most frequently
in ZBA cases. Always verify against the official code for production use.
"""

# Zoning district dimensional requirements
# Keys: max_far, max_height_ft, min_lot_sf, min_frontage_ft,
#        min_front_yard_ft, min_side_yard_ft, min_rear_yard_ft,
#        max_lot_coverage_pct, parking_per_unit, allowed_uses,
#        article, district_name, description

ZONING_REQUIREMENTS = {
    # === ARTICLE 53 — EAST BOSTON NEIGHBORHOOD DISTRICT ===
    "3A": {
        "article": 53, "district_name": "East Boston - Residential",
        "description": "Three-family residential, Jeffries Point / Eagle Hill",
        "max_far": 1.0, "max_height_ft": 35, "max_stories": 3,
        "min_lot_sf": 3000, "min_frontage_ft": 30,
        "min_front_yard_ft": 10, "min_side_yard_ft": 5, "min_rear_yard_ft": 20,
        "max_lot_coverage_pct": 60, "parking_per_unit": 1.0,
        "allowed_uses": ["1-3 family residential", "home occupation"],
    },
    "3A-3C": {
        "article": 53, "district_name": "East Boston - Residential Mixed",
        "description": "Mixed residential subdistricts, East Boston",
        "max_far": 1.5, "max_height_ft": 40, "max_stories": 3,
        "min_lot_sf": 2500, "min_frontage_ft": 25,
        "min_front_yard_ft": 10, "min_side_yard_ft": 5, "min_rear_yard_ft": 20,
        "max_lot_coverage_pct": 65, "parking_per_unit": 1.0,
        "allowed_uses": ["1-3 family residential", "home occupation", "small retail"],
    },

    # === ARTICLE 65 — JAMAICA PLAIN / MISSION HILL ===
    "5A-5E": {
        "article": 65, "district_name": "Jamaica Plain / Mission Hill - Residential",
        "description": "Mixed residential, Jamaica Plain and Mission Hill area",
        "max_far": 1.0, "max_height_ft": 35, "max_stories": 3,
        "min_lot_sf": 4000, "min_frontage_ft": 40,
        "min_front_yard_ft": 15, "min_side_yard_ft": 10, "min_rear_yard_ft": 25,
        "max_lot_coverage_pct": 50, "parking_per_unit": 1.0,
        "allowed_uses": ["1-3 family residential", "home occupation", "community facility"],
    },

    # === ARTICLE 56 — SOUTH END ===
    "11A-11E": {
        "article": 56, "district_name": "South End Neighborhood",
        "description": "Mixed use, South End brownstone district",
        "max_far": 2.0, "max_height_ft": 55, "max_stories": 5,
        "min_lot_sf": 2000, "min_frontage_ft": 20,
        "min_front_yard_ft": 0, "min_side_yard_ft": 0, "min_rear_yard_ft": 15,
        "max_lot_coverage_pct": 75, "parking_per_unit": 0.75,
        "allowed_uses": ["multi-family residential", "retail", "office", "restaurant", "mixed-use"],
    },

    # === ARTICLE 69 — DORCHESTER ===
    "12": {
        "article": 69, "district_name": "Dorchester Neighborhood",
        "description": "Residential, Dorchester area",
        "max_far": 1.0, "max_height_ft": 35, "max_stories": 3,
        "min_lot_sf": 5000, "min_frontage_ft": 50,
        "min_front_yard_ft": 15, "min_side_yard_ft": 10, "min_rear_yard_ft": 25,
        "max_lot_coverage_pct": 50, "parking_per_unit": 1.5,
        "allowed_uses": ["1-3 family residential", "home occupation"],
    },

    # === ARTICLE 50 — ROXBURY ===
    "6A/6B/6C": {
        "article": 50, "district_name": "Roxbury Neighborhood",
        "description": "Mixed residential/commercial, Roxbury",
        "max_far": 1.5, "max_height_ft": 45, "max_stories": 4,
        "min_lot_sf": 3000, "min_frontage_ft": 30,
        "min_front_yard_ft": 10, "min_side_yard_ft": 5, "min_rear_yard_ft": 20,
        "max_lot_coverage_pct": 60, "parking_per_unit": 1.0,
        "allowed_uses": ["1-4 family residential", "retail", "community facility"],
    },

    # === ARTICLE 55 — SOUTH BOSTON ===
    "9A-9C": {
        "article": 55, "district_name": "South Boston Neighborhood",
        "description": "Residential/mixed, South Boston",
        "max_far": 1.5, "max_height_ft": 45, "max_stories": 4,
        "min_lot_sf": 3000, "min_frontage_ft": 30,
        "min_front_yard_ft": 10, "min_side_yard_ft": 5, "min_rear_yard_ft": 20,
        "max_lot_coverage_pct": 60, "parking_per_unit": 1.0,
        "allowed_uses": ["1-3 family residential", "home occupation", "retail"],
    },

    # === ARTICLE 60 — ALLSTON/BRIGHTON ===
    "8ABC": {
        "article": 60, "district_name": "Allston-Brighton Neighborhood",
        "description": "Mixed residential, Allston/Brighton",
        "max_far": 1.0, "max_height_ft": 35, "max_stories": 3,
        "min_lot_sf": 5000, "min_frontage_ft": 50,
        "min_front_yard_ft": 15, "min_side_yard_ft": 10, "min_rear_yard_ft": 25,
        "max_lot_coverage_pct": 50, "parking_per_unit": 1.5,
        "allowed_uses": ["1-3 family residential", "home occupation"],
    },

    # === ARTICLE 67 — HYDE PARK ===
    "10A-10B": {
        "article": 67, "district_name": "Hyde Park Neighborhood",
        "description": "Residential, Hyde Park",
        "max_far": 0.8, "max_height_ft": 35, "max_stories": 2.5,
        "min_lot_sf": 6000, "min_frontage_ft": 50,
        "min_front_yard_ft": 20, "min_side_yard_ft": 10, "min_rear_yard_ft": 30,
        "max_lot_coverage_pct": 40, "parking_per_unit": 2.0,
        "allowed_uses": ["1-2 family residential", "home occupation"],
    },

    # === ARTICLE 68 — SOUTH BOSTON WATERFRONT ===
    "4F": {
        "article": 68, "district_name": "South Boston Waterfront",
        "description": "Mixed-use waterfront development district",
        "max_far": 3.0, "max_height_ft": 155, "max_stories": 15,
        "min_lot_sf": 5000, "min_frontage_ft": 50,
        "min_front_yard_ft": 0, "min_side_yard_ft": 0, "min_rear_yard_ft": 0,
        "max_lot_coverage_pct": 80, "parking_per_unit": 0.5,
        "allowed_uses": ["multi-family residential", "office", "retail", "hotel", "lab/R&D", "mixed-use"],
    },

    # === ARTICLE 62 — MATTAPAN ===
    "2E": {
        "article": 62, "district_name": "Mattapan Neighborhood",
        "description": "Residential, Mattapan",
        "max_far": 0.8, "max_height_ft": 35, "max_stories": 2.5,
        "min_lot_sf": 6000, "min_frontage_ft": 50,
        "min_front_yard_ft": 20, "min_side_yard_ft": 10, "min_rear_yard_ft": 30,
        "max_lot_coverage_pct": 40, "parking_per_unit": 1.5,
        "allowed_uses": ["1-3 family residential", "home occupation"],
    },

    # === ARTICLE 64 — WEST ROXBURY ===
    "1P": {
        "article": 64, "district_name": "West Roxbury Neighborhood",
        "description": "Single/two-family residential, West Roxbury",
        "max_far": 0.5, "max_height_ft": 35, "max_stories": 2.5,
        "min_lot_sf": 7500, "min_frontage_ft": 60,
        "min_front_yard_ft": 20, "min_side_yard_ft": 10, "min_rear_yard_ft": 30,
        "max_lot_coverage_pct": 35, "parking_per_unit": 2.0,
        "allowed_uses": ["1-2 family residential", "home occupation"],
    },

    # === DOWNTOWN / BACK BAY (Article 13 base) ===
    "1": {
        "article": 13, "district_name": "Downtown / Waterfront Core",
        "description": "High-density mixed-use, downtown Boston",
        "max_far": 10.0, "max_height_ft": 400, "max_stories": 40,
        "min_lot_sf": 0, "min_frontage_ft": 0,
        "min_front_yard_ft": 0, "min_side_yard_ft": 0, "min_rear_yard_ft": 0,
        "max_lot_coverage_pct": 100, "parking_per_unit": 0.25,
        "allowed_uses": ["residential", "office", "retail", "hotel", "institutional", "mixed-use"],
    },

    # === ARTICLE 59 — ROSLINDALE ===
    "6D": {
        "article": 59, "district_name": "Roslindale Neighborhood",
        "description": "Residential, Roslindale Village area",
        "max_far": 1.0, "max_height_ft": 35, "max_stories": 3,
        "min_lot_sf": 5000, "min_frontage_ft": 50,
        "min_front_yard_ft": 15, "min_side_yard_ft": 10, "min_rear_yard_ft": 25,
        "max_lot_coverage_pct": 50, "parking_per_unit": 1.0,
        "allowed_uses": ["1-3 family residential", "home occupation", "small retail"],
    },
}

# Fallback for unknown districts — conservative defaults
DEFAULT_REQUIREMENTS = {
    "article": None, "district_name": "Unknown District",
    "description": "District not in lookup table — verify against official zoning code",
    "max_far": 1.0, "max_height_ft": 35, "max_stories": 3,
    "min_lot_sf": 5000, "min_frontage_ft": 50,
    "min_front_yard_ft": 15, "min_side_yard_ft": 10, "min_rear_yard_ft": 25,
    "max_lot_coverage_pct": 50, "parking_per_unit": 1.0,
    "allowed_uses": ["residential"],
}


def get_zoning_requirements(district_code: str) -> dict:
    """Look up dimensional requirements for a zoning district."""
    if not district_code:
        return DEFAULT_REQUIREMENTS

    # Exact match
    if district_code in ZONING_REQUIREMENTS:
        return ZONING_REQUIREMENTS[district_code]

    # Try prefix match (e.g., "3A" matches "3A-3C")
    for key, reqs in ZONING_REQUIREMENTS.items():
        if district_code.startswith(key.split("-")[0]) or key.startswith(district_code.split("-")[0]):
            return reqs

    return DEFAULT_REQUIREMENTS


def check_compliance(district_code: str, proposal: dict) -> dict:
    """
    Check if a proposed project complies with zoning requirements.

    Args:
        district_code: Zoning district code (e.g., "3A-3C")
        proposal: dict with keys like:
            - proposed_far (float)
            - proposed_height_ft (int)
            - proposed_stories (int)
            - proposed_units (int)
            - lot_size_sf (float)
            - lot_frontage_ft (float)
            - proposed_use (str)
            - parking_spaces (int)

    Returns:
        dict with compliance results and needed variances
    """
    reqs = get_zoning_requirements(district_code)
    violations = []
    variances_needed = []
    compliant = True

    # FAR check
    proposed_far = proposal.get('proposed_far', 0)
    if proposed_far and proposed_far > reqs['max_far']:
        compliant = False
        violations.append({
            "type": "far",
            "requirement": f"Max FAR: {reqs['max_far']}",
            "proposed": f"Proposed FAR: {proposed_far}",
            "excess": f"{((proposed_far / reqs['max_far']) - 1) * 100:.0f}% over limit",
        })
        variances_needed.append("far")

    # Height check
    proposed_height = proposal.get('proposed_height_ft', 0)
    if proposed_height and proposed_height > reqs['max_height_ft']:
        compliant = False
        violations.append({
            "type": "height",
            "requirement": f"Max height: {reqs['max_height_ft']} ft",
            "proposed": f"Proposed height: {proposed_height} ft",
            "excess": f"{proposed_height - reqs['max_height_ft']} ft over limit",
        })
        variances_needed.append("height")

    # Stories check
    proposed_stories = proposal.get('proposed_stories', 0)
    max_stories = reqs.get('max_stories', 99)
    if proposed_stories and proposed_stories > max_stories:
        compliant = False
        violations.append({
            "type": "height",
            "requirement": f"Max stories: {max_stories}",
            "proposed": f"Proposed stories: {proposed_stories}",
            "excess": f"{proposed_stories - max_stories} stories over limit",
        })
        if "height" not in variances_needed:
            variances_needed.append("height")

    # Lot size check
    lot_size = proposal.get('lot_size_sf', 0)
    if lot_size and lot_size < reqs['min_lot_sf']:
        compliant = False
        violations.append({
            "type": "lot_area",
            "requirement": f"Min lot: {reqs['min_lot_sf']:,} sf",
            "proposed": f"Lot size: {lot_size:,.0f} sf",
            "deficit": f"{reqs['min_lot_sf'] - lot_size:,.0f} sf under minimum",
        })
        variances_needed.append("lot_area")

    # Frontage check
    frontage = proposal.get('lot_frontage_ft', 0)
    if frontage and frontage < reqs['min_frontage_ft']:
        compliant = False
        violations.append({
            "type": "lot_frontage",
            "requirement": f"Min frontage: {reqs['min_frontage_ft']} ft",
            "proposed": f"Lot frontage: {frontage} ft",
            "deficit": f"{reqs['min_frontage_ft'] - frontage:.0f} ft under minimum",
        })
        variances_needed.append("lot_frontage")

    # Parking check
    proposed_units = proposal.get('proposed_units', 0)
    parking_spaces = proposal.get('parking_spaces')
    if proposed_units and parking_spaces is not None:
        required_parking = int(proposed_units * reqs['parking_per_unit'])
        if parking_spaces < required_parking:
            compliant = False
            violations.append({
                "type": "parking",
                "requirement": f"Required: {required_parking} spaces ({reqs['parking_per_unit']} per unit)",
                "proposed": f"Provided: {parking_spaces} spaces",
                "deficit": f"{required_parking - parking_spaces} spaces short",
            })
            variances_needed.append("parking")

    # Complexity assessment
    num_violations = len(violations)
    if num_violations == 0:
        complexity = "low"
        complexity_note = "Project appears to comply with zoning requirements. May not need ZBA relief."
    elif num_violations <= 2:
        complexity = "moderate"
        complexity_note = f"Project needs {num_violations} variance(s). Common for Boston ZBA — most projects with 1-2 variances are approved."
    else:
        complexity = "high"
        complexity_note = f"Project needs {num_violations} variances. More complex ZBA case — consider reducing scope or hiring experienced zoning attorney."

    return {
        "compliant": compliant,
        "district": district_code,
        "district_name": reqs['district_name'],
        "article": reqs['article'],
        "requirements": reqs,
        "violations": violations,
        "variances_needed": variances_needed,
        "num_variances_needed": len(variances_needed),
        "complexity": complexity,
        "complexity_note": complexity_note,
    }
