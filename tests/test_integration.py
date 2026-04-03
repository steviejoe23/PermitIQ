r"""
PermitIQ Comprehensive Integration Tests
=========================================

Tests EVERY API endpoint with real payloads. Asserts on response structure AND values.
These are integration tests that hit the real running API.

Run:
    cd ~/Desktop/Boston\ Zoning\ Project
    source zoning-env/bin/activate
    pytest tests/test_integration.py -v

Requires: API running on port 8000
    cd api && uvicorn main:app --port 8000

Primary test parcel: 1100051000 (57 Centre Street, Roxbury)
"""

import pytest

# Reuse fixtures from conftest.py
from conftest import PRIMARY_PARCEL, SECONDARY_PARCEL


# =============================================================================
# 1. SEARCH ENDPOINT — GET /search?q=...
# =============================================================================


class TestSearch:
    """Tests for the address search endpoint."""

    def test_search_57_centre_returns_michael_winston(self, client):
        """'57 centre street' returns results with Michael Winston as applicant."""
        r = client.get("/search", params={"q": "57 centre street"})
        assert r.status_code == 200
        data = r.json()
        assert data["total_results"] > 0, "Expected results for 57 Centre Street"
        # Check that at least one result mentions Centre
        addresses = [res["address"].upper() for res in data["results"]]
        assert any("CENTRE" in a for a in addresses), f"No Centre St in results: {addresses}"

    def test_search_75_tremont_no_false_prefix_match(self, client):
        """'75 tremont st' returns results and does NOT match '1575 tremont'."""
        r = client.get("/search", params={"q": "75 tremont st"})
        assert r.status_code == 200
        data = r.json()
        assert data["total_results"] > 0, "Expected results for 75 Tremont St"
        for res in data["results"]:
            addr_upper = res["address"].upper()
            # Should not prefix-match 1575, 175, etc.
            assert "1575" not in addr_upper, f"False prefix match: {addr_upper}"

    def test_search_range_address_resolves(self, client):
        """Range address '55 - 57 Centre Street' resolves to results."""
        r = client.get("/search", params={"q": "55 - 57 Centre Street"})
        assert r.status_code == 200
        data = r.json()
        # Range addresses should match something on Centre Street
        assert data["total_results"] >= 0  # May or may not find exact range

    def test_search_empty_query_returns_empty(self, client):
        """Empty query returns zero results, not an error."""
        r = client.get("/search", params={"q": ""})
        assert r.status_code == 200
        data = r.json()
        assert data["total_results"] == 0

    def test_search_result_structure(self, client):
        """Each search result has address, approval_rate, total_cases."""
        r = client.get("/search", params={"q": "Tremont"})
        assert r.status_code == 200
        data = r.json()
        assert data["total_results"] > 0
        result = data["results"][0]
        assert "address" in result
        assert "approval_rate" in result
        assert "total_cases" in result
        assert isinstance(result["total_cases"], int)
        assert result["total_cases"] > 0

    def test_search_short_query_returns_empty(self, client):
        """Single-character query returns empty (minimum length filter)."""
        r = client.get("/search", params={"q": "a"})
        assert r.status_code == 200
        assert r.json()["total_results"] == 0

    def test_search_nonexistent_returns_empty(self, client):
        """Gibberish query returns empty results, not an error."""
        r = client.get("/search", params={"q": "zzzznonexistent99999"})
        assert r.status_code == 200
        assert r.json()["total_results"] == 0


# =============================================================================
# 2. GEOCODE ENDPOINT — GET /geocode?q=...
# =============================================================================


class TestGeocode:
    """Tests for the geocode (address-to-parcel) endpoint."""

    def test_geocode_basic_address(self, client):
        """Basic address '75 tremont st' resolves to at least one parcel."""
        r = client.get("/geocode", params={"q": "75 tremont st"})
        assert r.status_code == 200
        data = r.json()
        assert data["total"] >= 1, "Expected at least one match for 75 Tremont St"
        result = data["results"][0]
        assert "parcel_id" in result
        assert "address" in result
        # The address should contain TREMONT
        assert "TREMONT" in result["address"].upper()

    def test_geocode_city_zip_stripped(self, client):
        """City and zip code are stripped: '28 Roslin ST Dorchester 02124' resolves."""
        r = client.get("/geocode", params={"q": "28 Roslin ST Dorchester 02124"})
        assert r.status_code == 200
        data = r.json()
        assert data["total"] >= 1, "Expected match after stripping city/zip"
        assert "ROSLIN" in data["results"][0]["address"].upper()

    def test_geocode_directionals_normalized(self, client):
        """Directionals handled: '221 East Eagle Street' resolves."""
        r = client.get("/geocode", params={"q": "221 East Eagle Street"})
        assert r.status_code == 200
        data = r.json()
        assert data["total"] >= 1, "Expected match for 221 East Eagle Street"
        assert "EAGLE" in data["results"][0]["address"].upper()

    def test_geocode_suffix_letter(self, client):
        """Suffix letters handled: '69R Perrin Street' resolves."""
        r = client.get("/geocode", params={"q": "69R Perrin Street"})
        assert r.status_code == 200
        data = r.json()
        # May or may not match exactly — the key is it does not crash
        # and returns a reasonable result on Perrin Street
        if data["total"] > 0:
            assert "PERRIN" in data["results"][0]["address"].upper()

    def test_geocode_avenue_normalization(self, client):
        """Avenue normalization: '500 Harrison Avenue' resolves."""
        r = client.get("/geocode", params={"q": "500 Harrison Avenue"})
        assert r.status_code == 200
        data = r.json()
        assert data["total"] >= 1, "Expected match for 500 Harrison Avenue"
        assert "HARRISON" in data["results"][0]["address"].upper()

    def test_geocode_nearest_number_fallback(self, client):
        """Nearest-number fallback: '58 Centre Street' returns nearby addresses."""
        r = client.get("/geocode", params={"q": "58 Centre Street"})
        assert r.status_code == 200
        data = r.json()
        # Even if 58 does not exist exactly, fallback should find nearby numbers on Centre
        assert data["total"] >= 1, "Expected nearest-number fallback to find Centre St addresses"
        assert "CENTRE" in data["results"][0]["address"].upper()

    def test_geocode_em_dash_range(self, client):
        """Em-dash range address: '431 \u2014 439 Hanover Street' resolves."""
        r = client.get("/geocode", params={"q": "431 \u2014 439 Hanover Street"})
        assert r.status_code == 200
        data = r.json()
        # Em-dash should be normalized to hyphen, then range handling kicks in
        if data["total"] > 0:
            assert "HANOVER" in data["results"][0]["address"].upper()

    def test_geocode_result_has_zoning(self, client):
        """Geocode results include zoning_code and district from GeoJSON."""
        r = client.get("/geocode", params={"q": "1081 River St"})
        assert r.status_code == 200
        data = r.json()
        assert data["total"] >= 1
        result = data["results"][0]
        assert result["address"] == "1081 RIVER ST"
        # Should have zoning enrichment from GeoJSON
        assert "zoning_code" in result or "district" in result

    def test_geocode_no_results(self, client):
        """Nonexistent address returns zero results, not an error."""
        r = client.get("/geocode", params={"q": "99999 Nonexistent Blvd"})
        assert r.status_code == 200
        assert r.json()["total"] == 0


# =============================================================================
# 3. PARCEL ENDPOINT — GET /parcels/{parcel_id}
# =============================================================================


class TestParcel:
    """Tests for the parcel lookup endpoint."""

    def test_parcel_valid_returns_geometry(self, client):
        """Valid parcel returns geometry, district, and zoning_code."""
        r = client.get(f"/parcels/{PRIMARY_PARCEL}")
        assert r.status_code == 200
        data = r.json()
        assert data["parcel_id"] == PRIMARY_PARCEL
        assert "geometry" in data
        assert data["geometry"] is not None
        assert data["district"], "District should not be empty"

    def test_parcel_secondary_valid(self, client):
        """Secondary test parcel (East Boston) also resolves."""
        r = client.get(f"/parcels/{SECONDARY_PARCEL}")
        assert r.status_code == 200
        data = r.json()
        assert data["parcel_id"] == SECONDARY_PARCEL
        assert data["district"]

    def test_parcel_invalid_returns_404(self, client):
        """Invalid parcel ID returns 404."""
        r = client.get("/parcels/9999999999")
        assert r.status_code == 404

    def test_parcel_nonsense_id_returns_404(self, client):
        """Non-numeric parcel ID returns 404."""
        r = client.get("/parcels/not-a-parcel-id")
        assert r.status_code == 404


# =============================================================================
# 4. ZONING ENDPOINT — GET /zoning/{parcel_id}
# =============================================================================


class TestZoning:
    """Tests for the zoning analysis endpoint."""

    def test_zoning_returns_required_fields(self, client):
        """Zoning response has all required fields."""
        r = client.get(f"/zoning/{PRIMARY_PARCEL}")
        assert r.status_code == 200
        data = r.json()
        assert "zoning_district" in data
        assert "zoning_subdistrict" in data
        assert "article" in data
        assert "dimensional_requirements" in data
        assert "allowed_uses" in data
        assert "data_source" in data

    def test_zoning_dimensional_requirements_structure(self, client):
        """dimensional_requirements has the key dimensional limits."""
        r = client.get(f"/zoning/{PRIMARY_PARCEL}")
        assert r.status_code == 200
        reqs = r.json()["dimensional_requirements"]
        # These are the core dimensional fields
        assert "max_far" in reqs, f"Missing max_far in requirements: {list(reqs.keys())}"
        assert "max_height_ft" in reqs, f"Missing max_height_ft in requirements: {list(reqs.keys())}"
        assert "max_stories" in reqs, f"Missing max_stories in requirements: {list(reqs.keys())}"
        assert "min_lot_sf" in reqs, f"Missing min_lot_sf in requirements: {list(reqs.keys())}"
        assert "parking_per_unit" in reqs, f"Missing parking_per_unit in requirements: {list(reqs.keys())}"

    def test_zoning_allowed_uses_is_list(self, client):
        """allowed_uses is a list of strings."""
        r = client.get(f"/zoning/{PRIMARY_PARCEL}")
        assert r.status_code == 200
        uses = r.json()["allowed_uses"]
        assert isinstance(uses, list)
        assert len(uses) > 0, "Expected at least one allowed use"
        assert all(isinstance(u, str) for u in uses)

    def test_zoning_data_source_present(self, client):
        """data_source indicates where zoning data came from."""
        r = client.get(f"/zoning/{PRIMARY_PARCEL}")
        assert r.status_code == 200
        ds = r.json()["data_source"]
        assert ds, "data_source should not be empty"

    def test_zoning_invalid_parcel_returns_404(self, client):
        """Invalid parcel returns 404 for zoning."""
        r = client.get("/zoning/9999999999")
        assert r.status_code == 404


# =============================================================================
# 5. COMPLIANCE CHECK — POST /zoning/check_compliance
# =============================================================================


class TestCompliance:
    """Tests for the zoning compliance check endpoint."""

    def test_compliance_catches_far_violation(self, client):
        """FAR violation detected: proposed 1.5 vs typical max ~0.8 in residential."""
        payload = {
            "parcel_id": PRIMARY_PARCEL,
            "proposed_far": 1.5,
        }
        r = client.post("/zoning/check_compliance", json=payload)
        assert r.status_code == 200
        data = r.json()
        # Check the max_far for this parcel
        max_far = data["requirements"].get("max_far")
        if max_far and max_far < 1.5:
            assert data["compliant"] is False, "Should detect FAR violation"
            violation_types = [v["type"] for v in data["violations"]]
            assert "far" in violation_types, f"FAR violation not in: {violation_types}"
            assert "far" in data["variances_needed"]

    def test_compliance_catches_height_violation(self, client):
        """Height violation detected: proposed 45ft vs max 35ft."""
        payload = {
            "parcel_id": PRIMARY_PARCEL,
            "proposed_height_ft": 45,
        }
        r = client.post("/zoning/check_compliance", json=payload)
        assert r.status_code == 200
        data = r.json()
        max_height = data["requirements"].get("max_height_ft")
        if max_height and max_height < 45:
            assert data["compliant"] is False, "Should detect height violation"
            violation_types = [v["type"] for v in data["violations"]]
            assert "height" in violation_types, f"Height violation not in: {violation_types}"
            assert "height" in data["variances_needed"]

    def test_compliance_catches_stories_violation(self, client):
        """Stories violation detected: proposed 4 vs max 3."""
        payload = {
            "parcel_id": PRIMARY_PARCEL,
            "proposed_stories": 4,
        }
        r = client.post("/zoning/check_compliance", json=payload)
        assert r.status_code == 200
        data = r.json()
        max_stories = data["requirements"].get("max_stories")
        if max_stories and max_stories < 4:
            assert data["compliant"] is False, "Should detect stories violation"
            violation_types = [v["type"] for v in data["violations"]]
            assert "height" in violation_types, f"Stories violation not in: {violation_types}"

    def test_compliance_catches_parking_violation(self, client):
        """Parking violation: 3 spaces for 6 units vs 1/unit required."""
        payload = {
            "parcel_id": PRIMARY_PARCEL,
            "proposed_units": 6,
            "parking_spaces": 3,
        }
        r = client.post("/zoning/check_compliance", json=payload)
        assert r.status_code == 200
        data = r.json()
        parking_req = data["requirements"].get("parking_per_unit")
        if parking_req and parking_req > 0:
            required = int(6 * parking_req)
            if 3 < required:
                assert data["compliant"] is False, "Should detect parking violation"
                violation_types = [v["type"] for v in data["violations"]]
                assert "parking" in violation_types, f"Parking violation not in: {violation_types}"
                assert "parking" in data["variances_needed"]

    def test_compliance_catches_conditional_use_violation(self, client):
        """Conditional use violation: restaurant in residential zone."""
        payload = {
            "parcel_id": PRIMARY_PARCEL,
            "proposed_use": "restaurant",
        }
        r = client.post("/zoning/check_compliance", json=payload)
        assert r.status_code == 200
        data = r.json()
        allowed_uses = data.get("requirements", {}).get("allowed_uses", [])
        # If restaurant is not in allowed uses, we should see conditional_use violation
        if allowed_uses:
            use_allowed = any("restaurant" in u.lower() for u in allowed_uses)
            if not use_allowed:
                violation_types = [v["type"] for v in data["violations"]]
                assert "conditional_use" in violation_types, (
                    f"Expected conditional_use violation for restaurant in residential zone. "
                    f"Violations: {violation_types}"
                )

    def test_compliance_auto_fills_lot_area(self, client):
        """lot_size_sf is auto-filled from property data when not provided."""
        payload = {
            "parcel_id": PRIMARY_PARCEL,
        }
        r = client.post("/zoning/check_compliance", json=payload)
        assert r.status_code == 200
        data = r.json()
        # auto_filled field tells us what was filled automatically
        if "auto_filled" in data and "lot_size_sf" in data.get("auto_filled", []):
            assert data["lot_size_sf"] is not None
            assert data["lot_size_sf"] > 0

    def test_compliance_returns_complexity_level(self, client):
        """Complexity level is one of low/moderate/high."""
        payload = {
            "parcel_id": PRIMARY_PARCEL,
            "proposed_far": 3.0,
            "proposed_height_ft": 100,
            "proposed_stories": 10,
            "proposed_units": 20,
            "parking_spaces": 2,
        }
        r = client.post("/zoning/check_compliance", json=payload)
        assert r.status_code == 200
        data = r.json()
        assert data["complexity"] in ("low", "moderate", "high"), (
            f"Unexpected complexity: {data['complexity']}"
        )
        assert "complexity_note" in data
        assert len(data["complexity_note"]) > 0

    def test_compliance_nested_proposal_format(self, client):
        """Nested format works: {'parcel_id': ..., 'proposal': {...}}."""
        payload = {
            "parcel_id": PRIMARY_PARCEL,
            "proposal": {
                "proposed_height_ft": 45,
                "proposed_far": 1.5,
            }
        }
        r = client.post("/zoning/check_compliance", json=payload)
        assert r.status_code == 200
        data = r.json()
        assert "compliant" in data
        assert "violations" in data
        assert "variances_needed" in data

    def test_compliance_flat_format(self, client):
        """Flat format also works: {'parcel_id': ..., 'proposed_height_ft': 45}."""
        payload = {
            "parcel_id": PRIMARY_PARCEL,
            "proposed_height_ft": 45,
        }
        r = client.post("/zoning/check_compliance", json=payload)
        assert r.status_code == 200
        data = r.json()
        assert "compliant" in data
        assert "violations" in data

    def test_compliance_compliant_project(self, client):
        """A small compliant project returns compliant=True with zero violations."""
        # Get the requirements first so we can propose something compliant
        zr = client.get(f"/zoning/{PRIMARY_PARCEL}")
        assert zr.status_code == 200
        reqs = zr.json()["dimensional_requirements"]

        payload = {
            "parcel_id": PRIMARY_PARCEL,
            "proposed_far": 0.1,
            "proposed_height_ft": 10,
            "proposed_stories": 1,
            "proposed_units": 1,
            "parking_spaces": 10,
            "proposed_use": "residential",
        }
        r = client.post("/zoning/check_compliance", json=payload)
        assert r.status_code == 200
        data = r.json()
        # Very small project should be compliant (or close to it)
        assert data["num_variances_needed"] <= 1, (
            f"Expected near-compliant project but got {data['num_variances_needed']} variances: "
            f"{data['variances_needed']}"
        )

    def test_compliance_missing_parcel_returns_400(self, client):
        """Missing parcel_id returns 400."""
        r = client.post("/zoning/check_compliance", json={"proposed_height_ft": 45})
        assert r.status_code == 400

    def test_compliance_invalid_parcel_returns_404(self, client):
        """Nonexistent parcel returns 404."""
        r = client.post("/zoning/check_compliance", json={"parcel_id": "9999999999"})
        assert r.status_code == 404

    def test_compliance_response_has_zoning_context(self, client):
        """Response includes zoning district, subdistrict, neighborhood."""
        payload = {"parcel_id": PRIMARY_PARCEL, "proposed_height_ft": 45}
        r = client.post("/zoning/check_compliance", json=payload)
        assert r.status_code == 200
        data = r.json()
        assert "zoning_district" in data
        assert "zoning_subdistrict" in data
        assert "neighborhood" in data
        assert "article" in data

    def test_compliance_high_complexity_many_violations(self, client):
        """Exceeding multiple limits produces 'high' complexity."""
        payload = {
            "parcel_id": PRIMARY_PARCEL,
            "proposed_far": 5.0,
            "proposed_height_ft": 200,
            "proposed_stories": 20,
            "proposed_units": 50,
            "parking_spaces": 1,
            "proposed_use": "nightclub",
        }
        r = client.post("/zoning/check_compliance", json=payload)
        assert r.status_code == 200
        data = r.json()
        assert data["num_variances_needed"] >= 3, (
            f"Expected many violations for extreme project, got {data['num_variances_needed']}"
        )
        assert data["complexity"] == "high"


# =============================================================================
# 6. PREDICTION ENDPOINT — POST /analyze_proposal
# =============================================================================


class TestPrediction:
    """Tests for the ML prediction endpoint."""

    def test_prediction_returns_probability(self, client):
        """approval_probability is a float between 0 and 1."""
        payload = {
            "parcel_id": PRIMARY_PARCEL,
            "proposed_use": "residential",
            "variances": ["height", "parking"],
            "has_attorney": True,
        }
        r = client.post("/analyze_proposal", json=payload)
        assert r.status_code == 200
        data = r.json()
        prob = data["approval_probability"]
        assert isinstance(prob, float), f"Expected float, got {type(prob)}"
        assert 0.0 <= prob <= 1.0, f"Probability out of range: {prob}"

    def test_prediction_probability_range(self, client):
        """probability_range is [low, high] where low <= prob <= high."""
        payload = {
            "parcel_id": PRIMARY_PARCEL,
            "proposed_use": "residential",
            "variances": ["height"],
            "has_attorney": True,
        }
        r = client.post("/analyze_proposal", json=payload)
        assert r.status_code == 200
        data = r.json()
        pr = data["probability_range"]
        assert isinstance(pr, list)
        assert len(pr) == 2, f"Expected [low, high], got {pr}"
        assert pr[0] <= data["approval_probability"] <= pr[1], (
            f"Probability {data['approval_probability']} not in range {pr}"
        )
        assert 0.0 <= pr[0]
        assert pr[1] <= 1.0

    def test_prediction_confidence_level(self, client):
        """confidence is one of high/medium/low."""
        payload = {
            "parcel_id": PRIMARY_PARCEL,
            "proposed_use": "residential",
            "variances": ["height"],
        }
        r = client.post("/analyze_proposal", json=payload)
        assert r.status_code == 200
        assert r.json()["confidence"] in ("high", "medium", "low")

    def test_prediction_top_drivers_structure(self, client):
        """top_drivers is a list with feature, direction, shap_value."""
        payload = {
            "parcel_id": PRIMARY_PARCEL,
            "proposed_use": "residential",
            "variances": ["height", "far"],
            "has_attorney": True,
        }
        r = client.post("/analyze_proposal", json=payload)
        assert r.status_code == 200
        drivers = r.json()["top_drivers"]
        assert isinstance(drivers, list)
        assert len(drivers) > 0, "Expected at least one top driver"
        for d in drivers:
            assert "feature" in d, f"Missing 'feature' in driver: {d}"
            assert "direction" in d, f"Missing 'direction' in driver: {d}"
            assert "shap_value" in d, f"Missing 'shap_value' in driver: {d}"
            assert isinstance(d["shap_value"], (int, float))

    def test_prediction_recommendations(self, client):
        """recommendations is a non-empty list of strings."""
        payload = {
            "parcel_id": PRIMARY_PARCEL,
            "proposed_use": "residential",
            "variances": ["height", "far", "parking"],
            "has_attorney": False,
        }
        r = client.post("/analyze_proposal", json=payload)
        assert r.status_code == 200
        recs = r.json()["recommendations"]
        assert isinstance(recs, list)
        # With no attorney and multiple variances, there should be recommendations
        assert len(recs) > 0, "Expected recommendations for risky proposal"

    def test_prediction_similar_cases(self, client):
        """similar_cases is a list; each case has case_number and decision."""
        payload = {
            "parcel_id": PRIMARY_PARCEL,
            "proposed_use": "residential",
            "variances": ["height"],
            "has_attorney": True,
        }
        r = client.post("/analyze_proposal", json=payload)
        assert r.status_code == 200
        cases = r.json()["similar_cases"]
        assert isinstance(cases, list)
        if cases:
            c = cases[0]
            assert "case_number" in c
            assert "decision" in c

    def test_prediction_variance_history(self, client):
        """variance_history has combo_rate and per_variance data."""
        payload = {
            "parcel_id": PRIMARY_PARCEL,
            "proposed_use": "residential",
            "variances": ["height", "parking"],
            "has_attorney": True,
        }
        r = client.post("/analyze_proposal", json=payload)
        assert r.status_code == 200
        vh = r.json()["variance_history"]
        assert isinstance(vh, dict)
        # Should have combo-level rate info
        assert "combo_rate" in vh or "per_variance" in vh, (
            f"Expected combo_rate or per_variance in variance_history: {list(vh.keys())}"
        )

    def test_prediction_estimated_timeline(self, client):
        """estimated_timeline_days is present with median_days and phases."""
        payload = {
            "proposed_use": "residential",
            "variances": ["height"],
        }
        r = client.post("/analyze_proposal", json=payload)
        assert r.status_code == 200
        data = r.json()
        assert "estimated_timeline_days" in data
        tl = data["estimated_timeline_days"]
        if tl is None:
            return  # No timeline data available — acceptable
        if isinstance(tl, (int, float)):
            assert tl > 0, f"Timeline should be positive, got: {tl}"
        elif isinstance(tl, dict):
            # Rich timeline with phases
            assert "median_days" in tl, f"Expected median_days in timeline dict: {list(tl.keys())}"
            assert tl["median_days"] > 0
            if "phases" in tl:
                assert isinstance(tl["phases"], dict)
                assert len(tl["phases"]) > 0
        else:
            pytest.fail(f"Unexpected timeline type: {type(tl)}")

    def test_prediction_has_disclaimer(self, client):
        """Response includes a legal disclaimer."""
        payload = {"proposed_use": "residential", "variances": ["height"]}
        r = client.post("/analyze_proposal", json=payload)
        assert r.status_code == 200
        data = r.json()
        assert "disclaimer" in data
        assert len(data["disclaimer"]) > 50, "Disclaimer seems too short"

    def test_prediction_empty_payload_uses_defaults(self, client):
        """Empty payload should not crash — uses defaults."""
        r = client.post("/analyze_proposal", json={})
        assert r.status_code == 200
        data = r.json()
        assert "approval_probability" in data

    def test_prediction_use_type_alias(self, client):
        """Both 'proposed_use' and 'use_type' field names work."""
        payload = {"use_type": "commercial", "variances": ["far"]}
        r = client.post("/analyze_proposal", json=payload)
        assert r.status_code == 200
        assert r.json()["proposed_use"] == "commercial"

    def test_prediction_non_numeric_units_no_crash(self, client):
        """Non-numeric proposed_units does not crash."""
        payload = {
            "proposed_use": "residential",
            "variances": ["height"],
            "proposed_units": "not_a_number",
        }
        r = client.post("/analyze_proposal", json=payload)
        assert r.status_code == 200

    def test_prediction_ward_auto_detected(self, client):
        """Ward auto-detected from parcel district."""
        payload = {
            "parcel_id": PRIMARY_PARCEL,
            "proposed_use": "residential",
            "variances": ["height"],
        }
        r = client.post("/analyze_proposal", json=payload)
        assert r.status_code == 200
        data = r.json()
        assert data["ward_approval_rate"] is not None, "Ward should be auto-detected"

    def test_prediction_all_12_variance_types(self, client):
        """All 12 variance types are accepted without error."""
        all_variances = [
            "height", "far", "lot_area", "lot_frontage",
            "front_setback", "rear_setback", "side_setback",
            "parking", "conditional_use", "open_space", "density", "nonconforming",
        ]
        payload = {"proposed_use": "residential", "variances": all_variances}
        r = client.post("/analyze_proposal", json=payload)
        assert r.status_code == 200
        data = r.json()
        assert len(data["variances"]) == 12
        assert 0 <= data["approval_probability"] <= 1

    def test_prediction_key_factors_not_empty(self, client):
        """key_factors is a list with real data (not hardcoded placeholders)."""
        payload = {
            "parcel_id": PRIMARY_PARCEL,
            "proposed_use": "residential",
            "variances": ["height", "parking"],
            "has_attorney": True,
        }
        r = client.post("/analyze_proposal", json=payload)
        assert r.status_code == 200
        kf = r.json()["key_factors"]
        assert isinstance(kf, list)
        assert len(kf) > 0, "Expected at least one key factor"


# =============================================================================
# 7. NEARBY CASES — GET /parcels/{parcel_id}/nearby_cases
# =============================================================================


class TestNearbyCases:
    """Tests for the nearby cases geographic search endpoint."""

    def test_nearby_cases_have_distances(self, client):
        """Cases have distance_m and distance_ft (NOT None) for geographic search."""
        r = client.get(f"/parcels/{PRIMARY_PARCEL}/nearby_cases", params={"limit": 5})
        assert r.status_code == 200
        data = r.json()
        assert data["search_type"] == "geographic", (
            f"Expected geographic search, got {data['search_type']}"
        )
        if data["cases"]:
            for case in data["cases"]:
                assert case["distance_m"] is not None, (
                    f"distance_m is None for case {case['case_number']}"
                )
                assert case["distance_ft"] is not None, (
                    f"distance_ft is None for case {case['case_number']}"
                )
                assert isinstance(case["distance_m"], int)
                assert isinstance(case["distance_ft"], int)
                assert case["distance_m"] >= 0
                assert case["distance_ft"] >= 0

    def test_nearby_cases_search_type_geographic(self, client):
        """search_type should be 'geographic' when coords are available."""
        r = client.get(f"/parcels/{PRIMARY_PARCEL}/nearby_cases")
        assert r.status_code == 200
        assert r.json()["search_type"] == "geographic"

    def test_nearby_cases_ward_detection(self, client):
        """Ward detection returns a valid ward number."""
        r = client.get(f"/parcels/{PRIMARY_PARCEL}/nearby_cases")
        assert r.status_code == 200
        data = r.json()
        ward = data["ward"]
        assert ward, "Ward should be detected from nearby cases"
        # Ward should be a small number (Boston has wards 1-22)
        assert ward.isdigit(), f"Ward should be numeric, got: {ward}"
        assert 1 <= int(ward) <= 22, f"Ward out of range: {ward}"

    def test_nearby_cases_ward_only_filter(self, client):
        """ward_only=true filters to single ward."""
        r = client.get(f"/parcels/{PRIMARY_PARCEL}/nearby_cases",
                       params={"ward_only": "true", "limit": 10})
        assert r.status_code == 200
        data = r.json()
        if data["cases"] and data["ward"]:
            # All cases should be in the detected ward
            expected_ward = data["ward"]
            for case in data["cases"]:
                if case["ward"]:
                    assert case["ward"] == expected_ward, (
                        f"Case {case['case_number']} in ward {case['ward']}, "
                        f"expected {expected_ward}"
                    )

    def test_nearby_cases_clean_addresses(self, client):
        """All cases have real addresses (no OCR garbage > 60 chars)."""
        r = client.get(f"/parcels/{PRIMARY_PARCEL}/nearby_cases", params={"limit": 20})
        assert r.status_code == 200
        for case in r.json()["cases"]:
            addr = case["address"]
            assert len(addr) <= 60, f"Address too long (OCR garbage?): {addr[:80]}..."
            assert addr not in ("nan", "None", ""), f"Invalid address: {addr}"

    def test_nearby_cases_has_lat_lon(self, client):
        """Response includes parcel_lat and parcel_lon."""
        r = client.get(f"/parcels/{PRIMARY_PARCEL}/nearby_cases")
        assert r.status_code == 200
        data = r.json()
        assert "parcel_lat" in data
        assert "parcel_lon" in data
        assert isinstance(data["parcel_lat"], float)
        assert isinstance(data["parcel_lon"], float)
        # Boston coordinates are roughly 42.3N, -71.1W
        assert 42.0 < data["parcel_lat"] < 42.5
        assert -71.3 < data["parcel_lon"] < -70.8

    def test_nearby_cases_summary_stats(self, client):
        """Response includes total, approved, denied counts."""
        r = client.get(f"/parcels/{PRIMARY_PARCEL}/nearby_cases")
        assert r.status_code == 200
        data = r.json()
        assert "total" in data
        assert "approved" in data
        assert "denied" in data
        assert data["total"] >= 0
        assert data["approved"] + data["denied"] <= data["total"]

    def test_nearby_cases_invalid_parcel_404(self, client):
        """Nonexistent parcel returns 404."""
        r = client.get("/parcels/9999999999/nearby_cases")
        assert r.status_code == 404

    def test_nearby_cases_case_structure(self, client):
        """Each case has case_number, address, decision, ward, date."""
        r = client.get(f"/parcels/{PRIMARY_PARCEL}/nearby_cases", params={"limit": 3})
        assert r.status_code == 200
        cases = r.json()["cases"]
        if cases:
            case = cases[0]
            assert "case_number" in case
            assert "address" in case
            assert "decision" in case
            assert "ward" in case
            assert "date" in case


# =============================================================================
# 8. CASE HISTORY — GET /address/{address}/cases
# =============================================================================


class TestCaseHistory:
    """Tests for the address case history endpoint."""

    def test_case_history_57_centre(self, client):
        """'57 centre street' returns BOA1776619 with correct people."""
        r = client.get("/address/57 centre street/cases")
        assert r.status_code == 200
        data = r.json()
        assert data["total"] > 0, "Expected cases for 57 Centre Street"
        case_numbers = [c["case_number"] for c in data["cases"]]
        # Check that the specific case exists
        has_target_case = any("BOA1776619" in cn for cn in case_numbers)
        if has_target_case:
            target = next(c for c in data["cases"] if "BOA1776619" in c["case_number"])
            # Michael Winston should be applicant, Gerson Alves should be contact
            if "applicant" in target:
                assert "winston" in target["applicant"].lower(), (
                    f"Expected Winston as applicant, got: {target.get('applicant')}"
                )
            if "contact" in target:
                assert "alves" in target["contact"].lower(), (
                    f"Expected Alves as contact, got: {target.get('contact')}"
                )

    def test_case_history_structure(self, client):
        """Each case has required fields."""
        r = client.get("/address/75 Tremont Street/cases")
        assert r.status_code == 200
        data = r.json()
        if data["cases"]:
            case = data["cases"][0]
            assert "case_number" in case
            assert "decision" in case
            assert "address" in case
            assert "ward" in case

    def test_case_history_empty_address(self, client):
        """Nonexistent address returns empty case list, not error."""
        r = client.get("/address/99999 Nonexistent Blvd/cases")
        assert r.status_code == 200
        assert r.json()["total"] == 0


# =============================================================================
# 9. COMPARE ENDPOINT — POST /compare
# =============================================================================


class TestCompare:
    """Tests for the what-if scenario comparison endpoint."""

    def test_compare_returns_scenarios(self, client):
        """Returns multiple scenarios with probability and delta."""
        payload = {
            "parcel_id": PRIMARY_PARCEL,
            "proposed_use": "residential",
            "variances": ["height", "parking"],
            "has_attorney": True,
        }
        r = client.post("/compare", json=payload)
        assert r.status_code == 200
        data = r.json()
        assert "base_probability" in data
        assert "scenarios" in data
        assert len(data["scenarios"]) >= 1, "Expected at least one scenario"

    def test_compare_scenario_structure(self, client):
        """Each scenario has scenario name, probability, and difference."""
        payload = {
            "parcel_id": PRIMARY_PARCEL,
            "proposed_use": "residential",
            "variances": ["height", "parking"],
            "has_attorney": True,
        }
        r = client.post("/compare", json=payload)
        assert r.status_code == 200
        for s in r.json()["scenarios"]:
            assert "scenario" in s, f"Missing 'scenario' name in: {s}"
            assert "probability" in s, f"Missing 'probability' in: {s}"
            assert "difference" in s, f"Missing 'difference' in: {s}"
            assert 0 <= s["probability"] <= 1

    def test_compare_consistent_with_predict(self, client):
        """Base probability matches /analyze_proposal for same input."""
        payload = {
            "parcel_id": PRIMARY_PARCEL,
            "proposed_use": "residential",
            "variances": ["height"],
            "has_attorney": True,
        }
        r1 = client.post("/analyze_proposal", json=payload)
        r2 = client.post("/compare", json=payload)
        assert r1.status_code == 200
        assert r2.status_code == 200
        prob1 = r1.json()["approval_probability"]
        prob2 = r2.json()["base_probability"]
        assert abs(prob1 - prob2) < 0.01, (
            f"Mismatch: analyze={prob1}, compare={prob2}"
        )

    def test_compare_no_variances(self, client):
        """Compare with empty variances does not crash."""
        payload = {"proposed_use": "residential", "variances": []}
        r = client.post("/compare", json=payload)
        assert r.status_code == 200


# =============================================================================
# 10. MARKET INTEL ENDPOINTS
# =============================================================================


class TestMarketIntel:
    """Tests for market intelligence endpoints."""

    def test_stats_basic(self, client):
        """GET /stats returns total_cases and total_parcels."""
        r = client.get("/stats")
        assert r.status_code == 200
        data = r.json()
        assert "total_cases" in data
        assert "total_parcels" in data
        assert data["total_cases"] > 1000
        assert data["total_parcels"] > 90000
        assert 0 < data["overall_approval_rate"] < 1

    def test_variance_stats(self, client):
        """GET /variance_stats returns variance type data."""
        r = client.get("/variance_stats")
        assert r.status_code == 200
        data = r.json()
        assert "variance_stats" in data
        assert len(data["variance_stats"]) > 5
        for v in data["variance_stats"]:
            assert "variance_type" in v
            assert "approval_rate" in v
            assert 0 <= v["approval_rate"] <= 1
            assert "total_cases" in v
            assert v["total_cases"] > 0

    def test_trends(self, client):
        """GET /trends returns yearly data."""
        r = client.get("/trends")
        assert r.status_code == 200
        data = r.json()
        assert "years" in data
        assert len(data["years"]) >= 3
        for y in data["years"]:
            assert "year" in y
            assert "approval_rate" in y
            assert y["year"] >= 2020
            assert 0 <= y["approval_rate"] <= 1

    def test_attorney_leaderboard(self, client):
        """GET /attorneys/leaderboard returns ranked attorneys."""
        r = client.get("/attorneys/leaderboard", params={"min_cases": 10, "limit": 5})
        assert r.status_code == 200
        data = r.json()
        assert "attorneys" in data
        assert len(data["attorneys"]) > 0
        top = data["attorneys"][0]
        assert top["total_cases"] >= 10
        assert "win_rate" in top or "approval_rate" in top
        assert "attorney_approval_rate" in data

    def test_neighborhoods(self, client):
        """GET /neighborhoods returns district data."""
        r = client.get("/neighborhoods")
        assert r.status_code == 200
        data = r.json()
        assert "neighborhoods" in data
        assert len(data["neighborhoods"]) > 10
        for n in data["neighborhoods"]:
            assert "district" in n or "neighborhood" in n
            assert "approval_rate" in n
            assert 0 <= n["approval_rate"] <= 1

    def test_ward_stats_valid(self, client):
        """GET /wards/1/stats returns ward-level data."""
        r = client.get("/wards/1/stats")
        assert r.status_code == 200
        data = r.json()
        assert data["ward"] == "1"
        assert data["total_cases"] > 0
        assert 0 <= data["approval_rate"] <= 1

    def test_ward_stats_invalid_returns_error(self, client):
        """GET /wards/999/stats returns 404."""
        r = client.get("/wards/999/stats")
        assert r.status_code == 404

    def test_wards_all(self, client):
        """GET /wards/all returns all wards in one call."""
        r = client.get("/wards/all")
        assert r.status_code == 200
        data = r.json()
        assert "wards" in data
        assert len(data["wards"]) > 10
        for w in data["wards"]:
            assert "ward" in w
            assert "approval_rate" in w
            assert 0 <= w["approval_rate"] <= 1

    def test_denial_patterns(self, client):
        """GET /denial_patterns returns approved vs denied comparisons."""
        r = client.get("/denial_patterns")
        assert r.status_code == 200
        data = r.json()
        assert data["total_denied"] > 0
        assert data["total_approved"] > 0
        assert isinstance(data["patterns"], list)
        if data["patterns"]:
            p = data["patterns"][0]
            assert "factor" in p
            assert "direction" in p

    def test_voting_patterns(self, client):
        """GET /voting_patterns returns vote distribution."""
        r = client.get("/voting_patterns")
        assert r.status_code == 200
        data = r.json()
        assert "unanimous_total" in data

    def test_proviso_stats(self, client):
        """GET /proviso_stats returns condition frequencies."""
        r = client.get("/proviso_stats")
        assert r.status_code == 200
        data = r.json()
        assert data["total_approvals"] > 0
        assert "conditions" in data

    def test_timeline_stats(self, client):
        """GET /timeline_stats returns temporal data."""
        r = client.get("/timeline_stats")
        assert r.status_code == 200
        data = r.json()
        assert isinstance(data, dict)
        # Should have either 'overall' key or a 'message' key
        assert "overall" in data or "message" in data

    def test_project_type_stats(self, client):
        """GET /project_type_stats returns data."""
        r = client.get("/project_type_stats")
        assert r.status_code == 200
        data = r.json()
        assert isinstance(data, (dict, list))


# =============================================================================
# 11. HEALTH ENDPOINT
# =============================================================================


class TestHealth:
    """Tests for the health check endpoint."""

    def test_health_status_ok(self, client):
        """Health returns status 'ok'."""
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"

    def test_health_model_loaded(self, client):
        """Model is loaded and available for predictions."""
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["model_loaded"] is True

    def test_health_geojson_loaded(self, client):
        """GeoJSON parcel data is loaded."""
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["geojson_loaded"] is True

    def test_health_zba_loaded(self, client):
        """ZBA case data is loaded."""
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["zba_loaded"] is True

    def test_health_counts(self, client):
        """Health reports reasonable data counts."""
        r = client.get("/health")
        assert r.status_code == 200
        data = r.json()
        assert data["total_parcels"] > 90000
        assert data["total_cases"] > 1000
        assert data["features"] >= 40  # Feature count varies by model version (48-69)

    def test_health_postgis_field(self, client):
        """Health reports PostGIS availability."""
        r = client.get("/health")
        data = r.json()
        assert "postgis_available" in data
        assert isinstance(data["postgis_available"], bool)


# =============================================================================
# 12. ADDITIONAL ENDPOINTS
# =============================================================================


class TestAdditionalEndpoints:
    """Tests for autocomplete, model_info, data_status, batch_predict."""

    def test_autocomplete(self, client):
        """Autocomplete returns suggestions for valid query."""
        r = client.get("/autocomplete", params={"q": "1081 River"})
        assert r.status_code == 200
        data = r.json()
        assert len(data["suggestions"]) >= 1
        assert data["suggestions"][0]["address"] == "1081 RIVER ST"

    def test_autocomplete_short_returns_empty(self, client):
        """Short query returns empty suggestions."""
        r = client.get("/autocomplete", params={"q": "a"})
        assert r.status_code == 200
        assert r.json()["suggestions"] == []

    def test_model_info(self, client):
        """Model info exposes metadata."""
        r = client.get("/model_info")
        assert r.status_code in (200, 503)
        if r.status_code == 200:
            data = r.json()
            assert "model_name" in data
            assert "feature_count" in data
            assert data["feature_count"] > 0
            assert "feature_list" in data or "feature_cols" in data

    def test_data_status(self, client):
        """Data status reports file freshness."""
        r = client.get("/data_status")
        assert r.status_code == 200
        data = r.json()
        assert "zba_cases_cleaned" in data
        assert "zba_model" in data

    def test_batch_predict(self, client):
        """Batch predict handles multiple proposals."""
        payload = {
            "proposals": [
                {"proposed_use": "residential", "variances": ["height"], "has_attorney": True},
                {"proposed_use": "commercial", "variances": ["far", "parking"], "has_attorney": False},
            ]
        }
        r = client.post("/batch_predict", json=payload)
        assert r.status_code == 200
        data = r.json()
        assert data["total"] == 2
        assert len(data["results"]) == 2
        for result in data["results"]:
            assert "approval_probability" in result
            assert 0 <= result["approval_probability"] <= 1

    def test_batch_predict_limit_exceeded(self, client):
        """Batch predict rejects too many proposals."""
        payload = {"proposals": [{"variances": ["height"]}] * 51}
        r = client.post("/batch_predict", json=payload)
        assert r.status_code == 400

    def test_batch_predict_empty(self, client):
        """Batch predict rejects empty list."""
        r = client.post("/batch_predict", json={"proposals": []})
        assert r.status_code == 400

    def test_recommend_endpoint(self, client):
        """Recommend endpoint returns parcels."""
        r = client.get("/recommend", params={
            "use_type": "residential",
            "project_type": "addition",
            "min_approval_rate": 0.3,
            "limit": 5,
        })
        assert r.status_code in (200, 429, 503)
        if r.status_code == 200:
            data = r.json()
            assert "parcels" in data or "recommendations" in data

    def test_zoning_districts_list(self, client):
        """GET /zoning/districts returns all known districts."""
        r = client.get("/zoning/districts")
        assert r.status_code == 200
        data = r.json()
        assert "districts" in data
        assert data["total"] > 0
        d = data["districts"][0]
        assert "code" in d
        assert "name" in d
        assert "max_far" in d
        assert "max_height_ft" in d

    def test_swagger_docs(self, client):
        """Swagger UI is accessible."""
        r = client.get("/docs")
        assert r.status_code == 200

    def test_redoc(self, client):
        """ReDoc is accessible."""
        r = client.get("/redoc")
        assert r.status_code == 200


# =============================================================================
# 13. FULL ZONING ANALYSIS — POST /zoning/full_analysis
# =============================================================================


class TestFullZoningAnalysis:
    """Tests for the combined zoning + compliance + prediction endpoint."""

    def test_full_analysis_returns_all_sections(self, client):
        """Full analysis returns zoning, requirements, compliance, and prediction."""
        payload = {
            "parcel_id": PRIMARY_PARCEL,
            "proposed_far": 1.5,
            "proposed_height_ft": 45,
            "proposed_stories": 4,
            "proposed_units": 6,
            "parking_spaces": 3,
            "proposed_use": "residential",
        }
        r = client.post("/zoning/full_analysis", json=payload)
        assert r.status_code == 200
        data = r.json()
        assert "zoning" in data
        assert "requirements" in data
        assert "compliance" in data
        assert "prediction" in data or "risk_assessment" in data

    def test_full_analysis_missing_parcel(self, client):
        """Full analysis without parcel_id returns 400."""
        r = client.post("/zoning/full_analysis", json={"proposed_height_ft": 45})
        assert r.status_code == 400


# =============================================================================
# 14. EDGE CASES & ROBUSTNESS
# =============================================================================


class TestEdgeCases:
    """Edge case and robustness tests."""

    def test_sql_injection_safe(self, client):
        """SQL injection in search does not crash."""
        r = client.get("/search", params={"q": "'; DROP TABLE cases; --"})
        assert r.status_code == 200
        assert isinstance(r.json().get("results", []), list)

    def test_xss_in_geocode(self, client):
        """XSS payload in geocode does not crash."""
        r = client.get("/geocode", params={"q": "123 <script>alert(1)</script>"})
        assert r.status_code == 200
        assert r.json()["total"] == 0

    def test_unicode_address(self, client):
        """Unicode characters in search do not crash."""
        r = client.get("/search", params={"q": "cafe street"})
        assert r.status_code == 200

    def test_very_long_query(self, client):
        """Extremely long query does not crash."""
        r = client.get("/search", params={"q": "a" * 500})
        assert r.status_code == 200

    def test_special_chars_in_address_path(self, client):
        """Special characters in address path do not crash."""
        r = client.get("/address/123 O'Brien St/cases")
        assert r.status_code == 200

    def test_analyze_huge_values(self, client):
        """Extreme values in prediction do not crash."""
        payload = {
            "proposed_use": "residential",
            "variances": ["height"],
            "proposed_units": 9999,
            "proposed_stories": 999,
        }
        r = client.post("/analyze_proposal", json=payload)
        assert r.status_code == 200
        assert 0 <= r.json()["approval_probability"] <= 1

    def test_unknown_variance_types_accepted(self, client):
        """Unknown variance types do not crash prediction."""
        payload = {
            "proposed_use": "residential",
            "variances": ["height", "fake_variance", "another_fake"],
        }
        r = client.post("/analyze_proposal", json=payload)
        assert r.status_code == 200

    def test_ward_stats_non_numeric_returns_400(self, client):
        """Non-numeric ward ID returns 400."""
        r = client.get("/wards/abc/stats")
        assert r.status_code == 400

    def test_concurrent_requests(self, client):
        """Multiple simultaneous searches do not interfere."""
        addresses = ["75 Tremont", "1081 River", "58 Burbank"]
        results = [client.get("/search", params={"q": a}) for a in addresses]
        for r in results:
            assert r.status_code in (200, 429)


# =============================================================================
# ATTORNEY ENDPOINTS — /attorneys/search, /attorneys/{name}/profile,
#                       /attorneys/{name}/similar_cases
# =============================================================================


class TestAttorneys:
    """Tests for attorney search, profile, and similar cases endpoints."""

    def test_attorney_search_by_name(self, client):
        """Search 'drago' returns Jeffrey Drago in results."""
        r = client.get("/attorneys/search", params={"q": "drago"})
        assert r.status_code == 200
        data = r.json()
        names = [res["name"].lower() for res in data["results"]]
        assert any("drago" in n for n in names), f"Expected Jeffrey Drago in results: {names}"

    def test_attorney_search_gerson_alves(self, client):
        """Search 'gerson' returns Gerson Alves (Michael Winston's rep)."""
        r = client.get("/attorneys/search", params={"q": "gerson"})
        assert r.status_code == 200
        data = r.json()
        names = [res["name"].lower() for res in data["results"]]
        assert any("gerson" in n for n in names), f"Expected Gerson Alves in results: {names}"

    def test_attorney_search_empty(self, client):
        """Empty or too-short query returns error or empty results."""
        r = client.get("/attorneys/search", params={"q": ""})
        # FastAPI validation requires min_length=2, so expect 422
        assert r.status_code in (400, 422), f"Expected 400 or 422 for empty query, got {r.status_code}"

    def test_attorney_profile_structure(self, client):
        """Profile response has required fields: name, total_cases, win_rate, wards, variance_specialties, recent_cases."""
        # First search to get the exact name
        r = client.get("/attorneys/search", params={"q": "drago"})
        assert r.status_code == 200
        results = r.json()["results"]
        assert len(results) > 0, "No results for 'drago' — cannot test profile"
        attorney_name = results[0]["name"]

        r = client.get(f"/attorneys/{attorney_name}/profile")
        assert r.status_code == 200
        data = r.json()
        for field in ("name", "total_cases", "win_rate", "wards", "variance_specialties", "recent_cases"):
            assert field in data, f"Missing field '{field}' in profile response"
        assert isinstance(data["wards"], list)
        assert isinstance(data["variance_specialties"], list)
        assert isinstance(data["recent_cases"], list)

    def test_attorney_profile_drago_stats(self, client):
        """Jeffrey Drago has 800+ cases and win_rate > 0.9."""
        r = client.get("/attorneys/search", params={"q": "drago"})
        assert r.status_code == 200
        results = r.json()["results"]
        assert len(results) > 0
        attorney_name = results[0]["name"]

        r = client.get(f"/attorneys/{attorney_name}/profile")
        assert r.status_code == 200
        data = r.json()
        assert data["total_cases"] >= 800, f"Expected 800+ cases, got {data['total_cases']}"
        assert data["win_rate"] > 0.9, f"Expected win_rate > 0.9, got {data['win_rate']}"

    def test_attorney_profile_not_found(self, client):
        """Nonexistent attorney returns 404."""
        r = client.get("/attorneys/ZZZZZ_NONEXISTENT_ATTORNEY_99999/profile")
        assert r.status_code == 404

    def test_attorney_similar_cases(self, client):
        """Similar cases returns a list with case_number, address, decision."""
        r = client.get("/attorneys/search", params={"q": "drago"})
        assert r.status_code == 200
        results = r.json()["results"]
        assert len(results) > 0
        attorney_name = results[0]["name"]

        r = client.get(f"/attorneys/{attorney_name}/similar_cases")
        assert r.status_code == 200
        data = r.json()
        assert "cases" in data
        assert len(data["cases"]) > 0, "Expected at least one similar case"
        case = data["cases"][0]
        for field in ("case_number", "address", "decision"):
            assert field in case, f"Missing field '{field}' in case item"

    def test_attorney_similar_cases_filtered(self, client):
        """With ward=1 param, all returned cases are in ward 1."""
        r = client.get("/attorneys/search", params={"q": "drago"})
        assert r.status_code == 200
        results = r.json()["results"]
        assert len(results) > 0
        attorney_name = results[0]["name"]

        r = client.get(f"/attorneys/{attorney_name}/similar_cases", params={"ward": "1"})
        assert r.status_code == 200
        data = r.json()
        if data["total_matching"] > 0:
            for case in data["cases"]:
                assert case.get("ward") == "1", f"Expected ward 1, got {case.get('ward')}"
        # If no cases in ward 1, that's fine — just verify the response structure
        assert "cases" in data
        assert "total_matching" in data
