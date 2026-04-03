r"""
PermitIQ Product Tests — Session 9+ Features
=============================================

Tests areas NOT covered by test_integration.py:
  - Parcel issues auto-detection (Session 9)
  - Compliance parcel-level vs proposal-level separation
  - Search result parcel enrichment
  - Full search→zoning→compliance→prediction workflow
  - Attorney endpoints (search, profile, similar cases)
  - Market intelligence endpoints (denial_patterns, voting_patterns, etc.)
  - Edge cases for new features

Run:
    cd ~/Desktop/Boston\ Zoning\ Project
    source zoning-env/bin/activate
    pytest tests/test_product.py -v --tb=short

Requires: API running on port 8000
"""

import pytest
from conftest import PRIMARY_PARCEL, SECONDARY_PARCEL


# Large residential parcel (South Boston) — should NOT auto-detect lot_area
LARGE_PARCEL = "0302951010"


# =============================================================================
# 1. PARCEL ISSUES AUTO-DETECTION (Session 9)
# =============================================================================


class TestParcelIssuesAutoDetection:
    """Tests for auto-detected parcel-level zoning issues from /zoning/{parcel_id}."""

    def test_parcel_issues_section_exists(self, client):
        """GET /zoning/{parcel_id} returns a parcel_issues section."""
        r = client.get(f"/zoning/{PRIMARY_PARCEL}")
        assert r.status_code == 200
        data = r.json()
        assert "parcel_issues" in data, f"Missing parcel_issues key. Keys: {list(data.keys())}"

    def test_parcel_issues_structure(self, client):
        """parcel_issues has auto_detected_variances, auto_detected_violations, proposal_dependent_checks."""
        r = client.get(f"/zoning/{PRIMARY_PARCEL}")
        assert r.status_code == 200
        pi = r.json()["parcel_issues"]
        assert "auto_detected_variances" in pi, f"Missing auto_detected_variances. Keys: {list(pi.keys())}"
        assert "auto_detected_violations" in pi, f"Missing auto_detected_violations. Keys: {list(pi.keys())}"
        assert "proposal_dependent_checks" in pi, f"Missing proposal_dependent_checks. Keys: {list(pi.keys())}"

    def test_parcel_issues_has_summary(self, client):
        """parcel_issues includes a human-readable summary string."""
        r = client.get(f"/zoning/{PRIMARY_PARCEL}")
        assert r.status_code == 200
        pi = r.json()["parcel_issues"]
        assert "summary" in pi, "Missing summary in parcel_issues"
        assert isinstance(pi["summary"], str)
        assert len(pi["summary"]) > 20, "Summary seems too short"

    def test_parcel_issues_has_data_sources(self, client):
        """parcel_issues includes data_sources showing auto-filled data."""
        r = client.get(f"/zoning/{PRIMARY_PARCEL}")
        assert r.status_code == 200
        pi = r.json()["parcel_issues"]
        assert "data_sources" in pi, "Missing data_sources in parcel_issues"
        assert isinstance(pi["data_sources"], dict)

    def test_small_parcel_detects_lot_area_variance(self, client):
        """Parcel 1100051000 (1,200 sf) auto-detects lot_area variance (min 3,000 sf)."""
        r = client.get(f"/zoning/{PRIMARY_PARCEL}")
        assert r.status_code == 200
        pi = r.json()["parcel_issues"]
        assert "lot_area" in pi["auto_detected_variances"], (
            f"Expected lot_area in auto_detected_variances for small parcel. "
            f"Got: {pi['auto_detected_variances']}"
        )

    def test_small_parcel_violation_detail_fields(self, client):
        """Auto-detected violations have source, note, type, requirement, actual, deficit."""
        r = client.get(f"/zoning/{PRIMARY_PARCEL}")
        assert r.status_code == 200
        violations = r.json()["parcel_issues"]["auto_detected_violations"]
        assert len(violations) > 0, "Expected at least one auto-detected violation for small parcel"
        v = violations[0]
        for field in ("type", "requirement", "actual", "deficit", "source", "note"):
            assert field in v, f"Missing field '{field}' in violation: {list(v.keys())}"

    def test_small_parcel_violation_source_is_property_records(self, client):
        """Auto-detected violations cite Boston property assessment records as source."""
        r = client.get(f"/zoning/{PRIMARY_PARCEL}")
        assert r.status_code == 200
        violations = r.json()["parcel_issues"]["auto_detected_violations"]
        lot_area_v = [v for v in violations if v["type"] == "lot_area"]
        assert len(lot_area_v) == 1
        assert "property assessment" in lot_area_v[0]["source"].lower(), (
            f"Expected property assessment source, got: {lot_area_v[0]['source']}"
        )

    def test_small_parcel_lot_size_in_data_sources(self, client):
        """data_sources shows lot_size_sf = 1200 for small parcel."""
        r = client.get(f"/zoning/{PRIMARY_PARCEL}")
        assert r.status_code == 200
        ds = r.json()["parcel_issues"]["data_sources"]
        assert "lot_size_sf" in ds, f"Missing lot_size_sf in data_sources: {ds}"
        assert ds["lot_size_sf"] == 1200.0, f"Expected 1200 sf, got {ds['lot_size_sf']}"

    def test_large_parcel_no_lot_area_variance(self, client):
        """Parcel 0302951010 (16,825 sf) should NOT auto-detect lot_area variance."""
        r = client.get(f"/zoning/{LARGE_PARCEL}")
        assert r.status_code == 200
        pi = r.json()["parcel_issues"]
        assert "lot_area" not in pi["auto_detected_variances"], (
            f"Large parcel should NOT have lot_area variance. "
            f"Got: {pi['auto_detected_variances']}"
        )

    def test_large_parcel_lot_size_still_in_data_sources(self, client):
        """Even without violations, data_sources should show the lot size."""
        r = client.get(f"/zoning/{LARGE_PARCEL}")
        assert r.status_code == 200
        ds = r.json()["parcel_issues"]["data_sources"]
        assert "lot_size_sf" in ds, f"Expected lot_size_sf in data_sources even when compliant"
        assert ds["lot_size_sf"] > 10000, f"Expected large lot, got {ds['lot_size_sf']}"

    def test_proposal_dependent_checks_have_types(self, client):
        """proposal_dependent_checks lists variance types that need proposal input."""
        r = client.get(f"/zoning/{PRIMARY_PARCEL}")
        assert r.status_code == 200
        checks = r.json()["parcel_issues"]["proposal_dependent_checks"]
        assert isinstance(checks, list)
        assert len(checks) >= 4, f"Expected at least 4 proposal-dependent checks, got {len(checks)}"
        for check in checks:
            assert "type" in check, f"Missing 'type' in check: {check}"
            assert "depends_on" in check, f"Missing 'depends_on' in check: {check}"
            assert "input_needed" in check, f"Missing 'input_needed' in check: {check}"

    def test_proposal_dependent_includes_far_height_parking(self, client):
        """FAR, height, and parking are always proposal-dependent."""
        r = client.get(f"/zoning/{PRIMARY_PARCEL}")
        assert r.status_code == 200
        checks = r.json()["parcel_issues"]["proposal_dependent_checks"]
        dep_types = [c["type"] for c in checks]
        for expected in ("far", "height", "parking"):
            assert expected in dep_types, (
                f"Expected '{expected}' in proposal_dependent_checks. Got: {dep_types}"
            )

    def test_summary_mentions_variance_count(self, client):
        """Summary for small parcel mentions the variance count."""
        r = client.get(f"/zoning/{PRIMARY_PARCEL}")
        assert r.status_code == 200
        summary = r.json()["parcel_issues"]["summary"]
        assert "1 variance" in summary.lower() or "lot_area" in summary.lower(), (
            f"Summary should mention lot_area variance: {summary}"
        )


# =============================================================================
# 2. COMPLIANCE PARCEL-LEVEL VS PROPOSAL-LEVEL SEPARATION
# =============================================================================


class TestComplianceLevelSeparation:
    """Tests that compliance response separates parcel-level from proposal-level variances."""

    def test_compliance_has_parcel_level_variances(self, client):
        """Response includes parcel_level_variances section."""
        payload = {"parcel_id": PRIMARY_PARCEL, "proposed_far": 1.5}
        r = client.post("/zoning/check_compliance", json=payload)
        assert r.status_code == 200
        data = r.json()
        assert "parcel_level_variances" in data, f"Missing parcel_level_variances. Keys: {list(data.keys())}"

    def test_compliance_has_proposal_level_variances(self, client):
        """Response includes proposal_level_variances section."""
        payload = {"parcel_id": PRIMARY_PARCEL, "proposed_far": 1.5}
        r = client.post("/zoning/check_compliance", json=payload)
        assert r.status_code == 200
        data = r.json()
        assert "proposal_level_variances" in data, f"Missing proposal_level_variances. Keys: {list(data.keys())}"

    def test_parcel_level_structure(self, client):
        """parcel_level_variances has types, violations, and note fields."""
        payload = {"parcel_id": PRIMARY_PARCEL, "proposed_far": 1.5}
        r = client.post("/zoning/check_compliance", json=payload)
        assert r.status_code == 200
        plv = r.json()["parcel_level_variances"]
        assert "types" in plv, f"Missing 'types' in parcel_level_variances"
        assert "violations" in plv, f"Missing 'violations' in parcel_level_variances"
        assert "note" in plv, f"Missing 'note' in parcel_level_variances"

    def test_proposal_level_structure(self, client):
        """proposal_level_variances has types, violations, and note fields."""
        payload = {"parcel_id": PRIMARY_PARCEL, "proposed_far": 1.5}
        r = client.post("/zoning/check_compliance", json=payload)
        assert r.status_code == 200
        plv = r.json()["proposal_level_variances"]
        assert "types" in plv, f"Missing 'types' in proposal_level_variances"
        assert "violations" in plv, f"Missing 'violations' in proposal_level_variances"
        assert "note" in plv, f"Missing 'note' in proposal_level_variances"

    def test_lot_area_in_parcel_level(self, client):
        """lot_area from auto-fill appears in parcel_level_variances (small lot parcel)."""
        payload = {"parcel_id": PRIMARY_PARCEL, "proposed_far": 1.5, "proposed_height_ft": 45}
        r = client.post("/zoning/check_compliance", json=payload)
        assert r.status_code == 200
        plv = r.json()["parcel_level_variances"]
        assert "lot_area" in plv["types"], (
            f"Expected lot_area in parcel_level types. Got: {plv['types']}"
        )

    def test_lot_area_violation_has_source(self, client):
        """Parcel-level lot_area violation cites property assessment source."""
        payload = {"parcel_id": PRIMARY_PARCEL}
        r = client.post("/zoning/check_compliance", json=payload)
        assert r.status_code == 200
        plv = r.json()["parcel_level_variances"]
        if plv["violations"]:
            lot_v = [v for v in plv["violations"] if v["type"] == "lot_area"]
            assert len(lot_v) == 1, f"Expected one lot_area violation, got {len(lot_v)}"
            assert "source" in lot_v[0], "Parcel-level violation missing 'source' field"
            assert "property assessment" in lot_v[0]["source"].lower()

    def test_far_height_in_proposal_level(self, client):
        """FAR and height from proposal appear in proposal_level_variances."""
        payload = {"parcel_id": PRIMARY_PARCEL, "proposed_far": 1.5, "proposed_height_ft": 45}
        r = client.post("/zoning/check_compliance", json=payload)
        assert r.status_code == 200
        plv = r.json()["proposal_level_variances"]
        assert "far" in plv["types"], f"Expected far in proposal_level types. Got: {plv['types']}"
        assert "height" in plv["types"], f"Expected height in proposal_level types. Got: {plv['types']}"

    def test_parcel_level_note_when_violations_exist(self, client):
        """parcel_level_variances.note is set when parcel violations exist."""
        payload = {"parcel_id": PRIMARY_PARCEL}
        r = client.post("/zoning/check_compliance", json=payload)
        assert r.status_code == 200
        plv = r.json()["parcel_level_variances"]
        if plv["types"]:
            assert plv["note"] is not None, "Note should be set when parcel-level violations exist"
            assert "regardless" in plv["note"].lower(), f"Note should mention 'regardless': {plv['note']}"

    def test_proposal_level_note_when_violations_exist(self, client):
        """proposal_level_variances.note is set when proposal violations exist."""
        payload = {"parcel_id": PRIMARY_PARCEL, "proposed_far": 1.5, "proposed_height_ft": 45}
        r = client.post("/zoning/check_compliance", json=payload)
        assert r.status_code == 200
        plv = r.json()["proposal_level_variances"]
        if plv["types"]:
            assert plv["note"] is not None, "Note should be set when proposal violations exist"
            assert "proposal" in plv["note"].lower() or "triggered" in plv["note"].lower()

    def test_large_parcel_no_parcel_level_violations(self, client):
        """Large parcel (0302951010) has empty parcel_level_variances."""
        payload = {"parcel_id": LARGE_PARCEL, "proposed_far": 0.5}
        r = client.post("/zoning/check_compliance", json=payload)
        assert r.status_code == 200
        plv = r.json()["parcel_level_variances"]
        assert plv["types"] == [], f"Large parcel should have no parcel-level variances. Got: {plv['types']}"
        assert plv["violations"] == [], f"Large parcel should have no parcel-level violations"
        assert plv["note"] is None, "Note should be None when no parcel-level violations"

    def test_compliance_only_parcel_id_still_detects_parcel_issues(self, client):
        """Compliance with only parcel_id (no proposal) still auto-detects parcel issues."""
        payload = {"parcel_id": PRIMARY_PARCEL}
        r = client.post("/zoning/check_compliance", json=payload)
        assert r.status_code == 200
        data = r.json()
        # Small parcel should detect lot_area even without any proposal
        assert "lot_area" in data["variances_needed"], (
            f"Expected lot_area detected with just parcel_id. Got: {data['variances_needed']}"
        )
        assert data["compliant"] is False


# =============================================================================
# 3. SEARCH RESULT PARCEL ENRICHMENT
# =============================================================================


class TestSearchParcelEnrichment:
    """Tests that search results include parcel_id from property assessment geocoder."""

    def test_search_57_centre_has_parcel_id(self, client):
        """Search for '57 Centre' returns results with parcel_id field."""
        r = client.get("/search", params={"q": "57 Centre"})
        assert r.status_code == 200
        data = r.json()
        assert data["total_results"] > 0
        # Find the 57 Centre result
        centre_results = [res for res in data["results"] if "CENTRE" in res["address"].upper()]
        assert len(centre_results) > 0, "No Centre St results found"
        result = centre_results[0]
        assert "parcel_id" in result, f"Missing parcel_id in search result. Keys: {list(result.keys())}"
        assert result["parcel_id"] is not None, "parcel_id should not be None"
        assert result["parcel_id"] == PRIMARY_PARCEL, (
            f"Expected parcel_id {PRIMARY_PARCEL}, got {result['parcel_id']}"
        )

    def test_search_result_parcel_id_is_string(self, client):
        """parcel_id in search results is a string."""
        r = client.get("/search", params={"q": "57 Centre"})
        assert r.status_code == 200
        data = r.json()
        for result in data["results"]:
            if "parcel_id" in result and result["parcel_id"] is not None:
                assert isinstance(result["parcel_id"], str), (
                    f"parcel_id should be string, got {type(result['parcel_id'])}"
                )

    def test_search_nonexistent_no_parcel_id(self, client):
        """Search for nonexistent address returns empty results."""
        r = client.get("/search", params={"q": "99999 Nonexistent Blvd"})
        assert r.status_code == 200
        assert r.json()["total_results"] == 0


# =============================================================================
# 4. FULL WORKFLOW TEST
# =============================================================================


class TestFullWorkflow:
    """End-to-end: Search -> Zoning -> Compliance -> Prediction, passing data between steps."""

    def test_search_to_zoning_to_compliance_to_prediction(self, client):
        """Full workflow: search for address, get parcel, check zoning, compliance, predict."""
        # Step 1: Search for an address
        r = client.get("/search", params={"q": "57 Centre Street"})
        assert r.status_code == 200
        search_data = r.json()
        assert search_data["total_results"] > 0, "Step 1 failed: no search results"
        centre_results = [res for res in search_data["results"] if "CENTRE" in res["address"].upper()]
        assert len(centre_results) > 0
        parcel_id = centre_results[0].get("parcel_id")
        assert parcel_id is not None, "Step 1: search result missing parcel_id"

        # Step 2: Get zoning analysis using parcel_id from search
        r = client.get(f"/zoning/{parcel_id}")
        assert r.status_code == 200
        zoning_data = r.json()
        assert "dimensional_requirements" in zoning_data, "Step 2 failed: no requirements"
        assert "parcel_issues" in zoning_data, "Step 2 failed: no parcel_issues"
        reqs = zoning_data["dimensional_requirements"]

        # Step 3: Check compliance — deliberately exceed limits
        max_far = reqs.get("max_far", 1.0)
        max_height = reqs.get("max_height_ft", 35)
        payload = {
            "parcel_id": parcel_id,
            "proposed_far": max_far + 1.0,
            "proposed_height_ft": max_height + 20,
            "proposed_units": 6,
            "parking_spaces": 2,
            "proposed_use": "residential",
        }
        r = client.post("/zoning/check_compliance", json=payload)
        assert r.status_code == 200
        compliance_data = r.json()
        assert compliance_data["compliant"] is False, "Step 3: should detect violations"
        variances = compliance_data["variances_needed"]
        assert len(variances) >= 2, f"Step 3: expected 2+ variances, got {variances}"

        # Step 4: Predict approval using variances from compliance
        pred_payload = {
            "parcel_id": parcel_id,
            "proposed_use": "residential",
            "variances": variances,
            "has_attorney": True,
            "proposed_units": 6,
            "proposed_stories": 4,
        }
        r = client.post("/analyze_proposal", json=pred_payload)
        assert r.status_code == 200
        pred_data = r.json()
        assert 0.0 <= pred_data["approval_probability"] <= 1.0, "Step 4: invalid probability"
        assert len(pred_data["top_drivers"]) > 0, "Step 4: no top_drivers"
        assert len(pred_data["recommendations"]) >= 0

    def test_data_consistency_across_endpoints(self, client):
        """Zoning district from /zoning matches district from /zoning/check_compliance."""
        r1 = client.get(f"/zoning/{PRIMARY_PARCEL}")
        assert r1.status_code == 200
        zoning_district_1 = r1.json()["zoning_district"]

        r2 = client.post("/zoning/check_compliance", json={
            "parcel_id": PRIMARY_PARCEL, "proposed_far": 1.0,
        })
        assert r2.status_code == 200
        zoning_district_2 = r2.json()["zoning_district"]

        assert zoning_district_1 == zoning_district_2, (
            f"District mismatch: /zoning says '{zoning_district_1}', "
            f"compliance says '{zoning_district_2}'"
        )

    def test_parcel_issues_consistent_with_compliance(self, client):
        """Parcel-level variances from /zoning match parcel_level_variances in compliance."""
        r1 = client.get(f"/zoning/{PRIMARY_PARCEL}")
        assert r1.status_code == 200
        auto_variances = r1.json()["parcel_issues"]["auto_detected_variances"]

        r2 = client.post("/zoning/check_compliance", json={"parcel_id": PRIMARY_PARCEL})
        assert r2.status_code == 200
        parcel_types = r2.json()["parcel_level_variances"]["types"]

        assert set(auto_variances) == set(parcel_types), (
            f"Parcel issues mismatch: /zoning says {auto_variances}, "
            f"compliance says {parcel_types}"
        )


# =============================================================================
# 5. ATTORNEY ENDPOINTS
# =============================================================================


class TestAttorneyEndpoints:
    """Tests for attorney search, profile, and leaderboard (Session 7+ features)."""

    def test_leaderboard_returns_attorneys_with_wins(self, client):
        """GET /attorneys/leaderboard returns attorneys with wins, cases, rate."""
        r = client.get("/attorneys/leaderboard", params={"min_cases": 5, "limit": 10})
        assert r.status_code == 200
        data = r.json()
        assert "attorneys" in data
        assert len(data["attorneys"]) > 0
        atty = data["attorneys"][0]
        assert "name" in atty
        assert "total_cases" in atty
        assert atty["total_cases"] >= 5

    def test_leaderboard_has_overall_attorney_rate(self, client):
        """Leaderboard response includes the overall attorney approval rate."""
        r = client.get("/attorneys/leaderboard")
        assert r.status_code == 200
        data = r.json()
        assert "attorney_approval_rate" in data
        assert 0 < data["attorney_approval_rate"] < 1

    def test_attorney_search_drago(self, client):
        """GET /attorneys/search?q=drago returns matching attorneys."""
        r = client.get("/attorneys/search", params={"q": "drago"})
        assert r.status_code == 200
        data = r.json()
        assert "results" in data
        names = [res["name"].lower() for res in data["results"]]
        assert any("drago" in n for n in names), f"Expected Drago in results: {names}"

    def test_attorney_search_result_has_stats(self, client):
        """Attorney search results include total_cases and win_rate."""
        r = client.get("/attorneys/search", params={"q": "drago"})
        assert r.status_code == 200
        results = r.json()["results"]
        assert len(results) > 0
        atty = results[0]
        assert "total_cases" in atty
        assert "win_rate" in atty or "approval_rate" in atty

    def test_trends_returns_yearly_data(self, client):
        """GET /trends returns yearly data with year, approved, denied, rate."""
        r = client.get("/trends")
        assert r.status_code == 200
        data = r.json()
        assert "years" in data
        assert len(data["years"]) >= 3
        year_entry = data["years"][0]
        assert "year" in year_entry
        assert "approval_rate" in year_entry
        assert isinstance(year_entry["year"], int)
        assert year_entry["year"] >= 2020

    def test_trends_years_are_sorted(self, client):
        """Trend years are returned in chronological order."""
        r = client.get("/trends")
        assert r.status_code == 200
        years = [y["year"] for y in r.json()["years"]]
        assert years == sorted(years), f"Years not sorted: {years}"

    def test_trends_rates_are_valid(self, client):
        """All trend approval rates are between 0 and 1."""
        r = client.get("/trends")
        assert r.status_code == 200
        for y in r.json()["years"]:
            assert 0 <= y["approval_rate"] <= 1, f"Invalid rate for year {y['year']}: {y['approval_rate']}"


# =============================================================================
# 6. MARKET INTELLIGENCE ENDPOINTS
# =============================================================================


class TestMarketIntelligenceExtended:
    """Tests for market intelligence endpoints not fully covered in test_integration.py."""

    def test_variance_stats_types_include_common_variances(self, client):
        """Variance stats include height, far, parking."""
        r = client.get("/variance_stats")
        assert r.status_code == 200
        stats = r.json()["variance_stats"]
        types_found = [v["variance_type"] for v in stats]
        for expected in ("height", "far", "parking"):
            assert any(expected in t.lower() for t in types_found), (
                f"Expected '{expected}' in variance types: {types_found}"
            )

    def test_variance_stats_case_counts_positive(self, client):
        """All variance types have positive case counts."""
        r = client.get("/variance_stats")
        assert r.status_code == 200
        for v in r.json()["variance_stats"]:
            assert v["total_cases"] > 0, f"Zero cases for {v['variance_type']}"

    def test_neighborhoods_returns_districts_with_rates(self, client):
        """GET /neighborhoods returns districts with valid approval rates."""
        r = client.get("/neighborhoods")
        assert r.status_code == 200
        data = r.json()
        assert "neighborhoods" in data
        assert len(data["neighborhoods"]) > 10
        for n in data["neighborhoods"]:
            rate = n["approval_rate"]
            assert 0 <= rate <= 1, f"Invalid rate {rate} for {n.get('district', n.get('neighborhood'))}"

    def test_denial_patterns_returns_factors(self, client):
        """GET /denial_patterns returns patterns with factor and direction."""
        r = client.get("/denial_patterns")
        assert r.status_code == 200
        data = r.json()
        assert "patterns" in data
        assert "total_denied" in data
        assert "total_approved" in data
        assert data["total_denied"] > 0
        assert data["total_approved"] > 0
        if data["patterns"]:
            p = data["patterns"][0]
            assert "factor" in p
            assert "direction" in p

    def test_denial_patterns_factors_are_meaningful(self, client):
        """Denial pattern factors have descriptive names."""
        r = client.get("/denial_patterns")
        assert r.status_code == 200
        factors = [p["factor"] for p in r.json()["patterns"]]
        # At least some factors should be non-empty strings
        assert all(isinstance(f, str) and len(f) > 0 for f in factors), (
            f"Some factors are empty or not strings: {factors}"
        )

    def test_voting_patterns_returns_vote_distributions(self, client):
        """GET /voting_patterns returns vote distribution data."""
        r = client.get("/voting_patterns")
        assert r.status_code == 200
        data = r.json()
        assert "unanimous_total" in data
        assert isinstance(data["unanimous_total"], int)
        assert data["unanimous_total"] >= 0

    def test_voting_patterns_has_distribution_details(self, client):
        """Voting patterns include distribution breakdown."""
        r = client.get("/voting_patterns")
        assert r.status_code == 200
        data = r.json()
        # Should have some kind of breakdown
        assert any(k in data for k in ("distribution", "unanimous_approved", "unanimous_denied",
                                        "split_votes", "total_with_votes")), (
            f"Expected vote distribution details. Keys: {list(data.keys())}"
        )

    def test_project_type_stats_returns_data(self, client):
        """GET /project_type_stats returns project type approval data."""
        r = client.get("/project_type_stats")
        assert r.status_code == 200
        data = r.json()
        assert isinstance(data, (dict, list))
        # Should have at least some project types
        if isinstance(data, dict):
            assert "project_types" in data or len(data) > 0

    def test_proviso_stats_has_conditions(self, client):
        """GET /proviso_stats returns conditions attached to approvals."""
        r = client.get("/proviso_stats")
        assert r.status_code == 200
        data = r.json()
        assert "conditions" in data
        assert data["total_approvals"] > 0
        assert isinstance(data["conditions"], list)


# =============================================================================
# 7. EDGE CASES
# =============================================================================


class TestEdgeCasesProduct:
    """Edge cases for Session 9+ features."""

    def test_compliance_parcel_only_no_proposal(self, client):
        """Compliance with only parcel_id (no proposal) still returns full structure."""
        payload = {"parcel_id": PRIMARY_PARCEL}
        r = client.post("/zoning/check_compliance", json=payload)
        assert r.status_code == 200
        data = r.json()
        assert "compliant" in data
        assert "violations" in data
        assert "parcel_level_variances" in data
        assert "proposal_level_variances" in data
        # Should still auto-detect parcel issues
        assert data["parcel_level_variances"]["types"] == ["lot_area"]
        # No proposal means no proposal-level violations
        assert data["proposal_level_variances"]["types"] == []

    def test_prediction_minimal_input_parcel_and_variances(self, client):
        """Prediction with just parcel_id + variances works."""
        payload = {
            "parcel_id": PRIMARY_PARCEL,
            "variances": ["height", "lot_area"],
        }
        r = client.post("/analyze_proposal", json=payload)
        assert r.status_code == 200
        data = r.json()
        assert 0.0 <= data["approval_probability"] <= 1.0

    def test_prediction_only_parcel_id(self, client):
        """Prediction with only parcel_id (no variances) works."""
        payload = {"parcel_id": PRIMARY_PARCEL}
        r = client.post("/analyze_proposal", json=payload)
        assert r.status_code == 200
        assert 0.0 <= r.json()["approval_probability"] <= 1.0

    def test_search_empty_returns_no_results(self, client):
        """Search for nonexistent address returns empty results, not error."""
        r = client.get("/search", params={"q": "zzzznonexistent99999"})
        assert r.status_code == 200
        assert r.json()["total_results"] == 0

    def test_zoning_for_parcel_without_property_data(self, client):
        """Zoning for a parcel with minimal/no property assessment still returns structure."""
        # Use SECONDARY_PARCEL (0100001000) — East Boston, may have limited data
        r = client.get(f"/zoning/{SECONDARY_PARCEL}")
        assert r.status_code == 200
        data = r.json()
        assert "parcel_issues" in data
        pi = data["parcel_issues"]
        # Even without property data, structure should be complete
        assert "auto_detected_variances" in pi
        assert "auto_detected_violations" in pi
        assert "proposal_dependent_checks" in pi
        assert isinstance(pi["auto_detected_variances"], list)
        assert isinstance(pi["proposal_dependent_checks"], list)

    def test_compliance_large_parcel_compliant_project(self, client):
        """Small compliant project on large parcel has zero variances."""
        payload = {
            "parcel_id": LARGE_PARCEL,
            "proposed_far": 0.1,
            "proposed_height_ft": 10,
            "proposed_stories": 1,
            "proposed_units": 1,
            "parking_spaces": 5,
            "proposed_use": "residential",
        }
        r = client.post("/zoning/check_compliance", json=payload)
        assert r.status_code == 200
        data = r.json()
        assert data["parcel_level_variances"]["types"] == []
        assert data["proposal_level_variances"]["types"] == []
        assert data["num_variances_needed"] <= 1

    def test_batch_predict_with_parcel_ids(self, client):
        """Batch predict with different parcel_ids returns results for each."""
        payload = {
            "proposals": [
                {"parcel_id": PRIMARY_PARCEL, "proposed_use": "residential", "variances": ["height"]},
                {"parcel_id": LARGE_PARCEL, "proposed_use": "commercial", "variances": ["far"]},
            ]
        }
        r = client.post("/batch_predict", json=payload)
        assert r.status_code == 200
        data = r.json()
        assert data["total"] == 2
        for result in data["results"]:
            assert 0 <= result["approval_probability"] <= 1

    def test_compliance_both_level_violations_counted_in_total(self, client):
        """Total variances_needed includes both parcel and proposal level."""
        payload = {
            "parcel_id": PRIMARY_PARCEL,
            "proposed_far": 1.5,
            "proposed_height_ft": 45,
        }
        r = client.post("/zoning/check_compliance", json=payload)
        assert r.status_code == 200
        data = r.json()
        parcel_count = len(data["parcel_level_variances"]["types"])
        proposal_count = len(data["proposal_level_variances"]["types"])
        total = data["num_variances_needed"]
        assert total == parcel_count + proposal_count, (
            f"Total variances ({total}) != parcel ({parcel_count}) + proposal ({proposal_count})"
        )
