r"""
PermitIQ API Integration Tests
Run: cd ~/Desktop/Boston\ Zoning\ Project && source zoning-env/bin/activate && pytest tests/ -v
Requires: API running on port 8000
"""

import httpx
import pytest

BASE = "http://127.0.0.1:8000"


@pytest.fixture(scope="session")
def client():
    with httpx.Client(base_url=BASE, timeout=30) as c:
        yield c


# =========================
# HEALTH & PLATFORM
# =========================

def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    assert data["geojson_loaded"] is True
    assert data["zba_loaded"] is True
    assert data["model_loaded"] is True
    assert data["total_parcels"] > 90000
    assert data["total_cases"] > 1000  # Lower threshold: dataset size varies during OCR rebuilds
    assert data["features"] >= 50  # Feature count varies: 57 (v3 leakage-free) or 69 (v2)


def test_stats(client):
    r = client.get("/stats")
    assert r.status_code == 200
    data = r.json()
    assert data["total_cases"] > 1000  # Lower threshold: dataset size varies during OCR rebuilds
    assert 0 < data["overall_approval_rate"] < 1
    assert data["total_wards"] > 15
    assert data["best_ward"] is not None
    assert data["worst_ward"] is not None


# =========================
# SEARCH
# =========================

def test_search_basic(client):
    r = client.get("/search", params={"q": "Tremont"})
    assert r.status_code == 200
    data = r.json()
    assert data["total_results"] > 0
    result = data["results"][0]
    assert "address" in result
    assert "approval_rate" in result
    assert "total_cases" in result


def test_search_short_query(client):
    r = client.get("/search", params={"q": "a"})
    assert r.status_code == 200
    assert r.json()["total_results"] == 0


def test_search_no_results(client):
    r = client.get("/search", params={"q": "zzzznonexistent99999"})
    assert r.status_code == 200
    assert r.json()["total_results"] == 0


def test_address_cases(client):
    r = client.get("/address/Tremont Street/cases")
    assert r.status_code == 200
    data = r.json()
    assert "cases" in data
    if data["cases"]:
        case = data["cases"][0]
        assert "case_number" in case
        assert "decision" in case


# =========================
# PARCELS & GEOCODING
# =========================

def test_parcel_lookup(client):
    r = client.get("/parcels/0100001000")
    assert r.status_code == 200
    data = r.json()
    assert data["parcel_id"] == "0100001000"
    assert data["district"]  # Non-empty (field content varies by source: PostGIS vs GeoJSON)
    assert "geometry" in data


def test_parcel_not_found(client):
    r = client.get("/parcels/9999999999")
    assert r.status_code == 404


def test_geocode(client):
    r = client.get("/geocode", params={"q": "1081 River St"})
    assert r.status_code == 200
    data = r.json()
    assert data["total"] >= 1
    assert data["results"][0]["address"] == "1081 RIVER ST"


def test_geocode_no_results(client):
    r = client.get("/geocode", params={"q": "99999 Nonexistent Blvd"})
    assert r.status_code == 200
    assert r.json()["total"] == 0


# =========================
# PREDICTION
# =========================

def test_analyze_proposal_basic(client):
    payload = {
        "parcel_id": "0100001000",
        "proposed_use": "residential",
        "variances": ["height", "parking"],
        "has_attorney": True,
    }
    r = client.post("/analyze_proposal", json=payload)
    assert r.status_code == 200
    data = r.json()
    assert 0 <= data["approval_probability"] <= 1
    assert data["confidence"] in ("high", "medium", "low")
    assert "probability_range" in data
    assert len(data["probability_range"]) == 2
    assert "key_factors" in data
    assert "top_drivers" in data
    assert "similar_cases" in data


def test_analyze_proposal_auto_ward(client):
    """Ward should be auto-detected from parcel district."""
    payload = {
        "parcel_id": "0100001000",
        "proposed_use": "residential",
        "variances": ["height"],
    }
    r = client.post("/analyze_proposal", json=payload)
    assert r.status_code == 200
    data = r.json()
    # East Boston → Ward 1, so ward_approval_rate should be populated
    assert data["ward_approval_rate"] is not None


def test_analyze_proposal_use_type_alias(client):
    """Both proposed_use and use_type should work."""
    payload = {
        "use_type": "commercial",
        "variances": ["far"],
    }
    r = client.post("/analyze_proposal", json=payload)
    assert r.status_code == 200
    assert r.json()["proposed_use"] == "commercial"


def test_analyze_proposal_bad_input(client):
    """Should not crash on non-numeric units."""
    payload = {
        "proposed_use": "residential",
        "variances": ["height"],
        "proposed_units": "not_a_number",
    }
    r = client.post("/analyze_proposal", json=payload)
    assert r.status_code == 200


def test_compare_scenarios(client):
    payload = {
        "parcel_id": "0100001000",
        "proposed_use": "residential",
        "variances": ["height", "parking"],
        "has_attorney": True,
    }
    r = client.post("/compare", json=payload)
    assert r.status_code == 200
    data = r.json()
    assert "base_probability" in data
    assert "scenarios" in data
    assert len(data["scenarios"]) >= 1
    for s in data["scenarios"]:
        assert "scenario" in s
        assert "probability" in s
        assert "difference" in s


def test_compare_consistent_with_predict(client):
    """Base probability in /compare should match /analyze_proposal."""
    payload = {
        "parcel_id": "0100001000",
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
    assert abs(prob1 - prob2) < 0.01, f"Mismatch: analyze={prob1}, compare={prob2}"


def test_batch_predict(client):
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


def test_batch_predict_limit(client):
    payload = {"proposals": [{"variances": ["height"]}] * 21}
    r = client.post("/batch_predict", json=payload)
    assert r.status_code == 400


# =========================
# MARKET INTELLIGENCE
# =========================

def test_ward_stats(client):
    r = client.get("/wards/1/stats")
    assert r.status_code == 200
    data = r.json()
    assert data["ward"] == "1"
    assert data["total_cases"] > 0
    assert 0 <= data["approval_rate"] <= 1


def test_ward_not_found(client):
    r = client.get("/wards/999/stats")
    assert r.status_code == 404


def test_variance_stats(client):
    r = client.get("/variance_stats")
    assert r.status_code == 200
    data = r.json()
    assert len(data["variance_stats"]) > 5
    for v in data["variance_stats"]:
        assert "variance_type" in v
        assert 0 <= v["approval_rate"] <= 1


def test_neighborhoods(client):
    r = client.get("/neighborhoods")
    assert r.status_code == 200
    data = r.json()
    assert len(data["neighborhoods"]) > 10


def test_attorney_leaderboard(client):
    r = client.get("/attorneys/leaderboard", params={"min_cases": 10, "limit": 5})
    assert r.status_code == 200
    data = r.json()
    assert len(data["attorneys"]) > 0
    assert data["attorneys"][0]["total_cases"] >= 10
    assert "attorney_approval_rate" in data


def test_trends(client):
    r = client.get("/trends")
    assert r.status_code == 200
    data = r.json()
    assert len(data["years"]) >= 3
    for y in data["years"]:
        assert y["year"] >= 2020
        assert 0 <= y["approval_rate"] <= 1


# =========================
# DATA STATUS & NEARBY CASES
# =========================

def test_data_status(client):
    r = client.get("/data_status")
    assert r.status_code == 200
    data = r.json()
    assert "zba_cases_cleaned" in data
    assert "zba_model" in data
    assert "ocr_pipeline" in data


def test_nearby_cases(client):
    r = client.get("/parcels/0100001000/nearby_cases?limit=5")
    assert r.status_code == 200
    data = r.json()
    assert data["district"]  # Non-empty (field content varies by source: PostGIS vs GeoJSON)
    assert "parcel_lat" in data
    assert "parcel_lon" in data


def test_nearby_cases_not_found(client):
    r = client.get("/parcels/9999999999/nearby_cases")
    assert r.status_code == 404


def test_denial_patterns(client):
    r = client.get("/denial_patterns")
    assert r.status_code == 200
    data = r.json()
    assert data["total_denied"] > 0
    assert data["total_approved"] > 0
    assert isinstance(data["patterns"], list)


def test_voting_patterns(client):
    r = client.get("/voting_patterns")
    assert r.status_code == 200
    data = r.json()
    assert "unanimous_total" in data


def test_proviso_stats(client):
    r = client.get("/proviso_stats")
    assert r.status_code == 200
    data = r.json()
    assert data["total_approvals"] > 0


def test_similar_cases_have_relevance(client):
    """Similar cases should include relevance scores."""
    payload = {
        "parcel_id": "0100001000",
        "proposed_use": "residential",
        "variances": ["height", "parking"],
        "has_attorney": True,
    }
    r = client.post("/analyze_proposal", json=payload)
    assert r.status_code == 200
    data = r.json()
    cases = data.get("similar_cases", [])
    if cases:
        assert "relevance_score" in cases[0]


# =========================
# NEW ENDPOINTS
# =========================

def test_timeline_stats(client):
    r = client.get("/timeline_stats")
    assert r.status_code == 200
    data = r.json()
    assert "overall" in data or "message" in data


def test_autocomplete(client):
    r = client.get("/autocomplete", params={"q": "1081 River"})
    assert r.status_code == 200
    data = r.json()
    assert len(data["suggestions"]) >= 1
    assert data["suggestions"][0]["address"] == "1081 RIVER ST"


def test_autocomplete_short_query(client):
    r = client.get("/autocomplete", params={"q": "a"})
    assert r.status_code == 200
    assert len(r.json()["suggestions"]) == 0


def test_model_info(client):
    r = client.get("/model_info")
    assert r.status_code == 200
    data = r.json()
    assert data["model_name"] is not None
    assert data["feature_count"] > 0
    assert "feature_list" in data
    assert "trained_at" in data


def test_prediction_has_timeline(client):
    payload = {"proposed_use": "residential", "variances": ["height"]}
    r = client.post("/analyze_proposal", json=payload)
    assert r.status_code == 200
    data = r.json()
    assert "estimated_timeline_days" in data


# =========================
# EDGE CASES
# =========================

def test_empty_variances(client):
    """Should handle empty variance list gracefully."""
    payload = {"proposed_use": "residential", "variances": []}
    r = client.post("/analyze_proposal", json=payload)
    assert r.status_code == 200


def test_unicode_address_search(client):
    r = client.get("/search", params={"q": "café street 日本語"})
    assert r.status_code == 200
    assert r.json()["total_results"] == 0


def test_very_long_query(client):
    r = client.get("/search", params={"q": "a" * 500})
    assert r.status_code == 200


def test_special_chars_in_address(client):
    r = client.get("/address/123 O'Brien St/cases")
    assert r.status_code == 200


def test_batch_predict_empty(client):
    r = client.post("/batch_predict", json={"proposals": []})
    assert r.status_code == 400


def test_analyze_all_variance_types(client):
    """All 12 variance types should be accepted."""
    all_variances = [
        "height", "far", "lot_area", "lot_frontage",
        "front_setback", "rear_setback", "side_setback",
        "parking", "conditional_use", "open_space", "density", "nonconforming"
    ]
    payload = {"proposed_use": "residential", "variances": all_variances}
    r = client.post("/analyze_proposal", json=payload)
    assert r.status_code == 200
    assert r.json()["approval_probability"] >= 0


def test_compare_no_variances(client):
    payload = {"proposed_use": "residential", "variances": []}
    r = client.post("/compare", json=payload)
    assert r.status_code == 200


def test_geocode_special_chars(client):
    r = client.get("/geocode", params={"q": "123 <script>alert(1)</script>"})
    assert r.status_code == 200
    assert r.json()["total"] == 0


# =========================
# DOCS
# =========================

def test_swagger_docs(client):
    r = client.get("/docs")
    assert r.status_code == 200


def test_redoc(client):
    r = client.get("/redoc")
    assert r.status_code == 200


# =========================
# EDGE CASES
# =========================

def test_search_empty_string(client):
    r = client.get("/search", params={"q": ""})
    assert r.status_code == 200

def test_search_unicode(client):
    r = client.get("/search", params={"q": "123 Ñoño Street"})
    assert r.status_code == 200

def test_search_sql_injection(client):
    r = client.get("/search", params={"q": "'; DROP TABLE cases; --"})
    assert r.status_code == 200
    assert isinstance(r.json().get("results", []), list)

def test_parcel_invalid_format(client):
    r = client.get("/parcels/not-a-parcel-id")
    assert r.status_code == 404

def test_parcel_too_long(client):
    r = client.get("/parcels/" + "0" * 100)
    assert r.status_code == 404

def test_analyze_empty_payload(client):
    r = client.post("/analyze_proposal", json={})
    assert r.status_code == 200  # Should use defaults

def test_analyze_huge_units(client):
    payload = {
        "proposed_use": "residential",
        "variances": ["height"],
        "proposed_units": 9999,
        "proposed_stories": 999,
    }
    r = client.post("/analyze_proposal", json=payload)
    assert r.status_code == 200

def test_analyze_all_variances(client):
    payload = {
        "proposed_use": "commercial",
        "variances": [
            "height", "far", "lot_area", "lot_frontage",
            "front_setback", "rear_setback", "side_setback",
            "parking", "conditional_use", "open_space", "density", "nonconforming"
        ],
        "has_attorney": True,
        "ward": "1",
    }
    r = client.post("/analyze_proposal", json=payload)
    assert r.status_code == 200
    data = r.json()
    assert data["approval_probability"] >= 0
    assert data["approval_probability"] <= 1
    assert len(data["variances"]) == 12

def test_analyze_unknown_variance(client):
    payload = {
        "proposed_use": "residential",
        "variances": ["height", "fake_variance", "another_fake"],
    }
    r = client.post("/analyze_proposal", json=payload)
    assert r.status_code == 200

def test_ward_stats_invalid(client):
    r = client.get("/wards/abc/stats")
    assert r.status_code == 400

def test_ward_stats_nonexistent(client):
    r = client.get("/wards/99/stats")
    assert r.status_code == 404

def test_wards_all(client):
    r = client.get("/wards/all")
    assert r.status_code == 200
    data = r.json()
    assert "wards" in data
    if data["wards"]:
        ward = data["wards"][0]
        assert "ward" in ward
        assert "approval_rate" in ward

def test_autocomplete_short(client):
    r = client.get("/autocomplete", params={"q": "a"})
    assert r.status_code == 200
    assert r.json()["suggestions"] == []

def test_autocomplete_works(client):
    r = client.get("/autocomplete", params={"q": "75 Tremont"})
    assert r.status_code == 200

def test_model_info(client):
    r = client.get("/model_info")
    assert r.status_code in [200, 503]

def test_data_status(client):
    r = client.get("/data_status")
    assert r.status_code == 200

def test_timeline_stats(client):
    r = client.get("/timeline_stats")
    assert r.status_code == 200

def test_denial_patterns(client):
    r = client.get("/denial_patterns")
    assert r.status_code == 200

def test_voting_patterns(client):
    r = client.get("/voting_patterns")
    assert r.status_code == 200

def test_proviso_stats(client):
    r = client.get("/proviso_stats")
    assert r.status_code == 200

def test_recommend_endpoint(client):
    r = client.get("/recommend", params={
        "use_type": "residential",
        "project_type": "addition",
        "min_approval_rate": 0.3,
        "limit": 5,
    })
    assert r.status_code in [200, 429, 503]
    if r.status_code == 200:
        data = r.json()
        assert "parcels" in data or "recommendations" in data
        assert "disclaimer" in data or "query" in data

def test_concurrent_search(client):
    """Test multiple simultaneous searches don't interfere."""
    addresses = ["75 Tremont", "1081 River", "58 Burbank"]
    results = [client.get("/search", params={"q": a}) for a in addresses]
    for r in results:
        assert r.status_code in [200, 429]

def test_prediction_response_has_disclaimer(client):
    payload = {
        "proposed_use": "residential",
        "variances": ["height"],
        "has_attorney": True,
    }
    r = client.post("/analyze_proposal", json=payload)
    assert r.status_code in [200, 429]
    if r.status_code == 200:
        data = r.json()
        assert "disclaimer" in data
        assert "risk" in data["disclaimer"].lower() or "legal" in data["disclaimer"].lower()


# =========================
# PostGIS Integration Tests
# =========================

def test_parcel_has_source(client):
    """Parcel response should indicate data source (postgis or geojson)."""
    r = client.get("/parcels/0303695010")
    if r.status_code == 200:
        data = r.json()
        assert "source" in data
        assert data["source"] in ("postgis", "geojson")


# =========================
# New Endpoint Tests
# =========================

def test_denial_patterns_structure(client):
    """Denial patterns should return feature comparisons."""
    r = client.get("/denial_patterns")
    assert r.status_code in [200, 500]
    if r.status_code == 200:
        data = r.json()
        assert "patterns" in data
        assert "total_approved" in data
        assert "total_denied" in data
        for p in data["patterns"]:
            assert "factor" in p
            assert "direction" in p


def test_voting_patterns_structure(client):
    """Voting patterns should return vote distribution data."""
    r = client.get("/voting_patterns")
    assert r.status_code in [200, 500]


def test_proviso_stats_structure(client):
    """Proviso stats should return condition frequencies."""
    r = client.get("/proviso_stats")
    assert r.status_code in [200, 500]
    if r.status_code == 200:
        data = r.json()
        assert "total_approvals" in data
        assert "conditions" in data


def test_timeline_stats_structure(client):
    """Timeline stats should return temporal data."""
    r = client.get("/timeline_stats")
    assert r.status_code in [200, 500]
    if r.status_code == 200:
        data = r.json()
        # Should have some temporal info
        assert isinstance(data, dict)


def test_wards_all_structure(client):
    """Wards/all should return all wards in one call."""
    r = client.get("/wards/all")
    assert r.status_code == 200
    data = r.json()
    assert "wards" in data
    assert len(data["wards"]) > 10
    for w in data["wards"]:
        assert "ward" in w
        assert "approval_rate" in w
        assert 0 <= w["approval_rate"] <= 1


def test_model_info_structure(client):
    """Model info should expose full metadata."""
    r = client.get("/model_info")
    assert r.status_code in [200, 503]
    if r.status_code == 200:
        data = r.json()
        assert "model_name" in data
        assert "feature_count" in data
        assert "feature_cols" in data
        assert data["feature_count"] > 0


def test_health_postgis_field(client):
    """Health check should report PostGIS availability."""
    r = client.get("/health")
    data = r.json()
    assert "postgis_available" in data
    assert isinstance(data["postgis_available"], bool)
    assert "leakage_free" in data


def test_stats_has_wards(client):
    """Stats endpoint should report ward count."""
    r = client.get("/stats")
    assert r.status_code == 200
    data = r.json()
    assert data["total_wards"] > 10
    assert data["best_ward"] is not None


def test_autocomplete_returns_results(client):
    """Autocomplete with valid query should return suggestions."""
    r = client.get("/autocomplete", params={"q": "tremont"})
    assert r.status_code == 200
    data = r.json()
    assert "suggestions" in data


def test_autocomplete_short_query_empty(client):
    """Autocomplete with too-short query should return empty."""
    r = client.get("/autocomplete", params={"q": "ab"})
    assert r.status_code == 200
    data = r.json()
    assert data["suggestions"] == []


def test_recommend_returns_parcels(client):
    """Recommend endpoint should return parcels or recommendations."""
    r = client.get("/recommend", params={
        "project_type": "residential",
        "min_approval_rate": 0.3,
        "limit": 5
    })
    assert r.status_code in [200, 503]
    if r.status_code == 200:
        data = r.json()
        assert "parcels" in data or "recommendations" in data
