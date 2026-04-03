# PermitIQ API Quick-Start Guide

## Base URL

- **Local development:** `http://127.0.0.1:8000`
- **Production:** `https://api.permitiq.com` (replace with your deployment URL)
- **Interactive docs:** `{BASE_URL}/docs` (Swagger) or `{BASE_URL}/redoc` (ReDoc)

## Authentication

Authentication is optional in development. When `PERMITIQ_API_KEY` is set on the server, include the key in every request:

```
X-API-Key: your-api-key-here
```

Prediction endpoints (`/analyze_proposal`, `/batch_predict`, `/compare`) always require a key when one is configured. Read-only endpoints (search, zoning, stats) are exempt.

## Rate Limits

Default: **120 requests/minute** per IP. Configurable via `RATE_LIMIT_PER_MINUTE` env var. Localhost is exempt. When exceeded, the API returns `429 Too Many Requests`.

## Top 5 Endpoints

### 1. Search for an Address

Find ZBA case history for any Boston address.

```bash
curl "http://127.0.0.1:8000/search?q=75+Tremont+St"
```

Response (abbreviated):

```json
{
  "results": [
    {
      "address": "75 TREMONT ST",
      "cases": 14,
      "approval_rate": 0.62,
      "parcel_id": "0303456010",
      "case_history": [
        {"case_number": "BOA-1234567", "decision": "APPROVED", "year": 2024}
      ]
    }
  ]
}
```

### 2. Get Zoning Analysis for a Parcel

Returns zoning district, dimensional requirements, overlay districts, and auto-detected parcel issues.

```bash
curl "http://127.0.0.1:8000/zoning/0100001000"
```

Response (abbreviated):

```json
{
  "parcel_id": "0100001000",
  "zoning_district": "EBR-3",
  "subdistrict": "East Boston Residential-3",
  "requirements": {
    "max_height_ft": 35,
    "max_far": 0.5,
    "min_lot_area_sf": 5000,
    "min_frontage_ft": 50,
    "parking_per_unit": 1.0
  },
  "overlay_districts": ["GCOD"],
  "parcel_issues": {
    "auto_detected_variances": ["lot_area"],
    "violations": [
      {"type": "lot_area", "current": 3200, "required": 5000, "source": "property_assessment"}
    ]
  }
}
```

### 3. Check Compliance

Submit a proposal against a parcel to see which variances are needed.

```bash
curl -X POST "http://127.0.0.1:8000/zoning/check_compliance" \
  -H "Content-Type: application/json" \
  -d '{
    "parcel_id": "0100001000",
    "proposal": {
      "proposed_height_ft": 45,
      "proposed_far": 0.8,
      "proposed_units": 4,
      "proposed_stories": 3,
      "proposed_parking_spaces": 2
    }
  }'
```

Response (abbreviated):

```json
{
  "compliant": false,
  "parcel_level_variances": ["lot_area"],
  "proposal_level_variances": ["height", "far", "parking"],
  "violations": [
    {"type": "height", "proposed": 45, "allowed": 35, "excess": "28.6%"},
    {"type": "far", "proposed": 0.8, "allowed": 0.5, "excess": "60.0%"}
  ],
  "variance_history": {
    "height": {"approval_rate": 0.91, "total_cases": 4240},
    "far": {"approval_rate": 0.88, "total_cases": 3100}
  },
  "overlay_warnings": ["GCOD: Additional design review required"]
}
```

### 4. Get ML Prediction (Risk Assessment)

Predict approval probability for a specific proposal. Requires API key if configured.

```bash
curl -X POST "http://127.0.0.1:8000/analyze_proposal" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{
    "parcel_id": "0100001000",
    "use_type": "residential",
    "project_type": "new_construction",
    "variances": ["height", "far", "parking"],
    "ward": "1",
    "proposed_units": 4,
    "proposed_stories": 3,
    "has_attorney": true
  }'
```

Response (abbreviated):

```json
{
  "probability": 0.78,
  "probability_range": [0.72, 0.84],
  "confidence": "medium-high",
  "top_drivers": [
    {"feature": "Has legal representation", "impact": 0.12, "direction": "positive"},
    {"feature": "Height variance requested", "impact": -0.05, "direction": "negative"}
  ],
  "similar_cases": [
    {"case_number": "BOA-1234567", "decision": "APPROVED", "similarity": 0.89}
  ],
  "recommendations": [
    "Attorney representation increases approval odds by ~12pp",
    "Consider reducing to 3 units to eliminate density variance"
  ],
  "disclaimer": "This is a statistical risk assessment, not legal advice."
}
```

### 5. Compare What-If Scenarios

Test how changes to your proposal affect approval odds.

```bash
curl -X POST "http://127.0.0.1:8000/compare" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{
    "parcel_id": "0100001000",
    "use_type": "residential",
    "project_type": "new_construction",
    "variances": ["height", "far", "parking"],
    "ward": "1",
    "proposed_units": 4,
    "proposed_stories": 3
  }'
```

Response (abbreviated):

```json
{
  "baseline": {"probability": 0.72},
  "scenarios": [
    {"name": "Add attorney", "probability": 0.84, "delta": "+12pp"},
    {"name": "Reduce to 2 stories", "probability": 0.79, "delta": "+7pp"},
    {"name": "Remove parking variance", "probability": 0.76, "delta": "+4pp"},
    {"name": "Switch to renovation", "probability": 0.81, "delta": "+9pp"}
  ]
}
```

## Error Codes

| Code | Meaning |
|------|---------|
| 200  | Success |
| 400  | Bad request (missing/invalid parameters) |
| 401  | Unauthorized (missing or invalid API key) |
| 404  | Resource not found (parcel, address, case) |
| 429  | Rate limit exceeded |
| 500  | Internal server error |

All errors return JSON with a `detail` field:

```json
{"detail": "Parcel 9999999999 not found"}
```

## Additional Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/autocomplete?q=` | GET | Address autocomplete (175K properties) |
| `/address/{address}/cases` | GET | Full case history for an address |
| `/parcels/{parcel_id}` | GET | Parcel geometry and zoning details |
| `/nearby_cases/{parcel_id}` | GET | Nearby ZBA cases with distances |
| `/recommend?project_type=` | GET | Best parcels for a project type |
| `/batch_predict` | POST | Up to 50 predictions in one call |
| `/stats` | GET | Platform statistics |
| `/health` | GET | API health and model status |
| `/model_info` | GET | Model metadata, version, calibration |
| `/wards/all` | GET | All wards with approval stats |
| `/trends` | GET | Approval trends by year |
| `/variance_stats` | GET | Approval rates by variance type |
| `/attorneys/leaderboard` | GET | Top attorneys by win rate |
| `/neighborhoods` | GET | Approval rates by zoning district |
