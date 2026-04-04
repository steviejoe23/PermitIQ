# PermitIQ — Boston Zoning Intelligence Platform

## GOLDEN RULES — Read These First

1. **Never deploy or present a degraded version without warning Steven.** If Railway light mode is missing features, SAY SO before he discovers it. Compare deployed vs local capabilities explicitly.

2. **Every variance, every case, every stat must show historical data.** Never display a plain list like "Height — Your proposed building height." Always enrich with approval rates and case counts from `/variance_stats` (e.g., "Height: 91% approval, 4,240 ZBA cases"). This is the entire point of PermitIQ.

3. **Test the full flow before calling anything done.** Address search → zoning details → compliance check → ML prediction. If any step breaks, the product is broken.

4. **Steven expects autonomous execution.** Do the work, don't ask permission. He tests by clicking through the UI immediately — bugs are found fast.

5. **Maps must render.** The GeoJSON has Point centroids, NOT polygon boundaries. Use ScatterplotLayer for Points, PolygonLayer for Polygons. Check `geometry.type` before rendering.

6. **Never lose context.** If you're unsure of the current state, read the memory files and check git log before making changes. The memory system exists specifically to prevent context loss.

7. **The deployed product must match local quality.** If it can't, say what's missing and propose a fix — don't ship it silently.

## What Is PermitIQ?

PermitIQ predicts whether the Boston Zoning Board of Appeals (ZBA) will approve a development project — before the developer files. Developers spend $30–100K on permitting with no idea if they'll get approved. PermitIQ quantifies that risk using ML trained on every real ZBA decision from 2020–2026.

**Competitors** (UrbanForm, Zoneomics) tell you the zoning rules. **PermitIQ tells you if you'll win.**

**Key pitch:** "13,300+ real ZBA decisions, 85 ML features, no data leakage. We tell you if you'll win."

## Domain Glossary

| Term | Meaning |
|------|---------|
| **ZBA** | Zoning Board of Appeals — the 7-member Boston board that grants/denies variances and conditional use permits |
| **BOA** | Board of Appeals — same as ZBA. Case numbers start with "BOA-" |
| **Variance** | Permission to deviate from zoning code (e.g., build taller than allowed). 12 types: height, FAR, lot_area, lot_frontage, front/rear/side setback, parking, conditional_use, open_space, density, nonconforming |
| **Conditional Use** | A use allowed by zoning code only with ZBA approval (less risky than a variance) |
| **FAR** | Floor Area Ratio — total floor area ÷ lot area. Key density control |
| **Setback** | Required distance between building and lot line (front, rear, side) |
| **Subdistrict** | Granular zoning zone (e.g., "EBR-3" not just "3A-3C"). 286 unique from BPDA data |
| **Overlay District** | Additional rules layered on top of base zoning. We track: GCOD (Groundwater Conservation), Coastal Flood |
| **Building Appeal** | Appeal of ISD building code decision — only 58% approval vs 91% for zoning cases. Major risk factor |
| **BPDA** | Boston Planning & Development Agency — reviews large projects (Article 80) |
| **Proviso** | Condition attached to an approval (e.g., "must submit revised plans") |
| **Tracker** | City of Boston's ZBA case tracking system (data.boston.gov) — has filing dates, hearing dates, descriptions |

## Live Deployments

| Component | Platform | URL |
|-----------|----------|-----|
| **API** | Railway (1GB, LIGHT MODE) | https://overflowing-education-production-548c.up.railway.app |
| **Frontend** | Streamlit Cloud | https://permitiq-boston.streamlit.app |
| **Local** | Full features | API :8000, Frontend :8501 |
| **GitHub** | Private repo | https://github.com/steviejoe23/PermitIQ |

**CRITICAL: Railway is severely degraded.** Light mode (`PERMITIQ_LIGHT_MODE=1`) skips GeoJSON, ML model (507MB), property assessment, and tracker stats. It's a skeleton, not the real product. Local is the gold standard.

## Architecture

### Application Stack
```
api/
  main.py              — App shell: FastAPI, middleware, startup, router includes (195 lines)
  state.py             — Shared mutable state: gdf, zba_df, model_package, etc. (24 lines)
  utils.py             — Pure functions: normalize_address, safe_float/int/str, haversine (162 lines)
  constants.py         — VARIANCE_TYPES, PROJECT_TYPES, FEATURE_LABELS, DISCLAIMER (115 lines)
  api_models.py        — Pydantic request/response models (92 lines)
  services/
    data_loader.py     — Startup data loading, timeline stats, address index (266 lines)
    feature_builder.py — FEATURE_COLS: 85 features, shared with train_model_v2.py (132 lines)
    zoning_code.py     — Zoning requirements lookup by subdistrict (330 lines)
    database.py        — PostGIS queries, spatial index (145 lines)
    model_classes.py   — Custom model class definitions for pickle deserialization (95 lines)
    auth.py            — API key authentication middleware (370 lines)
    recommendations.py — Site selection scoring logic (809 lines)
  routes/
    search.py          — Address search, case history, autocomplete (231 lines)
    parcels.py         — Parcel lookup, nearby cases, geocode (277 lines)
    zoning.py          — Zoning details, compliance check, variance analysis (721 lines)
    prediction.py      — ML prediction, batch predict, what-if compare (1,029 lines)
    platform.py        — Stats, health, model info, data status (175 lines)
    recommend.py       — Site selection: "Where should I build?" (127 lines)
    market_intel.py    — 12 market intelligence endpoints (465 lines)
    attorneys.py       — Attorney search, profile, similar cases (413 lines)
frontend/
  app.py               — Streamlit frontend, dark theme, 4-step flow (~2,800 lines)
  requirements.txt     — Lightweight deps (streamlit, requests, pandas, pydeck)
```

### Data Pipeline
```
262 PDFs → OCR (PyMuPDF + Tesseract) → parse cases → build dataset
    → clean + dedup → extract features (regex) → integrate external data → fuzzy match
    → train model (stacking ensemble) → deploy to api/zba_model.pkl
```

### Key Data Files
| File | Size | Description |
|------|------|-------------|
| zba_cases_cleaned.csv | 30MB | 7,500+ cases, 85 features each |
| boston_parcels_zoning.geojson | 147MB | 98,510 parcels with **Point centroids** (NOT polygons) |
| api/zba_model.pkl | ~219MB | Stacking ensemble model package |
| property_assessment_fy2026.csv | 80MB | Boston property tax assessments (184K records) |
| building_permits.csv | 259MB | All Boston building permits (718K records) |
| zba_tracker.csv | 6MB | ZBA case tracker from data.boston.gov (15,932 records) |

## API Endpoints (34 total)

### Search (3)
- `GET /search?q=<address>` — Fuzzy address search with approval rates, case counts, parcel_ids
- `GET /address/{address}/cases` — Full ZBA case history
- `GET /autocomplete?q=<prefix>` — Address autocomplete from 175K property records

### Parcels (3)
- `GET /parcels/{parcel_id}` — Zoning details + Point geometry for mapping
- `GET /parcels/{parcel_id}/nearby_cases` — ZBA cases within 0.5 miles, sorted by distance
- `GET /geocode?address=<addr>` — Address → parcel_id

### Zoning & Compliance (4)
- `GET /zoning/{parcel_id}` — Subdistrict requirements, allowed uses, area approval rate, auto-detected `parcel_issues`
- `POST /zoning/check_compliance` — Proposal vs zoning limits → `parcel_level_variances` + `proposal_level_variances` with historical rates
- `POST /zoning/full_analysis` — Combined zoning + compliance
- `POST /zoning/variance_analysis` — Deep variance type analysis

### Prediction (3)
- `POST /analyze_proposal` — Core ML prediction: probability, confidence interval, SHAP drivers, similar cases, recommendations, variance_history
- `POST /compare` — What-if scenarios (4-6 scenario cards with probability deltas)
- `POST /batch_predict` — Batch predictions (up to 50)

### Market Intelligence (12)
- `GET /variance_stats` — Per-variance approval rates and case counts (**used by frontend for enrichment**)
- `GET /project_type_stats` — Rates by project type
- `GET /trends` — Yearly approval trends (2021-2026)
- `GET /neighborhoods` — 26 zoning districts ranked
- `GET /denial_patterns` — Denied vs approved distinguishing factors
- `GET /voting_patterns` — Vote distribution analysis
- `GET /proviso_stats` — Common approval conditions
- `GET /timeline_stats` — Filing-to-decision timeline (median 142 days)
- `GET /wards/{ward_id}/stats` — Ward-level stats
- `GET /wards/all` — All ward stats in one call
- `GET /wards/{ward_id}/trends` — Ward trends over time
- `GET /wards/{ward_id}/top_attorneys` — Top attorneys per ward

### Attorneys (4)
- `GET /attorneys/search?q=<name>` — Fuzzy attorney search
- `GET /attorneys/{name}/profile` — Win rate, wards, specialties, vs-average comparison
- `GET /attorneys/{name}/similar_cases` — Filtered by ward/variance
- `GET /attorneys/leaderboard` — Top attorneys ranked

### Platform (4)
- `GET /stats` — Dashboard stats
- `GET /health` — Model status, AUC, uptime
- `GET /model_info` — Model metadata, calibration buckets with trust ratings
- `GET /data_status` — Data freshness, pipeline status

### Site Selection (1)
- `GET /recommend?use_type=<type>&project_type=<type>` — ML-ranked parcels

## ML Model

- **Architecture:** Stacking ensemble — XGBoost_Deep + Gradient Boosting + Random Forest, balanced Logistic Regression meta-learner, 5-fold out-of-fold
- **85 features** across 14 categories (see `api/services/feature_builder.py` for full list)
- **Test AUC:** 0.7998 | **Honest CV AUC:** 0.7921 | **Denial Recall:** 69.7% | **ECE:** 1.0%
- **Calibration:** Platt scaling on separate holdout. 90-100% bucket: predicted 95.8%, actual 95.4%
- **Size:** 219MB (too large for Railway 1GB — needs lightweight retrain or paid plan)
- **CRITICAL: No post-hearing features.** 14 features removed for data leakage (see `feature_builder.py` REMOVED_FEATURES list)

## Frontend: 4-Step Flow

Each step unlocks the next:

1. **Address Search** → Color-coded results (green/yellow/red by approval rate), expandable case history, parcel_id linking
2. **Zoning & Parcel Details** → Subdistrict requirements, PyDeck map (ScatterplotLayer for Points), auto-detected parcel issues with historical rates, nearby ZBA cases
3. **Compliance Check** → User enters proposal → violations with parcel-level vs proposal-level breakdown, each enriched with approval rate + case count
4. **ML Prediction** → Probability gauge, confidence interval, SHAP drivers (top 8, human-readable labels), stratified similar cases (includes denied), what-if scenarios, downloadable HTML report

**Other sections:** Ward Insights, Market Intelligence (5 tabs), Attorney Lookup (search + profile + CSV export), Site Selection

## Common Pitfalls — Things That Have Gone Wrong Before

### Map Crashes
The GeoJSON contains **Point** centroids, not Polygon boundaries. Code that does `coordinates[0][0]` will crash with `'float' object is not subscriptable`. Always check `geometry.type` first:
- Point → `lon, lat = coords[0], coords[1]` → ScatterplotLayer
- Polygon → `coords[0]` is a ring of [lon, lat] pairs → PolygonLayer

### Variance Display
Never show variances as plain text. Every variance mention in the UI must include its historical approval rate and case count from `/variance_stats`. This applies to:
- Auto-detected parcel issues (Step 2)
- Proposal-dependent checks (Step 2)
- Compliance violations (Step 3)
- Variance selection hints (Step 4)

### Railway Light Mode
The 507MB model + 147MB GeoJSON + 80MB property data = OOM on Railway's 1GB. Light mode disables all of these. The fallback heuristic prediction is NOT the real model. Don't present Railway output as if it has real ML predictions.

### Feature Leakage
The model uses ONLY pre-hearing features. If you're tempted to add a feature, ask: "Would a developer know this BEFORE the hearing?" If not, it's leakage. See `feature_builder.py` for the safe/removed lists.

### Compliance Checker Parsing
`/zoning/check_compliance` accepts either flat params or a nested `proposal` object. If you change the API contract, test both formats. Previously, nested proposals silently caught 1/6 violations.

### Address Normalization
`normalize_address()` in `utils.py` handles: "Ave" vs "Av", city/zip suffixes, directionals (East→E), suffix letters (18R), em-dashes, range addresses (55-57). If you touch this function, you affect geocoding for 7,451 addresses.

### Frontend API URL
```python
_DEFAULT_API = "https://overflowing-education-production-548c.up.railway.app"
API_URL = os.environ.get("PERMITIQ_API_URL") or st.secrets.get("PERMITIQ_API_URL", _DEFAULT_API)
```
For local dev: `export PERMITIQ_API_URL=http://127.0.0.1:8000` or start the API and it'll use the Railway URL.

## How to Run

### Quick Start
```bash
cd ~/Desktop/Boston\ Zoning\ Project
source zoning-env/bin/activate
make run          # Starts API + Frontend (ports 8000 + 8501)
```

### Manual Start
```bash
# Terminal 1: API
cd ~/Desktop/Boston\ Zoning\ Project/api
uvicorn main:app --reload --port 8000

# Terminal 2: Frontend
cd ~/Desktop/Boston\ Zoning\ Project/frontend
streamlit run app.py --server.port 8501
```

### Other Commands
```bash
make test           # Integration tests (API must be running)
make retrain        # Retrain model from existing dataset
make retrain-clean  # Clean OCR → audit → retrain
make docker         # Start with Docker Compose (includes PostGIS)
make push           # Commit and push to GitHub
python3 train_model_v2.py  # Retrain and save to model_history/ with auto-comparison
```

### Update Data
```bash
python3 auto_update_data.py       # Pull fresh CSVs from boston.gov CKAN API
python3 auto_scrape_decisions.py  # Scrape new ZBA decision PDFs
```

## Environment
- **Python:** 3.9 (virtual env: `zoning-env/`)
- **Key packages:** fastapi, uvicorn, streamlit, scikit-learn, xgboost, pandas, numpy, pymupdf, pytesseract, pydeck, requests, beautifulsoup4, joblib, pydantic
- **Tesseract:** `/opt/homebrew/bin/tesseract`
- **PostgreSQL 18:** port 5432, password in PGPASSWORD env var (PostGIS optional — API falls back to in-memory GeoJSON)
- **Git LFS:** Not installed. LFS hooks removed from `.git/hooks/`. Don't add files >100MB to git.

## Demo Info
- **Best demo addresses (have parcel IDs, full flow works):**
  - 105 Norwell St → 15 cases, 100% approval (safe bet — Parcel 1401643000)
  - 1001 Boylston St → 12 cases, 0% approval (danger zone — Parcel 0504155010)
  - 4 Anawan Ave → 6 cases, 50% approval (coin flip)
  - 605 Tremont St → 6 cases, 67% approval (moderate risk)
  - 152 Hampden St → 6 cases, 83% approval (decent odds)
- **Note:** Model predictions cluster 90-96% because the base approval rate is 90.4%. Variation shows in what-if deltas and key factors, not in wildly different base probabilities.
- **Sample parcel IDs:** 1401643000 (Dorchester), 0504155010 (Back Bay), 0701823000 (South Boston)

## Testing Checklist

Before calling any change "done," verify:
- [ ] Address search returns results with approval rates (not empty or error)
- [ ] Parcel lookup shows zoning details and map renders (ScatterplotLayer for Points)
- [ ] Auto-detected parcel issues show historical approval rates (not plain text)
- [ ] Compliance check returns violations with parcel-level + proposal-level breakdown
- [ ] Prediction returns probability, SHAP drivers, similar cases (includes denied)
- [ ] What-if scenarios render with color-coded deltas
- [ ] All variance displays enriched with approval rate + case count
- [ ] No Python tracebacks in the Streamlit UI
- [ ] If deploying: compare Railway vs local output — flag any missing features

## GitHub Repo Status (April 4, 2026)
- **Cleaned for portfolio use** — 54 files, production code only
- CLAUDE.md is in .gitignore (local only, not visible on GitHub)
- Data files (CSV, GeoJSON, pkl) are in .gitignore — stay local
- No internal docs, leads, or AI context files on GitHub
- README is portfolio-grade with architecture diagram, metrics, API examples

## Known Issues (April 2026)
- Railway deployment missing most features (light mode) — needs smaller model or paid plan
- GeoJSON has Point centroids only — no parcel polygon boundaries
- Building appeal predictions ~10pp overconfident (calibration warning added)
- 490 denied cases have no variance data
- No lot_frontage in property assessment — can't auto-detect frontage violations
- Predictions below 50% have limited calibration data (31 test cases)
- Customer segment and pricing model undefined
- Boston-only TAM may be too small for venture scale
