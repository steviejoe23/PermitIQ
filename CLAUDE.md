# PermitIQ — Boston Zoning Board of Appeals Prediction Platform

## Project Vision
PermitIQ predicts whether the Boston Zoning Board of Appeals (ZBA) will approve a development project — before the developer files. Developers currently spend $30–100K on permitting with no idea if they'll get approved. PermitIQ quantifies that risk using ML trained on every real ZBA decision from 2020–2026.

Competitors (UrbanForm, Zoneomics) tell you the zoning rules. PermitIQ tells you if you'll win.

## What Exists Today
- **262 ZBA decision PDFs** scraped from boston.gov (OCR'd via Tesseract into structured data)
- **7,500+ unique ZBA cases** with 69 engineered features each
- **FastAPI backend** with ML prediction, parcel lookup, address search, and what-if scenario comparison
- **Streamlit frontend** — polished, demo-ready UI with stats dashboard, interactive maps, prediction panel, and downloadable reports
- **Trained ML model** (Gradient Boosting / Random Forest / Logistic Regression — best is auto-selected by AUC)
- **External data integration** — Boston property assessments (184K parcels), building permits (718K records), ZBA tracker (15K records)
- **98,510 Boston parcels** with zoning data from GeoJSON

## Architecture & File Map

### Data Pipeline (runs in sequence)
```
262 PDFs (pdfs/*.pdf)
    → zba_pipeline/extract_text.py    (OCR via PyMuPDF + Tesseract, 300 DPI, 400 DPI retry)
    → zba_pipeline/parse_cases.py     (split PDFs into individual BOA cases)
    → zba_pipeline/build_dataset.py   (orchestrator with checkpointing → zba_cases_dataset.csv)
    → rebuild_dataset.py              (dedup + clean decisions → zba_cases_cleaned.csv)
        → reextract_features.py       (regex feature extraction from raw_text: addresses, variances, votes, articles, provisos)
        → integrate_external_data.py  (merge ZBA Tracker + Property Assessment + Building Permits via case# and address)
        → fuzzy_match_properties.py   (fuzzy address matching for unmatched cases)
    → train_model_v2.py               (train 3 models, pick best → zba_model_v2.pkl + api/zba_model.pkl)
```

### Application Stack
```
api/main.py          — FastAPI backend (port 8000), 1053 lines
frontend/app.py      — Streamlit frontend (port 8501), 796 lines
api/zba_model.pkl    — Serialized model package (model + feature_cols + ward/zoning rates + metadata)
```

### Automation Scripts
```
auto_scrape_decisions.py  — Scrapes boston.gov for new ZBA decision PDFs (Google Drive + direct links)
auto_update_data.py       — Pulls fresh data from data.boston.gov CKAN API (weekly cron)
overnight_rebuild.py      — Full pipeline: OCR all PDFs → rebuild → retrain (run before bed)
```

## Key Files in Detail

### api/main.py (FastAPI Backend)
- **Endpoints (13 total, tagged in Swagger docs at /docs):**
  - **Search:** `GET /search?q=<address>` — Fuzzy address search with LRU cache (256 entries), pre-computed address normalization
  - **Search:** `GET /address/{address}/cases` — Full ZBA case history for an address
  - **Parcels:** `GET /parcels/{parcel_id}` — Parcel zoning details + geometry for mapping
  - **Prediction:** `POST /analyze_proposal` — Core ML prediction with confidence intervals, probability range, feature contribution analysis (top_drivers)
  - **Prediction:** `POST /compare` — What-if scenario analysis (tests 4–6 scenarios with real probability deltas)
  - **Market Intel:** `GET /wards/{ward_id}/stats` — Ward-level approval statistics
  - **Market Intel:** `GET /variance_stats` — Approval rates by variance type
  - **Market Intel:** `GET /project_type_stats` — Approval rates by project type
  - **Market Intel:** `GET /neighborhoods` — Approval rates by zoning district/neighborhood (26 districts)
  - **Market Intel:** `GET /attorneys/leaderboard` — Top attorneys ranked by ZBA win rate
  - **Market Intel:** `GET /trends` — Approval rate trends by year (2021–2026)
  - **Platform:** `GET /stats` — Overall platform statistics for dashboard
  - **Platform:** `GET /health` — Model status, dataset counts, AUC
- **Interactive API docs** at `http://127.0.0.1:8000/docs` (Swagger) and `/redoc` (ReDoc)
- **Feature builder** (`build_features()`) constructs a 69-dimensional feature vector from pre-hearing data only (no data leakage)
- **Loads on startup:** GeoJSON (98K parcels), zba_cases_cleaned.csv, zba_model.pkl, pre-computed address normalization
- **Structured logging** with timestamps, request timing middleware (logs method, path, status, ms)
- Uses Pydantic `ProposalInput` model for input validation; accepts `proposed_use`, `use_type`, or `use` field names
- Safe type helpers (`safe_float`, `safe_int`, `safe_str`) prevent crashes on NaN/None

### frontend/app.py (Streamlit UI)
- **Stats dashboard** — ZBA decisions count, parcels mapped, avg approval rate, wards covered, ML features
- **Clickable sidebar** — Demo addresses and parcels are buttons that auto-load results
- **Address search** — Color-coded approval rates (green ≥70%, yellow 40-70%, red <40%) with expandable case history drill-down
- **Parcel lookup** — Zoning details + interactive PyDeck map with parcel boundaries
- **Prediction panel** — Full input form (parcel, use type, project type, variances, ward, attorney, units, stories)
  - Variance approval hints shown when selecting variances (historical rates)
  - Output: probability, probability range (confidence interval), confidence badge, proposal summary, key factors, similar cases
  - Model Explainability section — top 8 feature drivers for this specific prediction
- **What-If scenarios** — Model-computed scenario cards with color-coded deltas
- **Report export** — Downloadable HTML report (styled, professional) + plain text fallback
- **Ward Insights** — Compare approval rates by ward
- **Market Intelligence** (expandable section with 5 tabs):
  - Approval Trends — bar chart by year
  - Variance Success Rates — color-coded approval rates for all 12 variance types
  - Project Type Rates — approval rates by project type
  - Attorney Leaderboard — top attorneys by win rate with W-L records
  - Neighborhoods — 26 zoning districts ranked by approval rate
- Dark theme (#1a1a2e), custom CSS, animated buttons, responsive layout
- XSS protection — all API data HTML-escaped before rendering
- `API_URL` configurable via `PERMITIQ_API_URL` env var (defaults to `http://127.0.0.1:8000`)

### train_model_v2.py (ML Training)
- **69 features across 14 categories:**
  - Variance types (13): height, FAR, lot_area, setbacks, parking, conditional_use, etc.
  - Violation types (5): excessive_far, insufficient_lot/frontage/yard/parking
  - Use type (2): is_residential, is_commercial
  - Representation (4): has_attorney, community_process, has_opposition, no_opposition_noted
  - Planning (3): planning_proviso, bpda_involved, planning_support
  - Project types (12): demolition, new_construction, addition, conversion, renovation, etc.
  - Political (5): councilor_involved, mayors_office, support/opposition/non-opposition letters
  - Legal (7): article_7/8/51/65/80, is_conditional_use, is_variance
  - Scale (2): proposed_units, proposed_stories
  - Location (2): ward_approval_rate, zoning_approval_rate (smoothed target encoding)
  - Complexity (4): text_length_log, num_articles, num_sections, year_recency
  - Deferrals (2): has_deferrals, num_deferrals
  - Property (6): lot_size_sf, total_value, property_age, living_area, is_high_value, value_per_sqft
  - Permits (2): prior_permits, has_prior_permits
- **3 models trained:** Gradient Boosting (with sample weights for class imbalance), Random Forest (class_weight='balanced'), Logistic Regression (class_weight='balanced')
- **Temporal train/test split:** Trains on older years, tests on most recent year (honest evaluation)
- **Confusion matrix output** for each model
- **5-fold cross-validation** on best model
- Best model auto-selected by AUC, saved to zba_model_v2.pkl + api/zba_model.pkl

### reextract_features.py (Feature Extraction)
- 398 lines of regex-based feature extraction from OCR raw_text
- Extracts: addresses (multi-pattern), applicant names, permit numbers, votes, zoning articles, units/stories, filing dates
- 12 variance type patterns, 8 proviso types
- Outputs enhanced columns in zba_cases_cleaned.csv

### integrate_external_data.py (Data Integration)
- Merges 3 external Boston datasets:
  - ZBA Tracker (15,932 records) — matched by BOA case number
  - Property Assessment FY2026 (184,552 records) — matched by normalized address
  - Building Permits (718,721 records) — aggregated prior permits per address
- Adds: lot_size_sf, total_value, property_age, living_area, value_per_sqft, prior_permits, has_prior_permits, is_high_value

### fuzzy_match_properties.py (Address Matching)
- Handles edge cases: range addresses (85-99 St), suffix letters (18R St), directionals (East→E), city/zip in address
- Fills ~10-15% more property data via variant matching

### overnight_rebuild.py (Full Pipeline Runner)
- Steps: OCR all PDFs → rebuild dataset → retrain model → deploy to API
- **Checkpointing:** Saves progress after each PDF in ocr_checkpoint.json. Can resume with `python3 overnight_rebuild.py` (no --fresh). Fresh run: `python3 overnight_rebuild.py --fresh`
- **No timeouts** — runs until complete
- OCR at 300 DPI with 400 DPI retry on empty pages

## Current Status (March 24, 2026)

### What's Running Right Now
- OCR rebuild is running (`overnight_rebuild.py --fresh`) — processing all 262 PDFs from scratch
- As of last check, ~32/262 PDFs done, ~2 min per scanned OCR PDF, ~0s per hybrid PDF
- Checkpoint file saves progress after each PDF — safe to resume if interrupted

### Recent Changes Made Today

**Earlier today (before this session):**
1. Bug fix: `district` variable undefined in `build_features()` — fixed
2. Bug fix: Three silent `except: pass` blocks now show errors
3. Model: Added temporal train/test split, class imbalance handling, confusion matrices
4. OCR: Added progress indicators (every 25 pages)
5. Pipeline: Added checkpoint system for overnight_rebuild.py

**This session — API improvements:**
6. Structured logging — all print→logger with timestamps, request timing middleware
7. LRU search cache (256 entries) + pre-computed address normalization at startup
8. Fixed `/compare` crash on non-numeric input (safe_int)
9. Fixed `use_type` alias not recognized in `/analyze_proposal` and `/compare`
10. Added confidence intervals (`probability_range`) to prediction response
11. Added feature contribution analysis (`top_drivers`) to prediction response
12. Fixed inconsistent error response formats (dict→HTTPException)
13. Fixed hardcoded `text_length_log`/`year_recency` in `build_features()`
14. Tagged all 13 endpoints for Swagger docs (`/docs`, `/redoc`)

**This session — new API endpoints:**
15. `GET /attorneys/leaderboard` — top attorneys by win rate
16. `GET /trends` — approval rate trends by year
17. `GET /neighborhoods` — 26 districts ranked by approval rate

**This session — frontend improvements:**
18. XSS fix — all API data HTML-escaped before rendering
19. `API_URL` env-configurable via `PERMITIQ_API_URL`
20. Clickable sidebar — demo addresses/parcels are buttons that auto-load results
21. Case history drill-down — expandable case list under each search result
22. Variance approval hints — historical rates shown when selecting variances
23. Model Explainability section — top 8 feature drivers for each prediction
24. Market Intelligence section (5 tabs): Trends, Variance Rates, Project Types, Attorney Leaderboard, Neighborhoods
25. HTML report export — professional styled report alongside plain text

**This session — training pipeline (train_model_v2.py):**
26. Fixed target encoding data leakage — ward/zoning rates now computed from training data only
27. Added calibration metrics (Brier score, log loss, calibration curves)
28. Added threshold optimization (sweeps 0.3–0.7, finds optimal F1)
29. Switched to stratified 5-fold CV
30. Saves `model_diagnostics.png` (ROC curves, calibration curves, feature importance)
31. Model package now includes `brier_score`, `optimal_threshold`

**This session — cleanup:**
32. Deleted orphaned `api/analyzer.py` and `api/model_loader.py`

**Session 2 — CRITICAL: Feature Leakage Fix:**
33. REMOVED 14 post-hearing features from training + API (has_opposition, no_opposition_noted, planning_support, planning_proviso, community_process, support_letters, opposition_letters, non_opposition_letter, councilor_involved, mayors_office_involved, hardship_mentioned, has_deferrals, num_deferrals, text_length_log)
34. Fixed calibration — now uses separate calibration holdout (was calibrating on test set)
35. Added 3-way split: train/test/calibration
36. Fixed has_attorney regex — was catching 1% of cases, now catches ~24%
37. Added attorney_win_rate as new feature (smoothed target encoding from training data)
38. Added case deduplication to training pipeline
39. Added decision recovery from raw_text (fixes ~34% missing decisions)

**Session 2 — New Features:**
40. `GET /wards/all` — single endpoint replacing 22 sequential calls
41. `GET /recommend` — Site selection: find best parcels for your project type
42. `GET /denial_patterns` — What distinguishes denied vs approved cases
43. `GET /voting_patterns` — Vote distribution analysis
44. `GET /proviso_stats` — Common conditions attached to approvals
45. `GET /timeline_stats` — Filing-to-decision timeline analysis
46. `GET /autocomplete` — Address autocomplete from 175K property records
47. `GET /model_info` — Full model metadata and version info
48. `GET /data_status` — Data freshness and OCR pipeline status
49. `POST /batch_predict` — Batch predictions (up to 50)

**Session 2 — Infrastructure:**
50. API key authentication (optional, set PERMITIQ_API_KEY env var)
51. Rate limiting (configurable via RATE_LIMIT_PER_MINUTE, default 60/min)
52. Shared feature_builder.py module (single source of truth for feature list)
53. Market intel router (api/routes/market_intel.py) extracted from monolith
54. Frontend startup caching (@st.cache_data, batched API calls)
55. Model versioning — saves to model_history/ with JSONL training log + auto-comparison
56. Docker Compose improved (env vars, restart policy, 2 workers)
57. Makefile: added `make retrain-clean`, `make audit`, `make docker`
58. OCR quality audit script (audit_ocr_quality.py) — scored 35/100
59. OCR cleanup script (cleanup_ocr.py) — address normalization, decision recovery, dedup
60. 63 tests passing (up from 33)

**Session 2 — Product Reframing:**
61. Rebranded from "prediction" to "risk assessment" throughout UI and API
62. Comprehensive liability disclaimers on all prediction outputs and reports
63. API version bumped to 3.0

### Known Issues / TODO
- ~~Frontend `API_URL` is hardcoded to localhost~~ → FIXED
- ~~No structured logging anywhere~~ → FIXED
- ~~No caching for address searches~~ → FIXED
- ~~Model doesn't save diagnostic plots~~ → FIXED
- ~~`analyzer.py` and `model_loader.py` orphaned~~ → FIXED
- ~~Target encoding data leakage~~ → FIXED
- ~~No rate limiting on API~~ → FIXED (configurable)
- ~~14 post-hearing features leaking~~ → FIXED (removed)
- ~~Calibration on test set~~ → FIXED (separate holdout)
- ~~has_attorney catching only 1%~~ → FIXED (broader regex)
- ~~No model versioning~~ → FIXED (model_history/)
- ~~No API authentication~~ → FIXED (optional API key)
- ~~Ward heatmap 22 API calls~~ → FIXED (/wards/all)
- ~~No site selection feature~~ → FIXED (/recommend)
- AUC will drop after retrain with clean features (expected: 0.70-0.82, was inflated to 0.94 by leakage)
- API monolith is partially split (market_intel router created, main.py still large)
- `proj_*` columns not in current CSV (will be added after OCR retrain)
- Need manual OCR quality audit (sample 50 cases vs original PDFs)
- Customer segment and pricing model undefined (business decision)
- Boston-only TAM too small for venture scale (expansion strategy needed)

## How to Run Everything

### Quick start (Makefile):
```bash
cd ~/Desktop/Boston\ Zoning\ Project
make run          # Starts API + Frontend
make test         # Run 63 tests (requires API running)
make retrain      # Retrain model from existing dataset
make retrain-clean # Clean OCR → audit → retrain (after OCR completes)
make audit        # Run OCR quality audit
make status       # Check OCR progress + data freshness
make docker       # Start with Docker Compose
```

### Manual start:
```bash
cd ~/Desktop/Boston\ Zoning\ Project
source zoning-env/bin/activate

# Terminal 1: API
cd api && uvicorn main:app --reload --port 8000

# Terminal 2: Frontend
cd ~/Desktop/Boston\ Zoning\ Project/frontend
streamlit run app.py --server.port 8501
```

### After OCR finishes (CRITICAL — run in order):
```bash
python3 cleanup_ocr.py           # Fix OCR artifacts, recover decisions
python3 audit_ocr_quality.py     # Check quality score
python3 train_model_v2.py        # Retrain with clean features
# Or: make retrain-clean
```

### Full rebuild from scratch:
```bash
caffeinate -s make rebuild
```

### Just retrain the model:
```bash
python3 train_model_v2.py        # Saves to model_history/ with auto-comparison
```

### Update data from boston.gov:
```bash
python3 auto_update_data.py      # pulls fresh CSVs from CKAN API
python3 auto_scrape_decisions.py  # scrapes new decision PDFs
```

## Key Data Files
| File | Size | Description |
|------|------|-------------|
| zba_cases_dataset.csv | 40MB | Raw OCR output (all cases from all PDFs) |
| zba_cases_cleaned.csv | 30MB | Cleaned + feature-enriched dataset |
| zba_model_v2.pkl | 2MB | Trained model package |
| api/zba_model.pkl | 2MB | Copy deployed to API |
| boston_parcels_zoning.geojson | 147MB | All 98,510 Boston parcels with zoning |
| property_assessment_fy2026.csv | 80MB | Boston property tax assessments |
| building_permits.csv | 259MB | All Boston building permits |
| zba_tracker.csv | 6MB | ZBA case tracker from boston.gov |
| ocr_checkpoint.json | varies | Checkpoint file (deleted on completion) |

## Python Environment
- **Virtual env:** `zoning-env/` (Python 3.9 via CommandLineTools)
- **Key packages:** fastapi, uvicorn, streamlit, scikit-learn, pandas, numpy, pymupdf (fitz), pytesseract, pydeck, requests, beautifulsoup4, joblib, pydantic
- **Tesseract path:** `/opt/homebrew/bin/tesseract`

## Demo Info
- **Demo addresses:** 75 Tremont St (14 cases, 62%), 1081 River St (13 cases, 73%), 58 Burbank St (10 cases, 44%)
- **Sample parcel IDs:** 0100001000 (East Boston, 3A-3C), 0302951010 (South Boston), 1000358010 (Jamaica Plain)
- **Key pitch:** "7,500+ real ZBA decisions, 69 features, no data leakage. We tell you if you'll win."
- **Business case:** Developers spend $30-100K with no idea of outcome. PermitIQ quantifies the risk before filing.
