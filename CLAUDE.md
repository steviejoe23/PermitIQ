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

**Session 3 — Foundational Fixes (March 25, 2026):**
64. CRITICAL: Removed 14 post-hearing features from training + API (data leakage fix)
65. Fixed model calibration — Platt scaling on separate holdout (was calibrating on test set)
66. Fixed has_attorney regex — was catching 1%, now catches ~24%
67. Added attorney_win_rate as new feature (smoothed target encoding)
68. Added case deduplication to training pipeline
69. OCR quality audit script (audit_ocr_quality.py) — scored 35/100
70. 3-way split: train/test/calibration holdout

**Session 3 — Infrastructure:**
71. GitHub repo: https://github.com/steviejoe23/PermitIQ (private)
72. PostGIS: 88,839 parcels imported, spatial indexes, API queries DB first
73. API monolith split: market_intel router extracted (794 lines removed, main.py 2,027 → 1,600)
74. Shared feature_builder.py module (single source of truth)
75. API key auth on prediction endpoints (optional via PERMITIQ_API_KEY)
76. Docker Compose with PostGIS service
77. 75 tests passing (up from 63)
78. Post-OCR retrain script (post_ocr_retrain.sh)

**Session 3 — New Features:**
79. Site Selection panel in frontend ("Where should I build?")
80. /recommend endpoint with ML-ranked parcels
81. /denial_patterns — what distinguishes denied vs approved
82. /timeline_stats — filing-to-decision analysis
83. /wards/all — single call replaces 22 sequential calls
84. /stats, /autocomplete, /model_info restored after monolith split

**Session 4 — Model Accuracy Improvements (March 28, 2026):**
85. Downloaded 11 missing PDFs from boston.gov (275 total, up from 263)
86. OCR processing 11 new PDFs (~700+ new cases)
87. XGBoost added as model candidate (pip install xgboost)
88. Composite model selection: 0.6*AUC + 0.4*DenialRecall (prevents "everything approved" models)
89. `is_building_appeal` feature — Building appeals 58% vs Zoning 91% approval (#5 feature)
90. Full year coverage from tracker dates — year_recency went from 55% → 100% coverage
91. Broader project type patterns for tracker descriptions (change of occupancy, erect, etc.)
92. Better units/stories extraction from tracker descriptions (1,235 units, 1,718 stories extracted)
93. 3 meta-features: project_complexity, total_violations, num_features_active
94. Smart dedup: OCR cases preferred over tracker (richer features)
95. Feature richness weighting: tracker cases get 0.3x weight
96. Model improved: AUC 0.7269 → 0.7665, CV AUC 0.7320 → 0.7706, Brier 0.0957 → 0.0912
97. 61 features (up from 57), 13,308 unique cases, Gradient Boosting winner

**Session 5 — Model Architecture & Honest Evaluation (March 28, 2026):**
98. Stacking ensemble — XGBoost_Deep + GB + RF with balanced LR meta-learner, 5-fold OOF
99. Feature selection — removes 22 noise features (<0.002 importance), auto-retrains
100. XGBoost_Deep variant — deeper trees (max_depth=7), more regularization
101. Honest CV — recomputes target encoding within each fold (reveals 0.06 inflation in simple CV)
102. Simple CV AUC: 0.8284, **Honest CV AUC: 0.7686** (the true generalizable performance)
103. **Test AUC: 0.7728** (stacking ensemble, +0.0063 over best individual model)
104. **Denial Recall: 63.2%** (catches 2/3 of denials, up from 56.7% individual model)
105. Brier Score: 0.0906 (well-calibrated after Platt scaling)
106. Manual Platt scaling fallback for custom model classes
107. API updated: contact_win_rate, ward_zoning_rate, year_ward_rate now served from model package
108. feature_builder.py synced to 70 features (was 65)
109. New features: prior_permits_log, contact_x_appeal, attorney_x_building, many_variances, has_property_data
110. Model package now saves all rate dictionaries (contact_win_rates, ward_zoning_rates, year_ward_rates)
111. Tested and rejected: LOO target encoding (killed signal — AUC dropped to 0.58), higher smoothing (reduced signal)

**Session 6 — Data Quality & Product Polish (March 29, 2026):**
112. Scraped 1,485 ZBA hearing agendas from boston.gov (scrape_zba_agendas.py) — pre-hearing variance data
113. Improved variance extraction for denied cases (reextract_denied_variances.py) — +159 denied cases filled
114. Inferred variances from tracker descriptions + zoning limits (infer_denied_variances.py) — +461 denied cases
115. Variance coverage: 47% → 69% overall, denied 44% → 74%
116. **Model: AUC 0.7727 → 0.7987 (+0.026), Honest CV 0.7686 → 0.7910, Denial Recall 63.2% → 69.7%**
117. Downloaded BPDA Zoning Subdistricts (1,640 polygons, 286 unique subdistricts)
118. Rebuilt boston_parcels_zoning.geojson with granular subdistricts (EBR-3 not "3A-3C")
119. /zoning/{parcel_id} now returns subdistrict, subdistrict_type, neighborhood, data_source
120. /zoning/check_compliance rewritten — uses SAME subdistrict requirements as /zoning endpoint (was inconsistent)
121. _get_parcel_zoning() — single source of truth for all zoning endpoints
122. Overlay districts: GCOD (10,069 parcels) and Coastal Flood (9,195 parcels) flagged in GeoJSON and API
123. /zoning/check_compliance returns overlay_warnings for GCOD and Coastal Flood
124. key_factors computed from REAL historical data (was hardcoded "~18%", "68%")
125. SHAP labels human-readable: "Recent approval trend in this ward" not "Year Ward Rate"
126. Similar cases: stratified sampling includes 1-2 denied cases for contrast
127. variance_history integrated into /analyze_proposal response (real rates for exact combo)
128. Actionable recommendations in /analyze_proposal — "hire attorney", "reduce units", "add parking", etc.
129. Timeline from real tracker data — median 142 days, by phase (filing→hearing→decision), by ward
130. Frontend: historical analysis section, subdistrict display, color-coded similar cases, SHAP labels
131. Tested and rejected: agenda features in model (added noise, AUC dropped 0.006 — rolled back)

### Known Issues / TODO
- ~~Frontend `API_URL` is hardcoded to localhost~~ → FIXED
- ~~No structured logging anywhere~~ → FIXED
- ~~No caching for address searches~~ → FIXED
- ~~Model doesn't save diagnostic plots~~ → FIXED
- ~~`analyzer.py` and `model_loader.py` orphaned~~ → FIXED
- ~~Target encoding data leakage~~ → FIXED
- ~~No rate limiting on API~~ → FIXED (configurable, 120/min, exempt localhost)
- ~~14 post-hearing features leaking~~ → FIXED (removed from training + API)
- ~~Calibration on test set~~ → FIXED (separate calibration holdout)
- ~~has_attorney catching only 1%~~ → FIXED (broader regex, ~24%)
- ~~No model versioning~~ → FIXED (model_history/ with JSONL log)
- ~~No API authentication~~ → FIXED (optional API key)
- ~~Ward heatmap 22 API calls~~ → FIXED (/wards/all single endpoint)
- ~~No site selection feature~~ → FIXED (/recommend + frontend panel)
- ~~API monolith too large~~ → FIXED (market_intel router, 794 lines extracted)
- ~~No PostGIS~~ → FIXED (88K parcels, spatial index, API uses DB first)
- ~~No version control~~ → FIXED (GitHub private repo)
- ~~AUC will drop after retrain with clean features~~ → Settled at honest test AUC 0.7728, honest CV 0.7686
- ~~CV AUC inflated by target encoding leakage~~ → FIXED (honest CV recomputes TE within folds)
- ~~Model ceiling ~0.77 AUC~~ → Pushed to **0.7987** with enriched denial data
- ~~Zoning districts too coarse (3A-3C)~~ → FIXED (286 granular subdistricts from BPDA)
- ~~Q1/Q2 inconsistency~~ → FIXED (_get_parcel_zoning single source of truth)
- ~~Hardcoded key_factors~~ → FIXED (computed from real historical data)
- ~~Cryptic SHAP labels~~ → FIXED (human-readable FEATURE_LABELS dict)
- ~~Similar cases all APPROVED~~ → FIXED (stratified sampling includes denied)
- ~~Timeline hardcoded 120 days~~ → FIXED (real tracker data, by phase, by ward)
- ~~No overlay district awareness~~ → FIXED (GCOD + Coastal Flood flagged)
- ~~No actionable recommendations~~ → FIXED (attorney, units, parking, project type advice)
- 490 denied cases still have no variance data (no tracker description, no OCR text)
- `proj_*` columns not in current CSV (will be added after OCR retrain)
- Need manual OCR quality audit (sample 50 cases vs original PDFs)
- Customer segment and pricing model undefined (business decision)
- Boston-only TAM too small for venture scale (expansion strategy needed)
- PostgreSQL 18 runs on port 5432 (installed at /Library/PostgreSQL/18), password: permitiq123

## How to Run Everything

### Quick start (Makefile):
```bash
cd ~/Desktop/Boston\ Zoning\ Project
make run          # Starts API + Frontend
make test         # Run 75 tests (requires API running)
make retrain      # Retrain model from existing dataset
make retrain-clean # Clean OCR → audit → retrain (after OCR completes)
make audit        # Run OCR quality audit
make status       # Check OCR progress + data freshness
make docker       # Start with Docker Compose
make push         # Commit and push to GitHub
# Or after OCR: bash post_ocr_retrain.sh
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
