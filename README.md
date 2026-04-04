<p align="center">
  <strong style="font-size: 36px;">PermitIQ</strong>
</p>

<h3 align="center">Know if you'll win before you file.</h3>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.9-blue?style=flat-square" alt="Python 3.9"/>
  <img src="https://img.shields.io/badge/FastAPI-0.100+-009688?style=flat-square&logo=fastapi" alt="FastAPI"/>
  <img src="https://img.shields.io/badge/ML-Stacking%20Ensemble-orange?style=flat-square" alt="ML: Stacking Ensemble"/>
  <img src="https://img.shields.io/badge/tests-40%20passing-brightgreen?style=flat-square" alt="Tests: 40 passing"/>
  <img src="https://img.shields.io/badge/AUC-0.7998-purple?style=flat-square" alt="AUC: 0.7998"/>
</p>

---

## The Problem

Boston developers spend **$30,000–$100,000** on zoning attorneys, architects, and filing fees before the Zoning Board of Appeals (ZBA) votes on their project. They have no data-driven way to assess their odds. The process is opaque, political, and expensive to get wrong.

## The Solution

**PermitIQ** predicts ZBA approval probability using machine learning trained on **13,300+ real decisions from 2020–2026**. Competitors tell you the zoning rules. PermitIQ tells you **if you'll win**.

Enter an address. Get a probability. Understand why.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  Frontend (Streamlit)          Port 8501                    │
│  ┌─────────┐ ┌──────────┐ ┌───────────┐ ┌──────────────┐   │
│  │ Address  │ │  Zoning  │ │Compliance │ │ ML Prediction│   │
│  │ Search   │ │ Details  │ │  Checker  │ │  + SHAP      │   │
│  └────┬─────┘ └────┬─────┘ └─────┬─────┘ └──────┬───────┘   │
│       └─────────────┴─────────────┴──────────────┘           │
│                           │ REST API                         │
├───────────────────────────┼─────────────────────────────────┤
│  API (FastAPI)             Port 8000       34 endpoints      │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────────┐   │
│  │ Search   │ │ Zoning   │ │Prediction│ │Market Intel  │   │
│  │ Parcels  │ │ Comply   │ │ Compare  │ │ Attorneys    │   │
│  └────┬─────┘ └────┬─────┘ └────┬─────┘ └──────┬───────┘   │
│       └─────────────┴────────────┴──────────────┘           │
│                           │                                  │
│  ┌────────────────────────┼─────────────────────────────┐   │
│  │           Data Layer                                  │   │
│  │  13,300 ZBA cases │ 98K parcels │ 85-feature model   │   │
│  │  184K properties  │ 718K permits │ PostGIS spatial    │   │
│  └───────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

## Key Features

| Feature | Description |
|---------|-------------|
| **Approval Probability** | ML risk score with confidence intervals and calibration warnings |
| **SHAP Explainability** | Top 8 prediction drivers with direction and magnitude for every prediction |
| **Auto-Detect Violations** | Parcel-level zoning issues identified from public records before proposal |
| **What-If Scenarios** | Toggle attorney, reduce units, remove variances — see probability deltas |
| **Compliance Checker** | Checks proposals against 286 zoning subdistricts (FAR, height, setbacks, parking) |
| **Attorney Intelligence** | Win rates, ward specialties, and case history for every ZBA attorney |
| **Site Selection** | ML-ranked parcels for your project type with interactive map |
| **Market Intelligence** | 12 endpoints: trends, denial patterns, voting analysis, ward stats |
| **Downloadable Reports** | Professional HTML reports with SHAP analysis, compliance, and timeline |

## Tech Stack

| Layer | Technology |
|-------|-----------|
| **ML Model** | Stacking Ensemble: XGBoost + Gradient Boosting + Random Forest, Logistic Regression meta-learner, Platt-calibrated |
| **Features** | 85 pre-hearing features across 14 categories — no data leakage |
| **Backend** | FastAPI, 34 RESTful endpoints, structured logging, API key auth |
| **Frontend** | Streamlit with custom dark theme, PyDeck maps, interactive charts |
| **Data Pipeline** | OCR (PyMuPDF + Tesseract) → regex extraction → fuzzy matching → feature engineering |
| **Spatial** | PostGIS / in-memory GeoJSON, 98,510 parcel centroids, overlay districts |
| **Infrastructure** | Docker Compose, GitHub Actions CI, Railway + Streamlit Cloud |

## ML Model Performance

| Metric | Value | Context |
|--------|-------|---------|
| **Test AUC** | 0.7998 | Strong discrimination between approvals and denials |
| **Honest CV AUC** | 0.7921 | 5-fold cross-validation, no leakage |
| **Denial Recall** | 69.7% | Catches 7 in 10 denials in a 90% approval-rate dataset |
| **Brier Score** | 0.09 | Well-calibrated probability estimates |
| **ECE** | 1.0% | Predicted probabilities closely match actual outcomes |
| **Training Data** | 13,356 cases | Deduplicated, cleaned from 262 ZBA decision PDFs |

**Anti-leakage discipline:** 14 post-hearing features identified and removed. Temporal train/test split. Target encoding recomputed within each CV fold. All features are knowable *before* the ZBA hearing.

## API Endpoints (34)

| Group | Endpoints | Highlights |
|-------|-----------|------------|
| **Search** | 3 | Fuzzy address search, case history, autocomplete across 175K properties |
| **Parcels** | 3 | Parcel zoning details, nearby ZBA cases within 0.5mi, geocoding |
| **Zoning** | 4 | Subdistrict requirements, compliance check, full analysis, variance analysis |
| **Prediction** | 3 | ML prediction with SHAP, batch predict (up to 50), what-if scenarios |
| **Market Intel** | 12 | Trends, variance rates, denial patterns, voting, ward stats, timeline |
| **Attorneys** | 4 | Search, profiles with win rates, similar cases, leaderboard |
| **Platform** | 4 | Health, stats, model info, data status |
| **Recommendation** | 1 | ML-ranked site selection with map coordinates |

```bash
# Example: predict approval probability
curl -X POST http://localhost:8000/analyze_proposal \
  -H "Content-Type: application/json" \
  -d '{
    "parcel_id": "0302951010",
    "proposed_use": "residential",
    "project_type": "new_construction",
    "variances": ["height", "far", "parking"],
    "has_attorney": true,
    "proposed_units": 6,
    "proposed_stories": 4
  }'
```

## Data Pipeline

```
262 ZBA Decision PDFs
  → OCR (PyMuPDF + Tesseract fallback)
  → Regex case parsing (case numbers, addresses, decisions, votes, variances)
  → Fuzzy matching to 184K property assessment records
  → External data integration (building permits, zoning districts, overlays)
  → Feature engineering (85 features across 14 categories)
  → Stacking ensemble training with 5-fold out-of-fold predictions
  → Platt calibration on held-out set
  → Deployed model package (.pkl)
```

## Project Structure

```
api/
  main.py                  FastAPI app, middleware, CORS, router registration
  routes/
    search.py              Address search, case history, autocomplete
    parcels.py             Parcel lookup, nearby cases, geocoding
    zoning.py              Zoning analysis, compliance checker
    prediction.py          ML prediction, SHAP, what-if scenarios
    market_intel.py        12 market intelligence endpoints
    attorneys.py           Attorney search, profiles, leaderboard
    recommend.py           Site selection engine
    platform.py            Health, stats, model metadata
  services/
    data_loader.py         Startup data loading and index building
    feature_builder.py     85 feature definitions (shared with training)
    zoning_code.py         Zoning requirements by subdistrict
    recommendations.py     Counterfactual recommendation engine
    database.py            PostGIS spatial queries
    auth.py                API key authentication

frontend/
  app.py                   Streamlit UI — 4-step flow with dark theme

zba_pipeline/
  extract_text.py          OCR pipeline (PyMuPDF + Tesseract)
  parse_cases.py           Regex extraction of case fields
  build_dataset.py         Dataset assembly and deduplication

train_model_v2.py          Model training with auto-comparison to previous versions
landing/index.html         Marketing landing page
tests/                     Unit + integration + product tests
```

## Getting Started

```bash
git clone https://github.com/steviejoe23/PermitIQ.git
cd PermitIQ
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
make run    # Starts API (port 8000) + Frontend (port 8501)
```

> **Note:** The ML model and data files are not included in the repository due to size. Contact me for access to the full dataset and trained model.

## License

Copyright 2026 Steven Spero. All rights reserved.

## Contact

Steven Spero — [hello@permitiq.com](mailto:hello@permitiq.com)
