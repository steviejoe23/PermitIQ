<p align="center">
  <img src="docs/screenshots/logo.png" alt="PermitIQ" width="280"/>
</p>

<h3 align="center">Know if you'll win before you file.</h3>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.9-blue?style=flat-square" alt="Python 3.9"/>
  <img src="https://img.shields.io/badge/FastAPI-0.100+-009688?style=flat-square&logo=fastapi" alt="FastAPI"/>
  <img src="https://img.shields.io/badge/license-proprietary-red?style=flat-square" alt="License: Proprietary"/>
  <img src="https://img.shields.io/badge/tests-119%20passing-brightgreen?style=flat-square" alt="Tests: 119 passing"/>
</p>

---

## The Problem

Boston developers spend **$30,000 -- $100,000** on zoning attorneys, architects, and filing fees before the Zoning Board of Appeals (ZBA) votes on their project. They have no data-driven way to know if they'll be approved. The process is opaque, political, and expensive to get wrong.

## The Solution

**PermitIQ** is the first platform that predicts ZBA approval probability using machine learning trained on **every real decision from 2020--2026**. Competitors like UrbanForm and Zoneomics tell you the zoning rules. PermitIQ tells you **if you'll win**.

Enter an address. Get a probability. Understand why.

## Demo

<p align="center">
  <img src="docs/screenshots/prediction_panel.png" alt="Prediction Panel" width="700"/>
</p>
<p align="center">
  <img src="docs/screenshots/compliance_check.png" alt="Compliance Check" width="700"/>
</p>
<p align="center">
  <img src="docs/screenshots/market_intelligence.png" alt="Market Intelligence" width="700"/>
</p>

## Key Features

- **Approval Probability** -- ML-powered risk score with confidence intervals for any proposed project
- **Auto-Detect Violations** -- Parcel-level zoning issues identified from public records before you even submit a proposal
- **What-If Scenarios** -- Change the attorney, reduce units, add parking -- see how each move shifts your odds
- **Attorney Intelligence** -- Win rates, ward specialties, and case history for every ZBA attorney on record
- **Compliance Checker** -- Checks your proposal against 286 zoning subdistricts with FAR, height, setback, and parking rules
- **Site Selection** -- Find the best parcels in Boston for your project type, ranked by predicted approval
- **Downloadable Reports** -- Professional HTML risk assessments for clients and investors
- **98,510 Parcels Mapped** -- Full parcel-level zoning data with overlay districts (GCOD, Coastal Flood)

## How It Works

```
1. SEARCH     Enter any Boston address or parcel ID
              PermitIQ returns zoning district, auto-detected violations, and case history.

2. ANALYZE    Describe your project (use type, units, stories, variances)
              The model returns approval probability, confidence range, and top risk factors.

3. OPTIMIZE   Run what-if scenarios to find the strongest filing strategy
              Compare attorneys, project scopes, and variance combinations.
```

## Tech Stack

| Layer | Technology |
|-------|-----------|
| **ML Model** | Stacking Ensemble (XGBoost + Gradient Boosting + Random Forest), Platt-calibrated |
| **Backend** | FastAPI, 33 endpoints, structured logging, rate limiting, API key auth |
| **Frontend** | Streamlit with custom dark theme, PyDeck maps, interactive charts |
| **Data** | 7,500+ ZBA decisions, 98K parcels (GeoJSON), 184K property assessments, 718K building permits |
| **Infrastructure** | Docker Compose, PostGIS, automated OCR pipeline (Tesseract), nightly retraining |

## API

33 RESTful endpoints organized across Search, Parcels, Prediction, Zoning, Market Intelligence, Attorneys, and Platform.

Interactive documentation available at `/docs` (Swagger) and `/redoc` when the server is running.

```bash
# Example: predict approval for a proposal
curl -X POST http://localhost:8000/analyze_proposal \
  -H "Content-Type: application/json" \
  -d '{"parcel_id": "0302951010", "use_type": "residential", "project_type": "addition", "variances": ["height", "far"]}'
```

## Model Performance

| Metric | Value |
|--------|-------|
| **Test AUC** | 0.7987 |
| **Honest CV AUC** | 0.7910 |
| **Denial Recall** | 69.7% (catches 7 in 10 denials) |
| **Brier Score** | 0.09 (well-calibrated) |
| **ECE** | 1.0% (predicted probabilities match reality) |
| **Features** | 48 pre-hearing features across 14 categories |
| **Training Data** | 13,308 deduplicated ZBA cases |

No data leakage. Temporal train/test split. Target encoding recomputed within each CV fold.

## Getting Started

```bash
git clone https://github.com/steviejoe23/PermitIQ.git
cd PermitIQ
make run
```

API at `http://localhost:8000/docs` | Frontend at `http://localhost:8501`

## License

Copyright 2026 PermitIQ. All rights reserved. This software is proprietary and confidential. Unauthorized copying, distribution, or use is strictly prohibited.

## Contact

hello@permitiq.com
