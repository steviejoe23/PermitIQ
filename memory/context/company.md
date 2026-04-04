# PermitIQ — Project Context

## What It Is
PermitIQ predicts whether the Boston Zoning Board of Appeals (ZBA) will approve a development project before the developer files. Developers spend $30-100K on permitting with no idea if they'll get approved. PermitIQ quantifies that risk using ML trained on every real ZBA decision from 2020-2026.

## Tech Stack
- **Backend:** FastAPI (Python 3.9), 33 endpoints across 8 route modules
- **Frontend:** Streamlit (port 8501)
- **ML:** Stacking ensemble (XGBoost + GB + RF), AUC 0.7987, 48 features
- **Data:** 7,500+ ZBA cases, 98K parcels, 175K property records
- **Database:** PostGIS (PostgreSQL 18, port 5432, password in env var)
- **Deployment:** Docker Compose, GitHub private repo
- **OCR:** PyMuPDF + Tesseract for 275 ZBA decision PDFs

## Key Directories
- `api/` — FastAPI backend (main.py + routes/ + services/)
- `frontend/` — Streamlit UI
- `zba_pipeline/` — OCR and data processing
- `tests/` — 96+ integration tests

## Competitors
- **UrbanForm, Zoneomics** — tell you the zoning rules
- **PermitIQ** — tells you if you'll WIN (prediction, not just lookup)

## Development Environment
- macOS, Python 3.9 virtual env at `zoning-env/`
- Tesseract at `/opt/homebrew/bin/tesseract`
- Run with `make run` or manual uvicorn + streamlit
