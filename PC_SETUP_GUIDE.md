# PermitIQ — PC Setup Guide (Complete Transfer from Mac)

## Step 1: Clone the Repo

```bash
git clone https://github.com/steviejoe23/PermitIQ.git "Boston Zoning Project"
cd "Boston Zoning Project"
git checkout transfer
```

This gives you ALL source code, scripts, notebooks, docs, memory files, leads, and data files under 100MB.

## Step 2: Download Large Files from Google Drive

Go to: https://drive.google.com → `PermitIQ-Complete-Transfer` folder

Download these and place in the project root:

| File | Size | Where to put it |
|------|------|-----------------|
| `building_permits.csv` | 253MB | project root |
| `zba_model_v2.pkl` | 219MB | project root |
| `boston_parcels_zoning.geojson` | 81MB | project root |
| `property_assessment_fy2026.csv` | 76MB | project root |
| `api/zba_model.pkl` | 219MB | `api/` subfolder (copy of zba_model_v2.pkl) |
| `pdfs/` | 4.2GB | project root (274 PDF files) |
| `model_history/` | 1.7GB | project root (50 model versions) |
| `parcels_2025/` | 91MB | project root |
| `parcels_2025_clean/` | 118MB | project root |

**Shortcut for api/zba_model.pkl:** If you have zba_model_v2.pkl, just copy it:
```bash
cp zba_model_v2.pkl api/zba_model.pkl
```

## Step 3: Set Up Python Environment

```bash
python -m venv zoning-env
# Windows:
zoning-env\Scripts\activate
# OR Linux/Mac:
source zoning-env/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

### Additional dependencies for specific tasks:
```bash
# For ML model training:
pip install xgboost scikit-learn shap

# For OCR pipeline (if reprocessing PDFs):
pip install pymupdf pytesseract beautifulsoup4

# For frontend:
pip install -r frontend/requirements.txt
```

## Step 4: Run the App

### Start the API (Terminal 1):
```bash
cd api
uvicorn main:app --reload --port 8000
```

Wait for: "Application startup complete" and "Loaded X cases" messages.

### Start the Frontend (Terminal 2):
```bash
cd frontend
set PERMITIQ_API_URL=http://127.0.0.1:8000
streamlit run app.py --server.port 8501
```

Open http://localhost:8501 in your browser.

## Step 5: Verify Everything Works

### Demo Flow Test:
1. **Search:** Type "105 Norwell St" → should return results with approval rates
2. **Parcel:** Click a result → zoning details + map should render
3. **Compliance:** Enter a proposal (4 units, 3 stories, residential) → violations with rates
4. **Prediction:** Select variances → ML probability, SHAP factors, similar cases

### Quick API Test:
```bash
curl http://127.0.0.1:8000/health
curl "http://127.0.0.1:8000/search?q=105+Norwell+St"
```

## Step 6: Set Up Claude Context

Copy `CLAUDE.md` to your Claude project settings so future Claude sessions have full context.

The `memory/` directory contains additional context files:
- `memory/glossary.md` — Domain terminology
- `memory/context/company.md` — Business context
- `memory/people/michael-winston.md` — Investor contact
- `memory/projects/permitiq.md` — Project status

## File Structure Summary

```
Boston Zoning Project/
├── api/                    # FastAPI backend (8 route modules, 34 endpoints)
│   ├── main.py             # App entry point
│   ├── routes/             # All API route handlers
│   ├── services/           # Business logic (feature builder, recommendations, etc.)
│   ├── zba_model.pkl       # The ML model (219MB) ← FROM GOOGLE DRIVE
│   └── ...
├── frontend/
│   ├── app.py              # Streamlit UI (~3000 lines)
│   └── requirements.txt
├── tests/                  # 41 unit tests + integration tests
├── zba_pipeline/           # OCR extraction pipeline
├── pdfs/                   # 274 ZBA decision PDFs ← FROM GOOGLE DRIVE
├── model_history/          # 50 previous model versions ← FROM GOOGLE DRIVE
├── memory/                 # Claude context files
├── leads/                  # Sales leads (attorneys, developers)
├── docs/                   # Deployment guides
├── zba_cases_cleaned.csv   # Main dataset (7,500+ cases)
├── building_permits.csv    # 718K building permits ← FROM GOOGLE DRIVE
├── property_assessment_fy2026.csv # Property data ← FROM GOOGLE DRIVE
├── boston_parcels_zoning.geojson   # 98K parcels ← FROM GOOGLE DRIVE
├── CLAUDE.md               # Full project context for AI
├── train_model_v2.py       # Model training script
├── upload_to_drive.py      # Transfer utility (Mac only)
└── ...
```

## Known Issues & Workarounds

1. **Tesseract not installed:** Only needed for OCR reprocessing. Install via:
   - Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki
   - Mac: `brew install tesseract`

2. **PostgreSQL/PostGIS not available:** API falls back to in-memory GeoJSON automatically. No action needed unless you want spatial queries.

3. **Model too large for Railway:** Railway's free tier (1GB) can't hold the 219MB model + data. Use local or upgrade Railway plan.

4. **Python 3.9 warnings:** Works fine, just shows deprecation warnings. Use Python 3.10+ to suppress.

## Deployment (when ready)

### Railway (API):
```bash
git push origin main  # Auto-deploys via railway.toml
```

### Streamlit Cloud (Frontend):
Auto-deploys from `main` branch. Config: `frontend/app.py` as entry point.

## Current Status (April 4, 2026)

- ✅ 41 unit tests passing, CI green
- ✅ Full demo flow working locally
- ✅ Smart ML recommendations with probability impacts
- ✅ Timeline phase breakdown (filing→hearing→decision)
- ✅ Variance auto-detection and auto-fill
- ⚠️ Railway deployment degraded (light mode, no ML)
- ⚠️ Need to retrain lightweight model OR upgrade Railway plan
