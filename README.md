# PermitIQ

**Boston ZBA approval prediction platform — ML trained on 7,500+ real zoning board decisions**

PermitIQ is an end-to-end machine learning platform that predicts whether the Boston Zoning Board of Appeal (ZBA) will approve or deny a zoning variance request. It combines OCR-processed historical decisions, geospatial analysis, and a stacking ensemble model to give property developers, attorneys, and city planners actionable intelligence before they file.

---

## Features

- **Approval Prediction** — Stacking ensemble model trained on 7,500+ real ZBA decisions with calibrated probability scores
- **Zoning Compliance Checker** — Automated analysis of whether a property and proposed use comply with Boston's zoning code
- **Variance Analysis** — Identifies key factors driving approval or denial for a given address and use case
- **Market Intelligence** — Real estate and zoning market context for site selection decisions
- **Geospatial Analysis** — PostGIS-powered spatial queries for parcel-level zoning lookups
- **OCR Pipeline** — Automated extraction and processing of ZBA decision documents
- **Interactive Frontend** — Streamlit-based UI with a unified flow: Address → Zoning → Compliance → Prediction

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| **API** | Python, FastAPI, Pydantic |
| **ML** | scikit-learn (Stacking Ensemble), joblib, NumPy, pandas |
| **Database** | PostgreSQL + PostGIS |
| **Geospatial** | GeoPandas, PostGIS |
| **Frontend** | Streamlit |
| **Infrastructure** | Docker, Docker Compose |
| **OCR** | Custom pipeline for ZBA document extraction |

---

## Project Structure

```
PermitIQ/
├── api/
│   ├── main.py              # FastAPI application (3,000+ lines)
│   ├── routes/
│   │   └── market_intel.py   # Market intelligence endpoints
│   ├── services/
│   │   ├── database.py       # PostGIS database layer
│   │   ├── feature_builder.py # ML feature engineering
│   │   ├── model_classes.py  # Custom model definitions
│   │   └── zoning_code.py    # Boston zoning code logic
│   ├── models.py             # SQLAlchemy/data models
│   ├── schema.py             # Pydantic request/response schemas
│   └── config.py             # Application configuration
├── frontend/
│   └── app.py                # Streamlit interactive UI
├── tests/                    # Test suite
├── zba_pipeline/             # ZBA decision processing pipeline
├── train_model.py            # Model training script
├── train_model_v2.py         # Updated training with compliance workflow
├── auto_scrape_decisions.py  # Automated ZBA decision scraping
├── rebuild_dataset.py        # Dataset rebuild utility
├── docker-compose.yml        # Multi-service Docker config
├── Dockerfile                # Container definition
├── Makefile                  # Build and run commands
└── requirements.txt          # Python dependencies
```

---

## Getting Started

### Prerequisites

- Python 3.10+
- PostgreSQL with PostGIS extension
- Docker & Docker Compose (recommended)

### Quick Start with Docker

```bash
# Clone the repository
git clone https://github.com/steviejoe23/PermitIQ.git
cd PermitIQ

# Copy environment template and configure
cp .env.example .env

# Build and run with Docker Compose
docker-compose up --build
```

### Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run the API
uvicorn api.main:app --reload --port 8000

# Run the frontend (separate terminal)
streamlit run frontend/app.py
```

---

## API Endpoints

The FastAPI backend exposes a comprehensive REST API. Key endpoints include:

- `POST /predict` — Submit an address and proposed use for approval prediction
- `POST /check-compliance` — Verify zoning compliance for a property
- `GET /variance-analysis` — Analyze key variance factors for a given case
- `GET /market-intel` — Retrieve market intelligence for site selection
- `GET /docs` — Interactive Swagger API documentation

---

## How It Works

1. **Data Collection** — ZBA decision documents are scraped and processed via OCR
2. **Feature Engineering** — Address, zoning district, use type, variance history, and geospatial features are extracted
3. **Model Training** — A stacking ensemble combines multiple classifiers with calibrated probability outputs
4. **Prediction** — New applications are scored against the trained model with explainable factor breakdowns
5. **Compliance Check** — Proposed uses are validated against Boston's zoning code in real time

---

## Author

**Steven Spero**
- GitHub: [@steviejoe23](https://github.com/steviejoe23)
- Email: stevenspero23@gmail.com

---

## License

This project is proprietary. All rights reserved.
