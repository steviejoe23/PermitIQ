# PermitIQ Deployment Guide

## Docker Compose (Recommended)

The project includes a `docker-compose.yml` with three services: PostGIS database, FastAPI backend, and Streamlit frontend.

### Quick Start

```bash
cd ~/Desktop/Boston\ Zoning\ Project

# Start all services (builds images on first run)
make docker
# or: docker-compose up --build -d

# Check status
docker-compose ps
docker-compose logs -f api

# Stop
docker-compose down

# Stop and remove data volume
docker-compose down -v
```

### Data Files

Large data files are mounted as read-only volumes (not baked into the image). Ensure these exist before starting:

- `zba_cases_cleaned.csv` (30 MB) -- ZBA case dataset
- `boston_parcels_zoning.geojson` (147 MB) -- Parcel geometries
- `property_assessment_fy2026.csv` (80 MB) -- Property assessments
- `api/zba_model.pkl` (2 MB) -- Trained model

### Service Ports

| Service  | Container Port | Host Port |
|----------|---------------|-----------|
| API      | 8000          | 8000      |
| Frontend | 8501          | 8501      |
| PostGIS  | 5432          | 5433      |

PostGIS is exposed on 5433 to avoid conflicts with a local PostgreSQL installation.

---

## Environment Variables

All environment variables with their defaults:

### API (`api/main.py`)

| Variable | Default | Description |
|----------|---------|-------------|
| `PERMITIQ_API_KEY` | *(none)* | API key for prediction endpoints. Leave unset to disable auth. |
| `RATE_LIMIT_PER_MINUTE` | `120` | Max requests per IP per minute. Localhost is exempt. |
| `DATABASE_URL` | `postgresql://postgres:permitiq123@localhost:5432/permitiq` | PostGIS connection string. API falls back to GeoJSON if unavailable. |

### Frontend (`frontend/app.py`)

| Variable | Default | Description |
|----------|---------|-------------|
| `PERMITIQ_API_URL` | `http://127.0.0.1:8000` | Base URL for API calls from the Streamlit frontend. |

### Database (Docker Compose)

| Variable | Default | Description |
|----------|---------|-------------|
| `POSTGRES_PASSWORD` | `permitiq123` | PostGIS password (used in `docker-compose.yml`). |

---

## Streamlit Cloud Deployment

Streamlit Cloud can host the frontend for free. The API must be deployed separately (see Railway/Render below).

### Steps

1. Push the repo to GitHub (already at `github.com/steviejoe23/PermitIQ`).

2. Go to [share.streamlit.io](https://share.streamlit.io) and connect your GitHub account.

3. Create a new app:
   - **Repository:** `steviejoe23/PermitIQ`
   - **Branch:** `main`
   - **Main file path:** `frontend/app.py`

4. Add a secret in Streamlit Cloud settings (Settings > Secrets):
   ```toml
   PERMITIQ_API_URL = "https://your-api-host.com"
   ```

5. Add a `requirements.txt` in the `frontend/` directory (or use the root one). Streamlit Cloud installs dependencies automatically.

6. Deploy. The frontend will be available at `https://your-app.streamlit.app`.

### Notes

- Streamlit Cloud only runs the frontend. The API and database must be hosted elsewhere.
- The free tier has resource limits. For production, use Streamlit Teams or self-host.

---

## Railway Deployment

[Railway](https://railway.app) can host the API, database, and frontend together.

### Steps

1. Create a new project on Railway and connect your GitHub repo.

2. Add a **PostgreSQL** service (Railway has a one-click PostGIS add-on).

3. Add a **Web Service** for the API:
   - **Start command:** `cd api && uvicorn main:app --host 0.0.0.0 --port $PORT`
   - Set environment variables: `PERMITIQ_API_KEY`, `DATABASE_URL` (auto-provided by Railway PostgreSQL), `RATE_LIMIT_PER_MINUTE`.

4. Add a second **Web Service** for the frontend:
   - **Start command:** `streamlit run frontend/app.py --server.port $PORT --server.address 0.0.0.0`
   - Set `PERMITIQ_API_URL` to the API service's internal URL.

5. Upload data files via Railway volumes or store them in the repo with Git LFS.

---

## Render Deployment

[Render](https://render.com) offers free-tier web services and managed PostgreSQL.

### Steps

1. Create a **Web Service** for the API:
   - **Build command:** `pip install -r requirements.txt`
   - **Start command:** `cd api && uvicorn main:app --host 0.0.0.0 --port $PORT`
   - Set environment variables as listed above.

2. Create a **Web Service** for the frontend:
   - **Start command:** `streamlit run frontend/app.py --server.port $PORT --server.address 0.0.0.0`
   - Set `PERMITIQ_API_URL` to the API service URL.

3. Create a **PostgreSQL** database (Render offers managed PostGIS). Use the provided `DATABASE_URL`.

4. Data files: use Render Disks (persistent storage) or download at build time.

---

## Production Checklist

Before going live, verify each item:

### Security

- [ ] `PERMITIQ_API_KEY` is set to a strong, random value
- [ ] CORS origins restricted (update `allow_origins` in `api/main.py` from `["*"]` to your domain)
- [ ] PostgreSQL password changed from default (`permitiq123`)
- [ ] No `.env` files or secrets committed to the repo
- [ ] API key is not logged or returned in responses

### Performance

- [ ] Rate limiting configured (`RATE_LIMIT_PER_MINUTE` set appropriately)
- [ ] Uvicorn running with `--workers 2` or more (default in Docker Compose)
- [ ] PostGIS spatial indexes verified (`CREATE INDEX` on geometry columns)
- [ ] Data files mounted as read-only volumes (not copied into container)

### Data

- [ ] `zba_cases_cleaned.csv` is up to date (run `make retrain-clean` after OCR)
- [ ] `api/zba_model.pkl` matches the latest training run
- [ ] `boston_parcels_zoning.geojson` includes BPDA subdistricts
- [ ] `property_assessment_fy2026.csv` is current fiscal year

### Monitoring

- [ ] `/health` endpoint returning `200` with model metadata
- [ ] Structured logging enabled (timestamps, request timing in stdout)
- [ ] Docker healthchecks active (API checked every 30s)
- [ ] Error alerting configured (e.g., Sentry, log aggregator)

### Frontend

- [ ] `PERMITIQ_API_URL` points to production API (not localhost)
- [ ] Disclaimer text visible on all prediction outputs
- [ ] XSS escaping active on all API-sourced data
