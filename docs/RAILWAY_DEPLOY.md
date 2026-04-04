# Railway Deployment Guide for PermitIQ API

## Prerequisites

- Railway account: https://railway.app
- Railway CLI installed
- Git repository (already at https://github.com/steviejoe23/PermitIQ)

## Step 1: Install Railway CLI

```bash
npm i -g @railway/cli
```

Or with Homebrew:

```bash
brew install railway
```

## Step 2: Login

```bash
railway login
```

This opens a browser for authentication.

## Step 3: Initialize Project

From the project root:

```bash
cd ~/Desktop/Boston\ Zoning\ Project
railway init
```

Select "Create new project" when prompted. This links the local directory to a Railway project.

## Step 4: Set Environment Variables

```bash
railway variables set PERMITIQ_API_KEYS="your-free-key:free,your-pro-key:pro,your-enterprise-key:enterprise"
railway variables set PERMITIQ_CORS_ORIGINS="https://permitiq.streamlit.app,https://permitiq.com"
railway variables set RATE_LIMIT_PER_MINUTE=120
railway variables set PYTHONUNBUFFERED=1
```

Optional (if using PostGIS on Railway):

```bash
railway variables set DATABASE_URL="postgresql://user:pass@host:5432/permitiq"
```

You can also set these in the Railway dashboard under your service's Variables tab.

## Step 5: Deploy

```bash
railway up
```

This builds the `Dockerfile.production` image and deploys it. First deploy takes 5-10 minutes due to the large data files (~500MB image).

## Step 6: Get Your Public URL

```bash
railway domain
```

This generates a public `*.up.railway.app` URL. You can also add a custom domain in the Railway dashboard.

## Step 7: Verify

```bash
# Health check
curl https://your-app.up.railway.app/health

# API docs
open https://your-app.up.railway.app/docs
```

## Step 8: Point Frontend to Railway API

Set the environment variable for your Streamlit frontend (locally or on Streamlit Cloud):

```bash
export PERMITIQ_API_URL=https://your-app.up.railway.app
```

On Streamlit Cloud, add this in the app's Secrets or Environment Variables settings.

---

## Data Files Strategy

The production Dockerfile (`Dockerfile.production`) copies data files directly into the image:

| File | Size | Purpose |
|------|------|---------|
| zba_cases_cleaned.csv | ~30MB | ZBA case dataset |
| boston_parcels_zoning.geojson | ~147MB | Parcel geometries + zoning |
| property_assessment_fy2026.csv | ~80MB | Property tax assessments |
| api/zba_model.pkl | ~2MB | Trained ML model |

This produces an image around 500MB. This is the simplest approach for MVP.

### Alternative approaches for production scale

**(b) Railway Volumes:**
Mount a persistent volume and upload data files once. Avoids rebuilding the image when data changes, but requires manual file management.

**(c) Cloud Storage (S3/GCS):**
Store data files in S3 or GCS and fetch them at startup. Add download logic to `api/services/data_loader.py`. Best for frequently updated data, but adds startup latency and requires cloud storage credentials.

---

## Updating the Deployment

### Code changes only (no data changes)

```bash
railway up
```

### After retraining the model

```bash
# Retrain locally
python3 train_model_v2.py

# Redeploy (rebuilds image with new model)
railway up
```

### After updating data files

```bash
# Update CSVs locally (e.g., auto_update_data.py)
python3 auto_update_data.py

# Redeploy
railway up
```

---

## Configuration Files

| File | Purpose |
|------|---------|
| `Procfile` | Fallback start command (if not using Dockerfile) |
| `railway.toml` | Railway build and deploy configuration |
| `railway.json` | Railway service schema configuration |
| `Dockerfile.production` | Production Docker image with data files baked in |

---

## Monitoring

- Railway dashboard shows logs, metrics, and deployment history
- Health endpoint: `GET /health` returns model status, dataset counts, AUC
- API docs: `/docs` (Swagger) and `/redoc` (ReDoc)

## Troubleshooting

**Build fails with out-of-memory:**
Railway free tier has limited memory. The GeoJSON file (147MB) requires significant RAM to load. Upgrade to a paid plan or reduce the GeoJSON file size.

**Startup timeout (healthcheck fails):**
The `healthcheckTimeout` is set to 300 seconds (5 minutes) in `railway.toml` to account for loading large data files at startup. If it still times out, increase this value.

**Port issues:**
Railway assigns a dynamic `$PORT`. The `railway.toml` start command uses `$PORT` automatically. Never hardcode port 8000 in Railway configuration.
