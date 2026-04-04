# Deploying PermitIQ Frontend on Streamlit Cloud

## Prerequisites

- GitHub repository: https://github.com/steviejoe23/PermitIQ (private)
- FastAPI backend deployed separately (Railway, Render, or similar)
- Backend URL available (e.g., https://api.permitiq.com)

## Architecture Note

Streamlit Cloud only hosts Streamlit applications. The FastAPI backend must be
deployed independently on a platform that supports Python ASGI servers (Railway,
Render, Fly.io, AWS, etc.). The frontend connects to the backend via the
`PERMITIQ_API_URL` environment variable.

## Step-by-Step Deployment

### 1. Deploy the API Backend First

Deploy the FastAPI backend to your chosen platform. The API lives in the `api/`
directory and is started with:

```bash
cd api && uvicorn main:app --host 0.0.0.0 --port 8000
```

Note the public URL once deployed (e.g., `https://permitiq-api.railway.app`).

### 2. Connect Streamlit Cloud to GitHub

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with your GitHub account
3. Click **New app**
4. Select the repository: `steviejoe23/PermitIQ`
5. Set the branch to: `main`

### 3. Configure the App

In the Streamlit Cloud deployment form:

- **Main file path:** `frontend/app.py`
  - Alternatively, use `streamlit_app.py` (root-level entry point that delegates
    to `frontend/app.py`). This is useful if Streamlit Cloud has trouble with
    subdirectory paths.
- **Python version:** 3.9 or later
- **Requirements file:** Streamlit Cloud will auto-detect `frontend/requirements.txt`
  if using `frontend/app.py` as the entry point, or the root `requirements.txt`
  if using `streamlit_app.py`. If it does not, specify the path manually.

### 4. Set Environment Variables

In the Streamlit Cloud app settings, go to **Secrets** or **Advanced settings**
and add:

```
PERMITIQ_API_URL=https://api.permitiq.com
```

Replace the URL with your actual deployed backend URL from Step 1.

The frontend reads this variable at runtime:
```python
API_URL = os.environ.get("PERMITIQ_API_URL", "http://127.0.0.1:8000")
```

### 5. Deploy

Click **Deploy** and wait for the build to complete. Streamlit Cloud will:

1. Clone the repository
2. Install dependencies from the requirements file
3. Start the Streamlit app

The app will be available at a URL like:
`https://steviejoe23-permitiq-frontend-app-XXXX.streamlit.app`

### 6. Verify

- Confirm the app loads and the dark theme renders correctly
- Test an address search to verify the API connection works
- Check that maps render (pydeck)
- Test a prediction to confirm full end-to-end flow

## Configuration Files

| File | Purpose |
|------|---------|
| `.streamlit/config.toml` | Theme (dark, matching #1a1a2e), server settings, upload limits |
| `frontend/requirements.txt` | Frontend-only Python dependencies |
| `streamlit_app.py` | Root-level entry point (alternative to `frontend/app.py`) |

## Troubleshooting

### App cannot connect to API
- Verify `PERMITIQ_API_URL` is set in Streamlit Cloud secrets/settings
- Confirm the backend is running and publicly accessible
- Check that the backend has CORS enabled for the Streamlit Cloud domain

### Import errors
- Streamlit Cloud should pick up `frontend/requirements.txt` automatically
- If not, try using `streamlit_app.py` as the main file and place a
  `requirements.txt` in the project root with the same contents

### Theme not applying
- The `.streamlit/config.toml` must be committed to the repository
- Streamlit Cloud reads it automatically from the repo root

### Private repository access
- Streamlit Cloud can access private repos once you authorize it via GitHub OAuth
- Ensure the GitHub account has read access to the repository

## Custom Domain (Optional)

Streamlit Cloud supports custom domains on paid plans. To use
`app.permitiq.com`:

1. Go to app settings in Streamlit Cloud
2. Add custom domain: `app.permitiq.com`
3. Create a CNAME DNS record pointing to the Streamlit Cloud URL
