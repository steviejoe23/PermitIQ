#!/bin/bash
# PermitIQ — Weekly Data Update (local backup)
# Primary method: GitHub Actions (runs every Sunday 2 AM EST)
# This script exists for manual local runs: ./auto_weekly_update.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "========================================"
echo "  PermitIQ Weekly Update — $(date)"
echo "========================================"

# Activate virtual environment
source zoning-env/bin/activate

# Step 1: Pull latest data from Boston Open Data
python3 auto_update_data.py

# Step 2: Scrape new ZBA decision PDFs from boston.gov
python3 auto_scrape_decisions.py

# Step 3: Pull new ZBA hearing transcripts from YouTube
echo ""
echo "========================================"
echo "  Pulling ZBA Transcripts from YouTube"
echo "========================================"
python3 auto_pull_transcripts.py

# Step 4: If data changed, commit and push (triggers auto-deploy)
if ! git diff --quiet -- '*.csv' 'data/'; then
    echo ""
    echo "New data found — committing and pushing..."
    git add zba_tracker.csv building_permits.csv property_assessment_fy2026.csv zba_cases_cleaned.csv
    git add data/transcript_manifest.json data/zba_transcripts/*.txt data/zba_transcripts/transcript_features.csv 2>/dev/null || true
    git commit -m "Auto-update: weekly data + transcripts refresh $(date +%Y-%m-%d)"
    git push origin main
    echo "Pushed. Streamlit Cloud and Railway will auto-deploy."
else
    echo ""
    echo "No new data. Everything is current."
fi
