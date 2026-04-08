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

# Pull latest data from Boston Open Data
python3 auto_update_data.py

# If data changed, commit and push (triggers auto-deploy)
if ! git diff --quiet -- '*.csv'; then
    echo ""
    echo "New data found — committing and pushing..."
    git add zba_tracker.csv building_permits.csv property_assessment_fy2026.csv zba_cases_cleaned.csv
    git commit -m "Auto-update: weekly Boston Open Data refresh $(date +%Y-%m-%d)"
    git push origin main
    echo "Pushed. Streamlit Cloud and Railway will auto-deploy."
else
    echo ""
    echo "No new data. Everything is current."
fi
