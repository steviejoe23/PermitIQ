#!/bin/bash
# PermitIQ Weekly Data Update
# Runs every Sunday at 2am via cron
# Scrapes new decisions, updates external data, retrains model
#
# Manual run:
#   cd ~/Desktop/Boston\ Zoning\ Project
#   ./auto_weekly_update.sh
#
# Install cron job:
#   ./setup_cron.sh

set -e
cd "/Users/stevenspero/Desktop/Boston Zoning Project"
source zoning-env/bin/activate

LOG="logs/weekly_update_$(date +%Y%m%d_%H%M%S).log"
mkdir -p logs

echo "=== PermitIQ Weekly Update: $(date) ===" | tee -a "$LOG"

# Step 1: Scrape new ZBA decision PDFs from boston.gov
echo "[1/5] Scraping new decision PDFs..." | tee -a "$LOG"
python3 auto_scrape_decisions.py >> "$LOG" 2>&1 || echo "WARNING: Scrape failed (see log)" | tee -a "$LOG"

# Step 2: Pull fresh external data (ZBA tracker, property assessment, building permits)
echo "[2/5] Updating external data from data.boston.gov..." | tee -a "$LOG"
python3 auto_update_data.py >> "$LOG" 2>&1 || echo "WARNING: Data update failed (see log)" | tee -a "$LOG"

# Step 3: OCR any new PDFs and rebuild dataset
echo "[3/5] OCR processing new PDFs..." | tee -a "$LOG"
python3 overnight_rebuild.py >> "$LOG" 2>&1 || echo "WARNING: OCR/rebuild failed (see log)" | tee -a "$LOG"

# Step 4: Clean and normalize dataset
echo "[4/5] Cleaning dataset..." | tee -a "$LOG"
python3 cleanup_ocr.py >> "$LOG" 2>&1 || true
python3 normalize_decisions.py >> "$LOG" 2>&1 || true

# Step 5: Retrain model
echo "[5/5] Retraining model..." | tee -a "$LOG"
python3 train_model_v2.py >> "$LOG" 2>&1

# Copy model to API
if [ -f zba_model_v2.pkl ]; then
    cp zba_model_v2.pkl api/zba_model.pkl
    echo "Model deployed to api/zba_model.pkl" | tee -a "$LOG"
else
    echo "WARNING: zba_model_v2.pkl not found, model not deployed" | tee -a "$LOG"
fi

echo "=== Update complete: $(date) ===" | tee -a "$LOG"
echo "Log saved to: $LOG"
echo "Restart API to pick up model changes: cd api && uvicorn main:app --reload --port 8000"
