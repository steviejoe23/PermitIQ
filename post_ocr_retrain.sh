#!/bin/bash
# PermitIQ — Post-OCR Retrain Pipeline
# Run this after overnight_rebuild.py completes.
# Usage: bash post_ocr_retrain.sh

set -e

cd "$(dirname "$0")"
source zoning-env/bin/activate

echo "============================================================"
echo "  PermitIQ Post-OCR Retrain Pipeline"
echo "============================================================"
echo ""

# Step 1: Clean OCR artifacts
echo ">>> Step 1/4: Cleaning OCR artifacts..."
python3 cleanup_ocr.py
echo ""

# Step 2: Audit quality
echo ">>> Step 2/4: Running OCR quality audit..."
python3 audit_ocr_quality.py
echo ""

# Step 3: Retrain model (leakage-free v3)
echo ">>> Step 3/4: Retraining model..."
python3 train_model_v2.py
echo ""

# Step 4: Run tests
echo ">>> Step 4/4: Running tests..."
# Start API in background for tests
cd api && uvicorn main:app --port 8000 &
API_PID=$!
sleep 5

cd ..
python3 -m pytest tests/test_api.py -q --tb=short

# Stop background API
kill $API_PID 2>/dev/null

echo ""
echo "============================================================"
echo "  RETRAIN COMPLETE"
echo "============================================================"
echo ""
echo "  Next steps:"
echo "    1. Review model_diagnostics.png for ROC/calibration curves"
echo "    2. Check model_history/training_log.jsonl for AUC comparison"
echo "    3. Restart the API: cd api && uvicorn main:app --reload --port 8000"
echo "    4. Restart the frontend: streamlit run frontend/app.py --server.port 8501"
echo ""
