#!/bin/bash
# PermitIQ Startup Script for Railway
# Downloads data files if not present, then starts the API

set -e

echo "=== PermitIQ API Startup ==="

# Check for required data files
MISSING=0
for f in zba_cases_cleaned.csv boston_parcels_zoning.geojson property_assessment_fy2026.csv; do
    if [ ! -f "$f" ]; then
        echo "MISSING: $f"
        MISSING=1
    else
        echo "FOUND: $f ($(du -h "$f" | cut -f1))"
    fi
done

if [ "$MISSING" = "1" ]; then
    echo ""
    echo "Data files missing. Downloading from DATA_URL..."
    if [ -z "$DATA_URL" ]; then
        echo "ERROR: DATA_URL environment variable not set."
        echo "Set DATA_URL to a URL containing the data files (zip or tar.gz)."
        echo "Starting without data files — some endpoints will fail."
    else
        echo "Downloading data from $DATA_URL ..."
        curl -sL "$DATA_URL" -o /tmp/data.tar.gz
        tar xzf /tmp/data.tar.gz -C /app/
        rm /tmp/data.tar.gz
        echo "Data files extracted."
    fi
fi

echo ""
echo "Starting API on port ${PORT:-8000}..."
exec uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 2
