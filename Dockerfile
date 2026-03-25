FROM python:3.9-slim

# Install system deps for tesseract, gdal (geopandas), and pdf processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    libgdal-dev \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY api/ api/
COPY frontend/ frontend/
COPY train_model_v2.py .
COPY .env.example .env

# Data files are mounted as volumes (too large for image)
# See docker-compose.yml

EXPOSE 8000 8501
