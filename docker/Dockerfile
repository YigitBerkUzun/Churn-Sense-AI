# ===== ChurnGuard AI — shared image for API and dashboard =====
FROM python:3.11-slim

# Avoid interactive prompts and keep Python output unbuffered.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# System deps needed by some scientific wheels (xgboost / shap).
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first to leverage Docker layer caching.
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy the application source.
COPY . .

# Make the project importable (src, api, app packages).
ENV PYTHONPATH=/app

# Default command runs the API; docker-compose overrides this for the dashboard.
EXPOSE 8000 8501
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
