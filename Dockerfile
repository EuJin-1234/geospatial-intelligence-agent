# GeoInsight Agent API container.
#
# Note: geospatial packages such as geopandas, shapely, and osmnx may need
# additional native libraries in stricter production images. The Python slim
# image is kept here as a lightweight starting point for API serving.
# Ollama is not run inside this container; configure it separately if needed.

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY configs ./configs
COPY src ./src
COPY app.py ./
COPY data/reports ./data/reports
COPY data/features ./data/features
RUN mkdir -p data/maps data/raw data/processed data/index

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir .

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "geoinsight.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
