FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1
WORKDIR /app
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --system --uid 10001 --no-create-home pulse
COPY requirements.txt .
RUN python -m pip install -r requirements.txt
COPY server.py model_adapter.py lite.py common.py feats.py clf_features.py ./
COPY static/ ./static/
COPY weights/ ./weights/
USER 10001
EXPOSE 8000
CMD ["python", "server.py"]
