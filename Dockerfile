FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PESTSCOPE_DEVICE=cpu \
    PESTSCOPE_MODEL_BUNDLE=/app/artifacts/models/pestnet_s_latest \
    PESTSCOPE_DEMO_CACHE_DIR=/app/artifacts/demo_assets \
    PESTSCOPE_REVIEW_DB=/app/artifacts/reviews.sqlite3

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 appuser

COPY requirements.txt .

RUN python -m pip install --upgrade pip setuptools wheel \
    && python -m pip install --index-url https://download.pytorch.org/whl/cpu "torch>=2.2,<3" \
    && python -m pip install -r requirements.txt

COPY --chown=appuser:appuser . .

RUN mkdir -p /app/artifacts/models /app/artifacts/demo_assets \
    && chown -R appuser:appuser /app/artifacts

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl --fail http://localhost:8000/api/v1/health/ready || exit 1

CMD ["python", "-m", "uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
