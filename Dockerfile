FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.cache/huggingface

ARG VISION_MODEL_NAME=openai/clip-vit-base-patch32
ENV VISION_MODEL_NAME=${VISION_MODEL_NAME}

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

RUN mkdir -p /app/artifacts /app/defaults /app/.cache/huggingface \
    && chmod +x /app/docker-entrypoint.sh \
    && chown -R appuser:appuser /app/artifacts /app/defaults /app/.cache

USER appuser

RUN python -c "from transformers import AutoProcessor, CLIPModel; name='${VISION_MODEL_NAME}'; AutoProcessor.from_pretrained(name); CLIPModel.from_pretrained(name)" \
    && VISION_STATE_PATH=/app/defaults/prototypes.json VISION_DEVICE=cpu python bootstrap_demo.py

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl --fail http://localhost:8000/health || exit 1

ENTRYPOINT ["/app/docker-entrypoint.sh"]
CMD ["python", "-m", "uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
