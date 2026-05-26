FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN python -m pip install --upgrade pip setuptools wheel \
    && python -m pip install --index-url https://download.pytorch.org/whl/cpu "torch>=2.0.0" "torchvision>=0.15.0" \
    && python -m pip install -r requirements.txt

COPY . .

RUN mkdir -p /app/data/FashionMNIST /app/checkpoints

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD sh -c 'curl --fail "http://localhost:${PORT:-${STREAMLIT_SERVER_PORT:-8501}}/_stcore/health" || exit 1'

CMD ["sh", "-c", "APP_PORT=\"${PORT:-${STREAMLIT_SERVER_PORT:-8501}}\"; exec python -m streamlit run app.py --server.address=0.0.0.0 --server.port=\"$APP_PORT\" --server.headless=true --browser.gatherUsageStats=false"]
