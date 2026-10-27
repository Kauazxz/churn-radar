FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY churn_radar ./churn_radar
RUN pip install . && useradd --create-home --uid 10001 app

COPY examples ./examples
USER app

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthz')"

# One worker on purpose: tenant state is in memory and would not be shared between processes.
CMD ["uvicorn", "churn_radar.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
