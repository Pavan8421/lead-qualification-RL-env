# Build from repo root:  docker build -f server/Dockerfile .
# Context directory must be lead_triage_env/ (this folder's parent of server/)

FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY __init__.py models.py client.py openenv.yaml ./
COPY task_tier.py dynamics.py features.py rewards.py grader.py ./
COPY server ./server/

RUN pip install --no-cache-dir -e .

ENV HOST=0.0.0.0
ENV PORT=8000
ENV PYTHONUNBUFFERED=1
ENV ENABLE_WEB_INTERFACE = true

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')" || exit 1

CMD ["sh", "-c", "uvicorn lead_triage_env.server.app:app --host ${HOST} --port ${PORT}"]
