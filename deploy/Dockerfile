# Non-root, pinned base, dependencies installed before source so a code change
# does not re-resolve the dependency tree.
FROM python:3.13-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# libpq is what psycopg needs at runtime; the binary wheel is unavailable on
# some platforms, so the library is installed rather than relied upon.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libpq5 \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt "psycopg[binary]==3.2.3"

COPY assistant/ ./assistant/
COPY config/ ./config/
COPY db/ ./db/
COPY eval/ ./eval/
COPY tests/ ./tests/

RUN useradd --create-home --uid 10001 assistant \
 && mkdir -p /app/data/index \
 && chown -R assistant:assistant /app
USER assistant

EXPOSE 8765
CMD ["python", "-m", "assistant.ui", "--no-browser", "--host", "0.0.0.0"]
