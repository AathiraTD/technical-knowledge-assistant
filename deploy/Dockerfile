# Non-root, pinned base, dependencies installed before source so a code change
# does not re-resolve the dependency tree.
FROM python:3.13-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app
ARG PIP_INDEX_URL

# libpq is what psycopg needs at runtime; the binary wheel is unavailable on
# some platforms, so the library is installed rather than relied upon.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libpq5 \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
# PIP_INDEX_URL is declared above and has to be used here, not merely accepted.
# On a network that intercepts TLS, a container reaching pypi.org directly fails
# with an SSLV3 handshake alert while the host installs perfectly well through
# its own proxied index; passing that index in is the difference between the
# deployment adapter being verifiable and being theoretical.
#
#   docker build -f deploy/Dockerfile \
#     --build-arg PIP_INDEX_URL="$(pip config get global.index-url)" -t assistant .
#
# It defaults to PyPI, so an unproxied network needs no argument.
ARG PIP_INDEX_URL=https://pypi.org/simple
RUN pip install --no-cache-dir --index-url "$PIP_INDEX_URL" \
        -r requirements.txt "psycopg[binary]==3.3.3"

COPY assistant/ ./assistant/
COPY config/ ./config/
COPY db/ ./db/
COPY eval/ ./eval/
COPY tests/ ./tests/
COPY data/cache/ ./data/cache/
COPY data/embeddings.db ./data/index/embeddings.db

RUN useradd --create-home --uid 10001 assistant \
 && mkdir -p /app/data/index \
 && chown -R assistant:assistant /app
USER assistant

EXPOSE 8765
CMD ["python", "-m", "assistant.ui", "--no-browser", "--host", "0.0.0.0"]
