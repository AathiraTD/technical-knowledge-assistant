#!/usr/bin/env bash
# Verify the container deployment: the image builds, runs as a non-root user,
# carries what it copies, and starts the application.
#
# The POSIX companion to verify-docker-deployment.ps1, and corrected the same
# way. It used to target the root docker-compose.yml -- a stale parallel stack
# that runs as root, never builds an index and is referenced by no CI job -- and
# to ask questions with `python -m assistant.cli "How much ..."`, which argparse
# rejects because the flag is -q. It now targets deploy/, which is what CI
# builds and what DECISIONS 19 describes.
#
#   scripts/verify-docker-deployment.sh                 image-level checks
#   scripts/verify-docker-deployment.sh --full          the whole stack
#   PIP_INDEX_URL=https://my.proxy/pypi/simple/ scripts/verify-docker-deployment.sh
#
# PIP_INDEX_URL matters on a network that intercepts TLS, where a container
# reaching pypi.org directly fails with SSLV3_ALERT_HANDSHAKE_FAILURE while the
# host installs perfectly well. Find yours with `pip config list`.

set -euo pipefail

# scripts/ is already this file's directory, so the repository is its parent.
# Splitting twice -- which the previous version did -- pointed every relative
# path below at the directory above the repository.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

COMPOSE="deploy/compose.yaml"
DOCKERFILE="deploy/Dockerfile"
TAG="${TAG:-lime-green-assistant:verify}"
PIP_INDEX_URL="${PIP_INDEX_URL:-https://pypi.org/simple}"
FULL=0
[ "${1:-}" = "--full" ] && FULL=1

FAILURES=0

step() { printf '\n%s\n' "$1"; }
pass() { printf '  ok    %s\n' "$1"; }
note() { printf '        %s\n' "$1"; }
fail() { printf '  FAIL  %s\n' "$1"; FAILURES=$((FAILURES + 1)); }

echo "Container deployment verification"
note "$DOCKERFILE and $COMPOSE -- the current stack"
note "the root Dockerfile / docker-compose.yml are stale; see docs/demo-runbook.md"

step "1/6  Docker available"
if docker version --format '{{.Server.Version}}' >/dev/null 2>&1; then
    pass "docker daemon responding"
else
    fail "docker is not available"
    note "install Docker, or start the daemon, then re-run"
    exit 1
fi

step "2/6  Compose file validates"
# --env-file the example, because POSTGRES_PASSWORD has no default and `config`
# fails on the missing variable exactly as `up` would. That is intended: it is
# what stops a database starting with a committed password.
if docker compose -f "$COMPOSE" --env-file deploy/.env.example config --quiet >/dev/null 2>&1; then
    pass "$COMPOSE is valid"
else
    fail "$COMPOSE did not validate"
fi

step "3/6  Image builds"
note "docker build -f $DOCKERFILE --build-arg PIP_INDEX_URL=$PIP_INDEX_URL -t $TAG ."
if build_log=$(docker build -f "$DOCKERFILE" \
        --build-arg PIP_INDEX_URL="$PIP_INDEX_URL" -t "$TAG" . 2>&1); then
    pass "image built as $TAG"
else
    fail "the image did not build"
    printf '%s\n' "$build_log" | tail -12
    case "$build_log" in
        *SSLV3_ALERT_HANDSHAKE_FAILURE*|*ssl-verification-failed*)
            note "TLS interception: re-run with PIP_INDEX_URL from: pip config list" ;;
    esac
    case "$build_log" in
        *"not found"*)
            note "a COPY target is missing from the build context -- check .dockerignore" ;;
    esac
    exit 1
fi

step "4/6  Runs as a non-root user"
uid="$(docker run --rm "$TAG" id -u 2>/dev/null || echo unknown)"
if [ "$uid" != "0" ] && [ "$uid" != "unknown" ]; then
    pass "runs as uid $uid"
else
    fail "container runs as root (uid '$uid')"
    note "deploy/Dockerfile creates uid 10001"
fi

step "5/6  The image carries what it copies"
# .dockerignore and the Dockerfile disagreed once and the result was a broken
# build rather than a slow one, so each COPY target is checked by name.
while IFS='|' read -r path what; do
    if docker run --rm "$TAG" test -e "$path" >/dev/null 2>&1; then
        pass "$what  $path"
    else
        fail "$what missing at $path"
        note "excluded by .dockerignore?"
    fi
done <<'PROBE'
/app/assistant/health.py|application
/app/config/routing.json|authored configuration
/app/db/schema.postgres.sql|deployment schema
/app/tests/test_repository_contract.py|repository contract
/app/data/index/embeddings.db|shipped embedding cache
/app/data/cache|crawled corpus
PROBE

# Importing is a different claim from installing: a native wheel can install and
# still fail to load.
if docker run --rm "$TAG" python -c \
        "import assistant.ui, assistant.health, assistant.graph" >/dev/null 2>&1; then
    pass "the application imports inside the image"
else
    fail "the application does not import inside the image"
fi

# Hosted tracing is closed in code, so this is checkable with no network.
tracing="$(docker run --rm -e LANGCHAIN_TRACING_V2=true -e LANGSMITH_API_KEY=fake \
    "$TAG" python -c "import assistant, os; print(os.environ['LANGCHAIN_TRACING_V2'])" 2>&1 || true)"
if [ "$tracing" = "false" ]; then
    pass "hosted tracing stays off under a hostile environment"
else
    fail "hosted tracing was not disabled inside the image (got '$tracing')"
fi

if [ "$FULL" = "1" ]; then
    step "6/6  Full stack"
    note "first run pulls about 4 GB of models; this is not a demonstration path"

    if [ ! -f deploy/.env ]; then
        fail "deploy/.env is missing"
        note "cp deploy/.env.example deploy/.env, then set POSTGRES_PASSWORD"
        exit 1
    fi

    docker compose -f "$COMPOSE" up -d
    note "waiting for readiness (up to 20 minutes on a first run)"
    ready=0
    for _ in $(seq 1 80); do
        if docker compose -f "$COMPOSE" exec -T app python -m assistant.health >/dev/null 2>&1; then
            ready=1
            break
        fi
        sleep 15
    done
    if [ "$ready" = "1" ]; then
        pass "the stack reports READY"
        docker compose -f "$COMPOSE" exec -T app python -m assistant.health
    else
        fail "the stack did not become ready"
        docker compose -f "$COMPOSE" logs --tail 40
    fi

    # A real question, with the flag argparse actually defines.
    if docker compose -f "$COMPOSE" exec -T app python -m assistant.cli \
            -q "How much water does Solo Onecoat need per bag?" 2>&1 | grep -qi litre; then
        pass "answered a question end to end"
    else
        fail "no answer from the container"
    fi

    count='from assistant.store.factory import open_repository
print(open_repository().snapshot().chunk_count)'
    before="$(docker compose -f "$COMPOSE" exec -T app python -c "$count" 2>/dev/null | tr -d '\r\n ')"
    docker compose -f "$COMPOSE" restart db >/dev/null
    sleep 20
    after="$(docker compose -f "$COMPOSE" exec -T app python -c "$count" 2>/dev/null | tr -d '\r\n ')"
    if [ -n "$before" ] && [ "$before" = "$after" ]; then
        pass "state survived a database restart -- $before passages"
    else
        fail "state did not survive a restart (before '$before' after '$after')"
    fi

    note "leaving the stack up; 'docker compose -f $COMPOSE down' to stop it"
else
    step "6/6  Full stack -- skipped"
    note "image-level checks only. Re-run with --full for the whole stack,"
    note "which pulls about 4 GB of models on a first run."
fi

echo
if [ "$FAILURES" = "0" ]; then
    echo "All checks passed."
    echo
    echo "For the demonstration, prefer the local path: scripts/start-demo.ps1"
    exit 0
fi
echo "$FAILURES check(s) failed."
exit 1
