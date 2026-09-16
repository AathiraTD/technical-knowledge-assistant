# Docker Compose Deployment

A simple production-like stack using Docker Compose: PostgreSQL 16 + pgvector, Ollama, and the Lime Green Technical Knowledge Assistant application.

## Prerequisites

- Docker 20.10+
- Docker Compose 1.29+
- 24 GB RAM (for Ollama models)
- 10 GB disk space (for PostgreSQL data + Ollama models)

## Quick Start

### 1. Clone and Setup

```bash
git clone <repository> lime-green-assistant
cd lime-green-assistant
cp .env.example .env
```

### 2. Configure Environment

Edit `.env` to customize:

```bash
# PostgreSQL credentials (change password in production)
POSTGRES_PASSWORD=secure_password_here

# Ollama models (download happens on first startup)
OLLAMA_HOST=http://ollama:11434

# Application port
APP_PORT=8000
```

### 3. Start the Stack

```bash
docker compose up -d
```

Watch startup:

```bash
docker compose logs -f app
```

Expected output:
- PostgreSQL schema applied
- Ollama models loaded
- Application listening on http://localhost:8000

### 4. Initialize the Index

First run only: the embedded SQLite index ships with the repository. To use PostgreSQL:

```bash
docker compose exec app python -m assistant.cli \
  --index-store postgres \
  "What products are available?"
```

The index will be created and populated on first query. Initialization takes 2–5 minutes.

### 5. Verify End-to-End

```bash
# Ask a question via CLI
docker compose exec app python -m assistant.cli \
  "How much water does Solo Onecoat need per bag?"

# Or open the web UI
open http://localhost:8000
```

### 6. Run Tests Against PostgreSQL

```bash
export ASSISTANT_POSTGRES_DSN="postgresql://postgres:postgres@localhost:5432/knowledge_assistant"
python -m pytest tests/test_repository_contract.py::test_repo_initializes -v
```

## Stopping the Stack

### Clean Shutdown

```bash
docker compose down
```

Keeps volumes (data persists):

```bash
docker compose down -v
```

Removes everything (clean slate on next startup).

## Health Checks

All services report health status:

```bash
docker compose ps
```

Expected output:
```
NAME          STATUS
tka-postgres  healthy
tka-ollama    healthy
tka-app       healthy
```

## Production Deployment

### Secrets Management

Never commit `.env` to version control. Use a secrets manager:

**AWS Secrets Manager**
```bash
aws secretsmanager get-secret-value --secret-id tka/postgres-password
```

**HashiCorp Vault**
```bash
vault kv get secret/tka/postgres
```

### Database Backup

PostgreSQL data is stored in a Docker volume. Backup it:

```bash
docker compose exec db pg_dump -U postgres knowledge_assistant > backup.sql
```

Restore from backup:

```bash
docker compose exec -T db psql -U postgres knowledge_assistant < backup.sql
```

### Scaling

For multiple application instances, use Docker Swarm or Kubernetes:

**Docker Swarm**
```bash
docker swarm init
docker stack deploy -c docker-compose.yml tka
```

**Kubernetes**
Generate a Helm chart from docker-compose.yml:
```bash
kompose convert -f docker-compose.yml
kubectl apply -f *.yaml
```

## Troubleshooting

### PostgreSQL Won't Start

**Error: "database ... does not exist"**
- First startup, schema hasn't been created yet
- Application will create it on first question

**Error: "password authentication failed"**
- Check `.env` credentials match `POSTGRES_PASSWORD` and `ASSISTANT_POSTGRES_DSN`

### Ollama Models Not Loading

**Error: "connection refused"**
- Ollama service may not be healthy yet
- Check: `docker compose logs ollama`
- Wait 30–60 seconds for model pull to complete

**Error: "out of memory"**
- 24 GB RAM is the minimum
- Reduce model size: change `qwen3.5:4b` to `qwen3:4b-instruct` in application configuration

### Application Crashes on Startup

**Error: "sqlite3.ProgrammingError: SQLite objects created in a thread..."**
- Using SQLite from CLI in multi-threaded UI server
- Must use PostgreSQL in production
- Ensure `ASSISTANT_POSTGRES_DSN` is set

**Error: "IndexMismatch"**
- Embedding model changed but index was built with a different model
- Rebuild index: delete volume and restart
- Check `tests/test_health.py` for schema validation

## Monitoring and Observability

### Logs

```bash
# All services
docker compose logs -f

# Specific service
docker compose logs -f app

# Last 100 lines
docker compose logs -n 100 app

# Structured JSON (from application)
docker compose logs app | grep -E '"level"|"message"'
```

### Metrics Endpoint

Application exports Prometheus metrics:

```bash
curl http://localhost:8000/metrics
```

Metrics include:
- `tka_questions_total` — total questions answered
- `tka_generation_seconds` — model generation latency
- `tka_retrieval_seconds` — database query latency
- `tka_checks_failed_total` — safety check failures

### Database Monitoring

Connect directly to PostgreSQL:

```bash
docker compose exec db psql -U postgres -d knowledge_assistant -c \
  "SELECT COUNT(*) as chunks, MIN(created_at) as oldest FROM chunks;"
```

## Schema and Data Management

### View Current Index State

```bash
docker compose exec db psql -U postgres -d knowledge_assistant -c \
  "SELECT snapshot_id, document_count, chunk_count, created_at FROM index_snapshots ORDER BY created_at DESC LIMIT 1;"
```

### Reset the Index

```bash
docker compose exec db psql -U postgres -d knowledge_assistant -c \
  "TRUNCATE chunks, document_versions, documents CASCADE;"
```

Then restart the application — the index will be rebuilt on the next question.

### Migrate from SQLite to PostgreSQL

1. Start the Compose stack
2. Index a question against the embedded SQLite (creates schema)
3. Export from SQLite: `sqlite3 /path/to/index.db .dump > export.sql`
4. Import to PostgreSQL: `psql -f export.sql` (with adjustments for pgvector)

Easier path: let the application recreate the index on PostgreSQL. The embedded cache will reuse vectors and skip re-embedding.

## Development

### Run Tests in the Stack

```bash
docker compose exec app python -m pytest tests/test_repository_contract.py -v
```

### Access PostgreSQL CLI

```bash
docker compose exec db psql -U postgres -d knowledge_assistant
```

### Rebuild Application Image

```bash
docker compose build app --no-cache
docker compose up -d app
```

### View Generated Application Config

```bash
docker compose config
```

## References

- [PostgreSQL + pgvector Docker Image](https://hub.docker.com/r/ankane/pgvector)
- [Ollama Official Repository](https://github.com/ollama/ollama)
- [Docker Compose Documentation](https://docs.docker.com/compose/)
- DECISIONS.md — architecture and design rationale
- CLAUDE.md — production requirements and safety gates
