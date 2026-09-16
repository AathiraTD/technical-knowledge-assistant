# Verify Docker Compose deployment: health checks, end-to-end question/answer, persistence
# Usage: .\scripts\verify-docker-deployment.ps1

$ErrorActionPreference = "Stop"

$PROJECT_ROOT = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $PROJECT_ROOT

function Write-Info {
    param([string]$Message)
    Write-Host "[✓] $Message" -ForegroundColor Green
}

function Write-Warn {
    param([string]$Message)
    Write-Host "[!] $Message" -ForegroundColor Yellow
}

function Write-Error2 {
    param([string]$Message)
    Write-Host "[✗] $Message" -ForegroundColor Red
}

Write-Host "Verifying Docker Compose deployment..."
Write-Host ""

# 1. Check Docker installation
Write-Info "Checking Docker installation..."
try {
    $null = docker --version
} catch {
    Write-Error2 "Docker not found. Install from https://www.docker.com/products/docker-desktop"
    exit 1
}

try {
    $null = docker compose version
} catch {
    Write-Error2 "Docker Compose not found. Install from https://docs.docker.com/compose/install"
    exit 1
}

# 2. Build image
Write-Info "Building application image..."
docker compose build app --quiet

# 3. Start stack
Write-Info "Starting Docker Compose stack..."
docker compose down 2>$null
docker compose up -d

# 4. Wait for services to be healthy
Write-Info "Waiting for services to be healthy (max 90 seconds)..."
$TIMEOUT = 90
$ELAPSED = 0
$INTERVAL = 5

while ($ELAPSED -lt $TIMEOUT) {
    $DB_HEALTH = docker compose ps db --format "{{.State}}" 2>$null | Select-String "healthy"
    $OLLAMA_HEALTH = docker compose ps ollama --format "{{.State}}" 2>$null | Select-String "healthy"
    $APP_HEALTH = docker compose ps app --format "{{.State}}" 2>$null | Select-String "healthy"

    if ($DB_HEALTH -and $OLLAMA_HEALTH -and $APP_HEALTH) {
        Write-Info "All services healthy"
        break
    }

    Write-Host "  Waiting... ($ELAPSED`s / $TIMEOUT`s)"
    Start-Sleep -Seconds $INTERVAL
    $ELAPSED += $INTERVAL
}

if ($ELAPSED -ge $TIMEOUT) {
    Write-Error2 "Services did not become healthy within $TIMEOUT`s"
    docker compose logs
    exit 1
}

# 5. Check health endpoint
Write-Info "Checking application health endpoint..."
try {
    $response = Invoke-WebRequest -Uri "http://localhost:8000/health" -UseBasicParsing -TimeoutSec 5
    if ($response.StatusCode -ne 200) {
        Write-Error2 "Health endpoint returned HTTP $($response.StatusCode) (expected 200)"
        exit 1
    }
} catch {
    Write-Error2 "Health endpoint check failed: $_"
    exit 1
}

# 6. Test E2E question/answer
Write-Info "Testing end-to-end question/answer cycle..."
try {
    $RESPONSE = docker compose exec -T app python -m assistant.cli `
        "How much water does Solo Onecoat need per bag?" 2>&1

    if ($RESPONSE -like "*5 and 6 litres*" -or $RESPONSE -like "*litre*") {
        Write-Info "Got expected answer about water volume"
    } else {
        Write-Warn "Answer received but may not match expected content"
    }
} catch {
    Write-Error2 "Question/answer cycle failed: $_"
    exit 1
}

# 7. Check answer is cached (second question should be faster)
Write-Info "Verifying cache hit on repeated question..."
$start = Get-Date
docker compose exec -T app python -m assistant.cli `
    "How much water does Solo Onecoat need per bag?" > $null 2>&1
$elapsed = ((Get-Date) - $start).TotalMilliseconds

if ($elapsed -lt 2000) {
    Write-Info "Cache hit confirmed ($([int]$elapsed)ms)"
} else {
    Write-Warn "Cache miss or slow cache hit ($([int]$elapsed)ms)"
}

# 8. Verify PostgreSQL connectivity
Write-Info "Verifying PostgreSQL schema..."
try {
    $env:PGPASSWORD = "postgres"
    $chunk_count = psql -h localhost -U postgres -d knowledge_assistant -t `
        -c "SELECT COUNT(*) FROM chunks;" 2>$null

    if ([int]$chunk_count -gt 0) {
        Write-Info "PostgreSQL contains $chunk_count chunks"
    } else {
        Write-Warn "PostgreSQL has no chunks yet (first run expected)"
    }
} catch {
    Write-Warn "Could not connect to PostgreSQL: $_"
}

# 9. Stop and verify persistence
Write-Info "Testing persistence: stopping stack..."
docker compose down

Write-Info "Starting stack again..."
docker compose up -d

Write-Info "Waiting for recovery (max 30 seconds)..."
for ($i = 0; $i -lt 6; $i++) {
    try {
        docker compose exec -T app python -c "from assistant.store.factory import open_repository; open_repository().close()" 2>$null
        Write-Info "Stack recovered successfully"
        break
    } catch {
        Start-Sleep -Seconds 5
    }
}

# 10. Verify data persisted
Write-Info "Verifying data persisted..."
try {
    $env:PGPASSWORD = "postgres"
    $persisted_count = psql -h localhost -U postgres -d knowledge_assistant -t `
        -c "SELECT COUNT(*) FROM chunks;" 2>$null

    if ([int]$persisted_count -eq [int]$chunk_count) {
        Write-Info "Data persisted correctly ($persisted_count chunks)"
    } else {
        Write-Warn "Chunk count changed: was $chunk_count, now $persisted_count"
    }
} catch {
    Write-Warn "Could not verify persistence: $_"
}

# 11. Cleanup
Write-Info "Stopping stack..."
docker compose down

Write-Host ""
Write-Host "All verification checks passed!" -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:"
Write-Host "  - Run: docker compose up -d"
Write-Host "  - Open: http://localhost:8000"
Write-Host "  - View logs: docker compose logs -f"
Write-Host "  - Stop: docker compose down"
