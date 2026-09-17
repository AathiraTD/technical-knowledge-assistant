<#
.SYNOPSIS
    Verify the container deployment: the image builds, runs as a non-root user,
    carries what it copies, and starts the application.

.DESCRIPTION
    This targets deploy/compose.yaml and deploy/Dockerfile, which are the
    current stack -- the ones CI builds and DECISIONS 19 describes. It used to
    target the root docker-compose.yml, which is a stale parallel stack that
    runs as root, never builds an index and is referenced by nothing.

    Three bugs of its own are fixed while retargeting it, each of which made it
    report on something other than this repository:

      - $PROJECT_ROOT resolved to the parent of the repository, because
        $PSScriptRoot is already scripts/ and it was split twice.
      - It asked a question with `python -m assistant.cli "How much ..."`,
        which argparse rejects: the flag is -q.
      - It shelled out to psql on the host, which is not a dependency of this
        project and is absent on the build machine.

    Two modes, because they cost three orders of magnitude apart in time:

      -Quick (default)  image-level checks. No model pull, no database. Minutes.
      -Full             the whole compose stack, including a ~4 GB model pull
                        on first run. Not a demonstration path.

.PARAMETER PipIndexUrl
    Passed to the build as PIP_INDEX_URL. Needed on a network that intercepts
    TLS, where a container reaching pypi.org directly fails with
    SSLV3_ALERT_HANDSHAKE_FAILURE while the host installs perfectly well.
    Find yours with `pip config list`. Defaults to PyPI.

.EXAMPLE
    .\scripts\verify-docker-deployment.ps1
    .\scripts\verify-docker-deployment.ps1 -PipIndexUrl "https://my.proxy/pypi/simple/"
    .\scripts\verify-docker-deployment.ps1 -Full
#>
[CmdletBinding()]
param(
    [switch] $Full,
    [switch] $Quick,
    [string] $PipIndexUrl = "https://pypi.org/simple",
    [string] $Tag = "lime-green-assistant:verify"
)

$ErrorActionPreference = "Stop"

# $PSScriptRoot is scripts/, so the repository is its parent. Splitting twice
# pointed the old version at the directory above the repository, where every
# relative path below silently referred to something else.
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

$Compose = "deploy/compose.yaml"
$Dockerfile = "deploy/Dockerfile"

$script:Failures = @()

function Write-Step { param([string] $T) Write-Host "`n$T" -ForegroundColor Cyan }
function Pass { param([string] $T) Write-Host "  ok    $T" -ForegroundColor Green }
function Fail {
    param([string] $T, [string] $Detail = "")
    Write-Host "  FAIL  $T" -ForegroundColor Red
    if ($Detail) { Write-Host "        $Detail" -ForegroundColor DarkGray }
    $script:Failures += $T
}
function Note { param([string] $T) Write-Host "        $T" -ForegroundColor DarkGray }

Write-Host "Container deployment verification" -ForegroundColor White
Note "$Dockerfile and $Compose — the current stack"
Note "the root Dockerfile / docker-compose.yml are stale; see docs/demo-runbook.md"

# ------------------------------------------------------------------- docker

Write-Step "1/6  Docker available"
try {
    $null = docker version --format '{{.Server.Version}}' 2>&1
    if ($LASTEXITCODE -ne 0) { throw "daemon not responding" }
    Pass "docker daemon responding"
} catch {
    Fail "docker is not available" $_.Exception.Message
    Note "install Docker Desktop, or start it, then re-run"
    exit 1
}

# ------------------------------------------------------------------ compose

Write-Step "2/6  Compose file validates"
# --env-file the example, because POSTGRES_PASSWORD has no default and `config`
# fails on the missing variable exactly as `up` would. That is the intended
# behaviour rather than a problem to work around: it is what stops a database
# starting with a committed password.
docker compose -f $Compose --env-file deploy/.env.example config --quiet 2>&1 | Out-Null
if ($LASTEXITCODE -eq 0) { Pass "$Compose is valid" } else { Fail "$Compose did not validate" }

# -------------------------------------------------------------------- build

Write-Step "3/6  Image builds"
Note "docker build -f $Dockerfile --build-arg PIP_INDEX_URL=$PipIndexUrl -t $Tag ."
$buildLog = docker build -f $Dockerfile --build-arg PIP_INDEX_URL="$PipIndexUrl" -t $Tag . 2>&1
if ($LASTEXITCODE -ne 0) {
    $tail = ($buildLog | Select-Object -Last 12) -join "`n"
    Fail "the image did not build" $tail
    if ($tail -match "SSLV3_ALERT_HANDSHAKE_FAILURE|ssl-verification-failed") {
        # Single quotes: a backtick is PowerShell's escape character, so a
        # backtick-quoted command inside a double-quoted string escapes the
        # closing quote and the parse error lands a hundred lines later.
        Note 'TLS interception: re-run with -PipIndexUrl from: pip config list'
    }
    if ($tail -match "not found") {
        Note "a COPY target is missing from the build context — check .dockerignore"
    }
    Write-Host ""
    exit 1
}
Pass "image built as $Tag"

# ---------------------------------------------------------------- non-root

Write-Step "4/6  Runs as a non-root user"
$uid = (docker run --rm $Tag id -u 2>&1).Trim()
if ($uid -and $uid -ne "0") { Pass "runs as uid $uid" }
else { Fail "container runs as root (uid '$uid')" "deploy/Dockerfile creates uid 10001" }

# ------------------------------------------------- what the image carries

Write-Step "5/6  The image carries what it copies"
# The .dockerignore and the Dockerfile disagreed once and the result was a
# broken build rather than a slow one, so each COPY target is checked by name.
$probe = @(
    @{ Path = "/app/assistant/health.py";        What = "application" },
    @{ Path = "/app/config/routing.json";        What = "authored configuration" },
    @{ Path = "/app/db/schema.postgres.sql";     What = "deployment schema" },
    @{ Path = "/app/tests/test_repository_contract.py"; What = "repository contract" },
    @{ Path = "/app/data/index/embeddings.db";   What = "shipped embedding cache" },
    @{ Path = "/app/data/cache";                 What = "crawled corpus" }
)
foreach ($item in $probe) {
    docker run --rm $Tag test -e $item.Path 2>&1 | Out-Null
    if ($LASTEXITCODE -eq 0) { Pass "$($item.What)  $($item.Path)" }
    else { Fail "$($item.What) missing at $($item.Path)" "excluded by .dockerignore?" }
}

# The application must import inside the image, which is a different claim from
# the packages having installed: a native wheel can install and fail to load.
$imported = docker run --rm $Tag python -c "import assistant.ui, assistant.health, assistant.graph; print('ok')" 2>&1
if ($imported -match "ok") { Pass "the application imports inside the image" }
else { Fail "the application does not import inside the image" (($imported | Select-Object -Last 4) -join " ") }

# Hosted tracing must be off inside the image too, not only on the build
# machine -- it is closed in code, so this is checkable without a network.
$tracing = docker run --rm -e LANGCHAIN_TRACING_V2=true -e LANGSMITH_API_KEY=fake $Tag `
    python -c "import assistant, os; print(os.environ['LANGCHAIN_TRACING_V2'])" 2>&1
if ($tracing -match "false") { Pass "hosted tracing stays off under a hostile environment" }
else { Fail "hosted tracing was not disabled inside the image" "got '$tracing'" }

# ----------------------------------------------------------------- the stack

if ($Full) {
    Write-Step "6/6  Full stack"
    Note "first run pulls about 4 GB of models; this is not a demonstration path"

    if (-not (Test-Path "deploy/.env")) {
        Fail "deploy/.env is missing" "cp deploy/.env.example deploy/.env, then set POSTGRES_PASSWORD"
        Write-Host ""
        exit 1
    }

    docker compose -f $Compose up -d 2>&1 | Out-Null
    if ($LASTEXITCODE -ne 0) { Fail "compose up failed" }

    Note "waiting for readiness (up to 20 minutes on a first run)"
    $ready = $false
    foreach ($attempt in 1..80) {
        docker compose -f $Compose exec -T app python -m assistant.health 2>&1 | Out-Null
        if ($LASTEXITCODE -eq 0) { $ready = $true; break }
        Start-Sleep -Seconds 15
    }
    if ($ready) {
        Pass "the stack reports READY"
        docker compose -f $Compose exec -T app python -m assistant.health
    } else {
        Fail "the stack did not become ready"
        docker compose -f $Compose logs --tail 40
    }

    # A real question, with the flag argparse actually defines.
    $answer = docker compose -f $Compose exec -T app `
        python -m assistant.cli -q "How much water does Solo Onecoat need per bag?" 2>&1
    if ($answer -match "litre") { Pass "answered a question end to end" }
    else { Fail "no answer from the container" (($answer | Select-Object -Last 4) -join " ") }

    # Persistence: the database must survive a restart of the stack.
    # Held in a variable rather than written inline, because PowerShell parses
    # the parentheses of a Python call in an inline argument and fails with
    # "An expression was expected after '('".
    $countPassages = @'
from assistant.store.factory import open_repository
print(open_repository().snapshot().chunk_count)
'@

    $before = docker compose -f $Compose exec -T app python -c $countPassages 2>&1
    docker compose -f $Compose restart db 2>&1 | Out-Null
    Start-Sleep -Seconds 20
    $after = docker compose -f $Compose exec -T app python -c $countPassages 2>&1
    $beforeCount = "$before".Trim()
    $afterCount = "$after".Trim()
    if ($beforeCount -and $beforeCount -eq $afterCount) {
        Pass "state survived a database restart — $beforeCount passages"
    } else {
        Fail "state did not survive a restart" "before '$beforeCount' after '$afterCount'"
    }

    Note "leaving the stack up; 'docker compose -f $Compose down' to stop it"
} else {
    Write-Step "6/6  Full stack — skipped"
    Note "image-level checks only. Re-run with -Full for the whole stack,"
    Note "which pulls about 4 GB of models on a first run."
}

# ------------------------------------------------------------------ verdict

Write-Host ""
if ($script:Failures.Count -eq 0) {
    Write-Host "All checks passed." -ForegroundColor Green
    Write-Host ""
    Write-Host "For the demonstration, prefer the local path:" -ForegroundColor DarkGray
    Write-Host "  .\scripts\start-demo.ps1" -ForegroundColor DarkGray
    exit 0
}
Write-Host "$($script:Failures.Count) check(s) failed:" -ForegroundColor Red
foreach ($f in $script:Failures) { Write-Host "  $f" -ForegroundColor Red }
Write-Host ""
exit 1
