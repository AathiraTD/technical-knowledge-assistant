<#
.SYNOPSIS
    Start the Lime Green technical assistant for a demonstration, or say
    precisely why it cannot start.

.DESCRIPTION
    One command between a clean clone and a working page. It runs the checks in
    dependency order -- interpreter, packages, model server, models, index,
    readiness -- and stops at the first one that fails with the command that
    fixes it, rather than letting the failure surface later as a traceback from
    inside a module the reader has never opened.

    Nothing here is a second implementation of the readiness check.
    `scripts/preflight.py` covers what must be true before `assistant` can be
    imported; `python -m assistant.health` covers everything after. This script
    orders them, offers to build the index, and then starts the server.

    The local path, not the container path. Docker is the production-shaped
    stack and is verified separately by scripts/verify-docker-deployment.ps1 --
    for a demonstration, a local Ollama that already holds the models beats a
    container that has to pull four gigabytes of them.

.PARAMETER Port
    Port for the web page. Default 8765, matching `python -m assistant.ui`.

.PARAMETER CheckOnly
    Run every check and print the readiness table, but do not start the server.
    This is what to run before walking into the room.

.PARAMETER BuildIndex
    Build the index without asking. Without it, a missing index is offered as a
    prompt; with -CheckOnly it is only reported.

.PARAMETER Rebuild
    Force a full rebuild, retaining version history. Slower -- use it when the
    embedding model or the chunker has changed.

.PARAMETER VisionDemo
    Make the vision model part of readiness. Off by default: images are handled
    by policy rather than by capability (DECISIONS 16), so a machine without a
    vision model is not a machine that cannot answer questions.

.PARAMETER Audience
    Which audiences this server may read: public, trade, staff. Default public.
    A request can narrow this and can never widen it.

.PARAMETER NoBrowser
    Do not open a browser window on start.

.EXAMPLE
    .\scripts\start-demo.ps1 -CheckOnly
    Everything the demonstration needs, verified, without starting anything.

.EXAMPLE
    .\scripts\start-demo.ps1
    The same checks, then the page at http://127.0.0.1:8765/
#>
[CmdletBinding()]
param(
    [int]    $Port = 8765,
    [switch] $CheckOnly,
    [switch] $BuildIndex,
    [switch] $Rebuild,
    [switch] $VisionDemo,
    [string] $Audience = "public",
    [switch] $NoBrowser
)

$ErrorActionPreference = "Stop"

# The repository root is the parent of scripts/. Resolved rather than assumed,
# because every path below is relative to it and the caller's working directory
# is not something this script gets to choose.
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

function Write-Step  { param([string] $Text) Write-Host "`n$Text" -ForegroundColor Cyan }
function Write-Good  { param([string] $Text) Write-Host "  ok    $Text" -ForegroundColor Green }
function Write-Bad   { param([string] $Text) Write-Host "  FAIL  $Text" -ForegroundColor Red }
function Write-Note  { param([string] $Text) Write-Host "        $Text" -ForegroundColor DarkGray }

function Stop-With {
    param([string] $Problem, [string[]] $Fix)
    Write-Bad $Problem
    Write-Host ""
    Write-Host "To fix:" -ForegroundColor Yellow
    foreach ($line in $Fix) { Write-Host "  $line" -ForegroundColor Yellow }
    Write-Host ""
    exit 1
}

Write-Host "Lime Green technical assistant — demonstration startup" -ForegroundColor White
Write-Host $Root -ForegroundColor DarkGray

# --------------------------------------------------------------- interpreter

Write-Step "1/5  Interpreter and packages"

$python = (Get-Command python -ErrorAction SilentlyContinue)
if (-not $python) {
    Stop-With "python is not on PATH" @(
        "install Python 3.11 or later from https://www.python.org/downloads/",
        "then re-run this script"
    )
}

# Everything that must hold before `assistant` is importable, checked by a
# standard-library-only script so a missing dependency prints a sentence rather
# than a ModuleNotFoundError from inside the package.
$preflight = & python scripts/preflight.py 2>&1
$preflightOk = ($LASTEXITCODE -eq 0)
$preflight | ForEach-Object { Write-Note $_ }

if (-not $preflightOk) {
    Stop-With "the application cannot be imported on this machine" @(
        "pip install -r requirements.txt",
        "then re-run this script"
    )
}
Write-Good "python and the six pinned packages"

# ------------------------------------------------------------------- ollama

Write-Step "2/5  Model server"

$ollamaHost = if ($env:OLLAMA_HOST) { $env:OLLAMA_HOST } else { "http://127.0.0.1:11434" }
$tags = $null
try {
    $tags = (Invoke-RestMethod -Uri "$($ollamaHost.TrimEnd('/'))/api/tags" -TimeoutSec 5).models.name
} catch {
    Stop-With "nothing is answering at $ollamaHost" @(
        "start the model server in another terminal:  ollama serve",
        "or install it from https://ollama.com/download",
        "if it runs elsewhere, set OLLAMA_HOST and re-run"
    )
}
Write-Good "ollama reachable at $ollamaHost"

# Read the model tags out of the application's own configuration rather than
# repeating them here. Two copies of a model tag is how a demonstration ends up
# pulling one model and querying another.
$models = & python -c @"
from assistant import ollama
from assistant.vision import VISION_MODEL
print(ollama.EMBED_MODEL); print(ollama.GENERATION_MODEL); print(VISION_MODEL)
"@ 2>&1
if ($LASTEXITCODE -ne 0) {
    Stop-With "could not read the configured model tags" @($models)
}
$embedModel, $generationModel, $visionModel = $models

# `ollama list` prints qwen3.5:4b for a tag pulled as qwen3.5:4b and
# nomic-embed-text:latest for one pulled without a tag, so compare on the bare
# name as well as the whole string.
function Test-Pulled {
    param([string] $Tag, [string[]] $Have)
    if ($Have -contains $Tag) { return $true }
    $bare = $Tag.Split(":")[0]
    foreach ($h in $Have) { if ($h.Split(":")[0] -eq $bare) { return $true } }
    return $false
}

$needed = [ordered] @{ "embedding" = $embedModel; "generation" = $generationModel }
if ($VisionDemo) { $needed["vision"] = $visionModel }

$absent = @()
foreach ($role in $needed.Keys) {
    $tag = $needed[$role]
    if (Test-Pulled -Tag $tag -Have $tags) {
        Write-Good "$role model $tag"
    } else {
        Write-Bad "$role model $tag is not pulled"
        $absent += "ollama pull $tag"
    }
}
if ($absent.Count -gt 0) {
    Stop-With "a required model is not pulled" ($absent + @(
        "about 4 GB in total; allow time on a slow connection"
    ))
}

# -------------------------------------------------------------------- index

Write-Step "3/5  Knowledge index"

$indexPath = Join-Path $Root "data/index/knowledge.db"
$wantBuild = $Rebuild -or (-not (Test-Path $indexPath))

if ($wantBuild -and -not $BuildIndex -and -not $Rebuild) {
    if ($CheckOnly) {
        Stop-With "no index at data/index/knowledge.db" @(
            "build it:  python -m assistant.index",
            "about two minutes; the embedding cache ships, so nothing is re-embedded"
        )
    }
    Write-Host "  No index at data/index/knowledge.db." -ForegroundColor Yellow
    $reply = Read-Host "  Build it now? About two minutes. [Y/n]"
    if ($reply -and $reply.Trim().ToLower().StartsWith("n")) {
        Stop-With "cannot serve without an index" @("python -m assistant.index")
    }
    $BuildIndex = $true
}

if ($wantBuild -and ($BuildIndex -or $Rebuild)) {
    Write-Note "python -m assistant.index$(if ($Rebuild) { ' --rebuild' })"
    if ($Rebuild) { & python -m assistant.index --rebuild } else { & python -m assistant.index }
    if ($LASTEXITCODE -ne 0) {
        Stop-With "indexing failed" @(
            "the previous release is untouched — a failed build cannot replace a good one",
            "read the error above, then re-run:  python -m assistant.index"
        )
    }
}
Write-Good "index present"

# ---------------------------------------------------------------- readiness

Write-Step "4/5  Readiness"

if ($VisionDemo) { $env:ASSISTANT_VISION_DEMO = "1" }

& python -m assistant.health
$ready = ($LASTEXITCODE -eq 0)

if (-not $ready) {
    Write-Host ""
    Write-Host "Not ready — the table above names each failing check and its fix." -ForegroundColor Red
    Write-Host ""
    exit 1
}

# ------------------------------------------------------------------- serve

if ($CheckOnly) {
    Write-Step "5/5  Check only — not starting the server"
    Write-Host ""
    Write-Host "Ready. Start it with:" -ForegroundColor Green
    Write-Host "  .\scripts\start-demo.ps1" -ForegroundColor Green
    Write-Host ""
    exit 0
}

Write-Step "5/5  Serving"
Write-Host "  page        http://127.0.0.1:$Port/"
Write-Host "  readiness   http://127.0.0.1:$Port/ready"
Write-Host "  metrics     http://127.0.0.1:$Port/metrics"
Write-Host "  audience    $Audience  (asserted, not authenticated)"
Write-Host "  new chat    http://127.0.0.1:$Port/new"
Write-Host ""
Write-Host "  Ctrl-C to stop. Structured JSON events go to stderr." -ForegroundColor DarkGray
Write-Host ""

$serveArgs = @("-m", "assistant.ui", "--port", "$Port", "--allow-audience", $Audience)
if ($NoBrowser) { $serveArgs += "--no-browser" }

& python @serveArgs
exit $LASTEXITCODE
