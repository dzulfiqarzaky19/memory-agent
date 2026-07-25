<#
.SYNOPSIS
  One-command setup for memory-agent -- checks prereqs, configures .env, starts Docker.

.DESCRIPTION
  Runs on Windows (PowerShell 5.1+). Detects Docker, prompts for provider
  choice, writes .env, starts containers, and prints harness configs.

  Usage: .\scripts\setup.ps1
#>

$ErrorActionPreference = "Stop"
$REPO_DIR = Split-Path -Parent $PSScriptRoot
Set-Location $REPO_DIR

# -- Colors --
$GREEN = "Green"; $YELLOW = "Yellow"; $RED = "Red"; $CYAN = "Cyan"
function info  { Write-Host "  == $args" -ForegroundColor $GREEN }
function warn  { Write-Host "  == $args" -ForegroundColor $YELLOW }
function fail  { Write-Host "  == $args" -ForegroundColor $RED; exit 1 }
function header{ Write-Host "`n$args" -ForegroundColor $CYAN }

# -- 1. Prerequisites --
header ">> Checking prerequisites"

$docker = Get-Command docker -ErrorAction SilentlyContinue
if (-not $docker) { fail "Docker not found. Install Docker Desktop from https://docs.docker.com/desktop/setup/install/windows-install/" }
info "docker found: $((docker --version 2>$null))"

# -- 2. Create .env if missing --
header ">> Configuration"

if (-not (Test-Path ".env")) {
    Copy-Item ".env.example" ".env"
    info "Created .env from .env.example"
} else {
    warn ".env already exists -- will not overwrite"
}

# -- 3. Provider picker --
Write-Host ""
Write-Host "  Which LLM / embedding provider do you want to use?" -ForegroundColor $CYAN
Write-Host "    1) Local (LM Studio / Ollama) -- default, works offline"
Write-Host "    2) OpenAI API"
Write-Host "    3) Groq"
Write-Host "    4) Other OpenAI-compatible API"
Write-Host "    5) Skip -- I will edit .env myself"
$choice = Read-Host "  Enter 1-5 (default: 1)"
if ([string]::IsNullOrWhiteSpace($choice)) { $choice = "1" }

switch ($choice) {
    "1" {
        info "Configuring for local provider"
        $llmUrl = "http://host.docker.internal:1234/v1"
        $embUrl = "http://host.docker.internal:1234/v1"
        $model = "google/gemma-4-e4b"
        $embModel = "text-embedding-nomic-embed-text-v1.5@q8_0"
        $embDims = "768"
        $apiKey = "not-needed"

        $localType = Read-Host "  Local type? (1) LM Studio (default) / (2) Ollama"
        if ($localType -eq "2") {
            $llmUrl = "http://host.docker.internal:11434/v1"
            $embUrl = "http://host.docker.internal:11434/v1"
            $model = "llama3.2"
            $embModel = "nomic-embed-text"
            $embDims = "768"
        }
    }
    "2" {
        info "Configuring for OpenAI"
        $apiKey = Read-Host "  Enter your OpenAI API key (sk-...)"
        $model = Read-Host "  LLM model (default: gpt-4o-mini)"
        if ([string]::IsNullOrWhiteSpace($model)) { $model = "gpt-4o-mini" }
        $embModel = Read-Host "  Embedding model (default: text-embedding-3-small)"
        if ([string]::IsNullOrWhiteSpace($embModel)) { $embModel = "text-embedding-3-small" }
        $embDims = Read-Host "  Embedding dimensions (1536 for 3-small, 3072 for 3-large, default: 1536)"
        if ([string]::IsNullOrWhiteSpace($embDims)) { $embDims = "1536" }
        $llmUrl = "https://api.openai.com/v1"
        $embUrl = "https://api.openai.com/v1"
    }
    "3" {
        info "Configuring for Groq"
        $apiKey = Read-Host "  Enter your Groq API key (gsk_...)"
        $model = Read-Host "  LLM model (default: llama-3.3-70b-versatile)"
        if ([string]::IsNullOrWhiteSpace($model)) { $model = "llama-3.3-70b-versatile" }
        $embModel = Read-Host "  Embedding model (default: nomic-embed-text-v1.5)"
        if ([string]::IsNullOrWhiteSpace($embModel)) { $embModel = "nomic-embed-text-v1.5" }
        $embDims = Read-Host "  Embedding dimensions (default: 768)"
        if ([string]::IsNullOrWhiteSpace($embDims)) { $embDims = "768" }
        $llmUrl = "https://api.groq.com/openai/v1"
        $embUrl = "https://api.groq.com/openai/v1"
    }
    "4" {
        info "Configuring custom OpenAI-compatible API"
        $llmUrl = Read-Host "  LLM API base URL (e.g. https://api.openai.com/v1)"
        $embUrl = Read-Host "  Embedding API base URL (same as above or different)"
        $apiKey = Read-Host "  API key"
        $model = Read-Host "  LLM model"
        $embModel = Read-Host "  Embedding model"
        $embDims = Read-Host "  Embedding dimensions"
    }
    default {
        warn "Skipping auto-config -- edit .env manually then re-run"
    }
}

# -- 4. Write provider settings to .env --
if ($choice -ne "5") {
    $envContent = Get-Content ".env" -Raw

    $replacements = @{
        "OPENAI_API_KEY=not-needed" = "OPENAI_API_KEY=$apiKey"
        "EMBEDDING_PROVIDER=openai" = "EMBEDDING_PROVIDER=openai"
        "EMBEDDING_MODEL=text-embedding-nomic-embed-text-v1.5@q8_0" = "EMBEDDING_MODEL=$embModel"
        "EMBEDDING_DIMENSIONS=768" = "EMBEDDING_DIMENSIONS=$embDims"
        "LLM_MODEL=google/gemma-4-e4b" = "LLM_MODEL=$model"
    }

    $envContent = $envContent -replace "OPENAI_BASE_URL=http://127.0.0.1:1234/v1", ""
    $envContent = $envContent -replace "(?m)^EMBEDDING_BASE_URL=.*`n?", ""
    $envContent = $envContent -replace "(?m)^LLM_BASE_URL=.*`n?", ""
    $envContent = $envContent -replace "(?m)^LLM_API_KEY=.*`n?", ""

    $envContent = $envContent.TrimEnd("`r`n") + @"

# -- Provider URLs (set by setup script) --
EMBEDDING_BASE_URL=$embUrl
LLM_BASE_URL=$llmUrl
"@

    if ($apiKey -ne "not-needed" -and $apiKey -ne "") {
        $envContent = $envContent + "`r`nLLM_API_KEY=$apiKey"
    }

    foreach ($k in $replacements.Keys) {
        $envContent = $envContent -replace [regex]::Escape($k), $replacements[$k]
    }

    Set-Content ".env" -Value $envContent -NoNewline
    info ".env updated"
}

# -- 5. Start Docker containers --
header ">> Starting containers"
info "Running: docker compose up -d"
$output = docker compose up -d 2>&1 | Out-String
Write-Host "   $output"
if ($LASTEXITCODE -ne 0) { fail "docker compose failed -- see output above" }

# -- 6. Wait for health --
header ">> Waiting for service to be ready"
$maxAttempts = 30
for ($i = 1; $i -le $maxAttempts; $i++) {
    try {
        $resp = Invoke-RestMethod -Uri "http://localhost:8000/health" -TimeoutSec 3
        if ($resp.status -eq "ok") {
            info "memory-agent is running! Memories stored: $($resp.memory_count)"
            break
        }
    } catch {
        # still starting
    }
    if ($i -eq $maxAttempts) {
        warn "Health check not responding after $maxAttempts seconds -- check 'docker compose logs app'"
    } else {
        Write-Host "   Waiting... ($i/$maxAttempts)" -NoNewline
        Start-Sleep -Seconds 2
    }
}

# -- 7. Harness configs --
header ">> Setup complete"
Write-Host "  Memory agent is running at http://localhost:8000" -ForegroundColor $GREEN
Write-Host "  Database: PostgreSQL 16 + pgvector on port 5433" -ForegroundColor $GREEN

Write-Host ""
Write-Host "  -- MCP Config ----------------------------------------" -ForegroundColor $CYAN
Write-Host "  Add to your AI tool's MCP settings to connect the memory agent."
Write-Host ""

Write-Host "  opencode.json / .mcp.json:" -ForegroundColor $YELLOW
Write-Host '  {'
Write-Host '    "mcpServers": {'
Write-Host '      "memory": {'
Write-Host '        "command": "docker",'
Write-Host '        "args": ["exec", "-i", "memory-agent-app", "python", "memory_mcp.py"]'
Write-Host '      }'
Write-Host '    }'
Write-Host '  }'

Write-Host ""
Write-Host "  Or use the HTTP API directly:" -ForegroundColor $YELLOW
Write-Host '  curl http://localhost:8000/health'
Write-Host '  curl -X POST http://localhost:8000/add -H "Content-Type: application/json" -d "{...}"'
Write-Host ""

Write-Host "  -- Quick test ----------------------------------------" -ForegroundColor $CYAN
Write-Host '  curl -s http://localhost:8000/health'
