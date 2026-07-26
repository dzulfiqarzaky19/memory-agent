<#
.SYNOPSIS
  One-command setup for memory-agent -- checks prereqs, configures .env, starts Docker.

.DESCRIPTION
  Runs on Windows (PowerShell 5.1+). Detects Docker, prompts for provider
  choice, writes .env, starts containers, and prints harness configs.

  Usage:
    .\scripts\setup.ps1
    .\scripts\setup.ps1 -Claude          # also install Claude Code auto-capture plugin
    .\scripts\setup.ps1 -ClaudeOnly      # skip docker/provider; only install plugin
#>

param(
    [switch]$Claude,
    [switch]$ClaudeOnly
)

$ErrorActionPreference = "Stop"
$REPO_DIR = Split-Path -Parent $PSScriptRoot
Set-Location $REPO_DIR

function Install-ClaudePlugin {
    header ">> Claude Code plugin (auto-capture)"
    $integrations = Join-Path $REPO_DIR "integrations"
    $pluginDir = Join-Path $integrations "claude-code"
    if (-not (Test-Path (Join-Path $pluginDir ".claude-plugin\plugin.json"))) {
        fail "Plugin missing at $pluginDir"
    }

    $claude = Get-Command claude -ErrorAction SilentlyContinue
    if (-not $claude) {
        warn "claude CLI not on PATH — wire manually:"
        Write-Host "  claude plugin marketplace add `"$integrations`""
        Write-Host "  claude plugin install memory-agent@memory-agent-integrations -s user"
        return
    }

    info "Validating plugin manifest"
    & claude plugin validate $pluginDir 2>&1 | ForEach-Object { Write-Host "   $_" }
    & claude plugin validate $integrations 2>&1 | ForEach-Object { Write-Host "   $_" }

    info "Adding local marketplace: $integrations"
    & claude plugin marketplace add $integrations 2>&1 | ForEach-Object { Write-Host "   $_" }

    info "Installing memory-agent@memory-agent-integrations (user scope)"
    & claude plugin install "memory-agent@memory-agent-integrations" -s user 2>&1 | ForEach-Object { Write-Host "   $_" }
    if ($LASTEXITCODE -ne 0) {
        warn "plugin install reported failure — try: claude plugin install memory-agent@memory-agent-integrations -s user"
    } else {
        info "Plugin installed. Restart Claude Code / open a new session to load hooks + MCP."
    }

    $uid = $env:MEMORY_AGENT_USER_ID
    if ([string]::IsNullOrWhiteSpace($uid)) { $uid = $env:USERNAME }
    Write-Host ""
    Write-Host "  Optional env (user profile / shell):" -ForegroundColor $CYAN
    Write-Host "    MEMORY_AGENT_URL=http://127.0.0.1:8000"
    Write-Host "    MEMORY_AGENT_USER_ID=$uid"
    Write-Host "    MEMORY_API_SECRET=<same as .env>  # or rely on ~/.memory-agent/api-secret"
    Write-Host "  Capture log: ~/.claude/hooks/logs/memory-auto-capture.jsonl"
}

if ($ClaudeOnly) {
    Install-ClaudePlugin
    exit 0
}

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

# Door: ensure MEMORY_API_SECRET in .env + host file for Stop hook (value never printed).
$header = ">> API door (localhost + shared secret)"
header $header
$ensureDoor = @'
import secrets, pathlib, os, re
root = pathlib.Path(".")
env_path = root / ".env"
text = env_path.read_text(encoding="utf-8") if env_path.exists() else ""
m = re.search(r"(?m)^MEMORY_API_SECRET=(.*)$", text)
secret = (m.group(1).strip().strip('"').strip("'") if m else "") or ""
if not secret:
    secret = secrets.token_urlsafe(32)
    line = f"MEMORY_API_SECRET={secret}\n"
    if m:
        text = re.sub(r"(?m)^MEMORY_API_SECRET=.*$", f"MEMORY_API_SECRET={secret}", text)
    else:
        if text and not text.endswith("\n"):
            text += "\n"
        text += "\n# HTTP door — clients send header X-Memory-Key\n" + line
        if "MEMORY_ALLOW_RELOAD=" not in text:
            text += "MEMORY_ALLOW_RELOAD=false\n"
    env_path.write_text(text, encoding="utf-8")
    print("generated")
else:
    print("exists")
home = pathlib.Path(os.path.expanduser("~")) / ".memory-agent"
home.mkdir(parents=True, exist_ok=True)
sec_file = home / "api-secret"
sec_file.write_text(secret + "\n", encoding="utf-8")
try:
    os.chmod(sec_file, 0o600)
except Exception:
    pass
print(str(sec_file))
'@
$doorOut = python -c $ensureDoor 2>&1
if ($LASTEXITCODE -ne 0) {
    warn "Could not ensure MEMORY_API_SECRET (python failed). Set it in .env manually."
} else {
    $doorLines = @($doorOut | ForEach-Object { "$_" })
    if ($doorLines[0] -eq "generated") { info "Generated MEMORY_API_SECRET in .env" }
    else { info "MEMORY_API_SECRET already set in .env" }
    if ($doorLines.Count -ge 2) { info "Host secret file: $($doorLines[1]) (Stop hook reads if env unset)" }
    info "HTTP published on 127.0.0.1 only; /reload off unless MEMORY_ALLOW_RELOAD=true"
}

# -- 3. Provider picker --
Write-Host ""
Write-Host "  Compose default embeddings: TEI MiniLM 384-d (no host embedder)." -ForegroundColor $CYAN
Write-Host "  Which LLM / embedding provider do you want to use?" -ForegroundColor $CYAN
Write-Host "    1) Compose defaults (TEI MiniLM 384 + LLM from .env / host) -- recommended"
Write-Host "    2) Local LLM override (LM Studio / Ollama) -- optional; set dims to match embed model"
Write-Host "    3) OpenAI API"
Write-Host "    4) Groq"
Write-Host "    5) Other OpenAI-compatible API"
Write-Host "    6) Skip -- I will edit .env myself"
$choice = Read-Host "  Enter 1-6 (default: 1)"
if ([string]::IsNullOrWhiteSpace($choice)) { $choice = "1" }

$writeEnv = $true
$apiKey = "not-needed"
$embUrl = $null
$llmUrl = $null
$model = $null
$embModel = $null
$embDims = $null

switch ($choice) {
    "1" {
        info "Keeping compose TEI MiniLM 384-d embeddings; only ensure LLM_* if present in .env"
        $writeEnv = $false
        info "docker compose overrides EMBEDDING_* to TEI (sentence-transformers/all-MiniLM-L6-v2, 384)."
        info "Edit .env LLM_BASE_URL / LLM_MODEL for your chat endpoint (host.docker.internal as needed)."
    }
    "2" {
        info "Configuring local LLM/embed override (not compose TEI default)"
        $llmUrl = "http://host.docker.internal:1234/v1"
        $embUrl = "http://host.docker.internal:1234/v1"
        $model = "google/gemma-4-e4b"
        $embModel = "text-embedding-nomic-embed-text-v1.5@q8_0"
        $embDims = "768"
        $apiKey = "not-needed"

        $localType = Read-Host "  Local type? (1) LM Studio / (2) Ollama"
        if ($localType -eq "2") {
            $llmUrl = "http://host.docker.internal:11434/v1"
            $embUrl = "http://host.docker.internal:11434/v1"
            $model = "llama3.2"
            $embModel = "nomic-embed-text"
            $embDims = "768"
        }
        warn "Compose still forces TEI unless you override app.environment EMBEDDING_* in docker-compose.override.yml"
    }
    "3" {
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
    "4" {
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
    "5" {
        info "Configuring custom OpenAI-compatible API"
        $llmUrl = Read-Host "  LLM API base URL (e.g. https://api.openai.com/v1)"
        $embUrl = Read-Host "  Embedding API base URL (same as above or different)"
        $apiKey = Read-Host "  API key"
        $model = Read-Host "  LLM model"
        $embModel = Read-Host "  Embedding model"
        $embDims = Read-Host "  Embedding dimensions"
    }
    default {
        $writeEnv = $false
        warn "Skipping auto-config -- edit .env manually then re-run"
    }
}

# -- 4. Write provider settings to .env --
if ($writeEnv -and $choice -ne "6") {
    $envContent = Get-Content ".env" -Raw

    # Match either legacy nomic 768 example lines or current MiniLM 384 defaults.
    $envContent = $envContent -replace "(?m)^EMBEDDING_MODEL=.*$", "EMBEDDING_MODEL=$embModel"
    $envContent = $envContent -replace "(?m)^EMBEDDING_DIMENSIONS=.*$", "EMBEDDING_DIMENSIONS=$embDims"
    if ($model) {
        $envContent = $envContent -replace "(?m)^LLM_MODEL=.*$", "LLM_MODEL=$model"
    }
    if ($apiKey -and $apiKey -ne "not-needed") {
        $envContent = $envContent -replace "(?m)^OPENAI_API_KEY=.*$", "OPENAI_API_KEY=$apiKey"
    }

    $envContent = $envContent -replace "(?m)^OPENAI_BASE_URL=.*`r?`n?", ""
    $envContent = $envContent -replace "(?m)^EMBEDDING_BASE_URL=.*`r?`n?", ""
    $envContent = $envContent -replace "(?m)^LLM_BASE_URL=.*`r?`n?", ""
    $envContent = $envContent -replace "(?m)^LLM_API_KEY=.*`r?`n?", ""

    $envContent = $envContent.TrimEnd("`r", "`n") + @"

# -- Provider URLs (set by setup script) --
EMBEDDING_BASE_URL=$embUrl
LLM_BASE_URL=$llmUrl
"@

    if ($apiKey -ne "not-needed" -and $apiKey -ne "") {
        $envContent = $envContent + "`r`nLLM_API_KEY=$apiKey"
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

# -- 7. Optional Claude Code plugin --
if ($Claude) {
    Install-ClaudePlugin
}

# -- 8. Harness configs --
header ">> Setup complete"
Write-Host "  Memory agent is running at http://localhost:8000" -ForegroundColor $GREEN
Write-Host "  Database: PostgreSQL 16 + pgvector on port 5433" -ForegroundColor $GREEN
Write-Host "  Auto-capture API: POST http://localhost:8000/capture" -ForegroundColor $GREEN

Write-Host ""
Write-Host "  -- Claude Code (easiest) ------------------------------" -ForegroundColor $CYAN
if (-not $Claude) {
    Write-Host "  Re-run with -Claude to install the auto-capture plugin:"
    Write-Host "    .\scripts\setup.ps1 -ClaudeOnly"
} else {
    Write-Host "  Plugin install attempted above. New session → turns auto-save to L0."
}
Write-Host ""
Write-Host "  -- MCP Config (other tools) ---------------------------" -ForegroundColor $CYAN
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
Write-Host "  -- Quick test ----------------------------------------" -ForegroundColor $CYAN
Write-Host '  curl -s http://localhost:8000/health'
Write-Host '  curl -s -X POST http://localhost:8000/capture -H "Content-Type: application/json" -d "{\"user_id\":\"demo\",\"session_key\":\"s1\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"},{\"role\":\"assistant\",\"content\":\"hey\"}]}"'
