<#
.SYNOPSIS
  설치 → 색인 확인 → 백엔드 → 프런트엔드까지 한 번에 띄운다.

.EXAMPLE
  # 처음 한 번 (AS007 한 문서로 색인까지 만든다)
  powershell -ExecutionPolicy Bypass -File scripts\serve.ps1 -Bootstrap

  # 평소 실행
  powershell -ExecutionPolicy Bypass -File scripts\serve.ps1

  # 인터넷에 공개 (cloudflared 필요)
  powershell -ExecutionPolicy Bypass -File scripts\serve.ps1 -Public
#>
[CmdletBinding()]
param(
  [switch]$Bootstrap,          # 청킹 + 두 임베딩 색인까지 실행
  [switch]$Full,               # -Bootstrap 과 함께: 120개 전체 문서로 색인 (오래 걸림)
  [switch]$Public,             # cloudflared 로 공개 URL 발급
  [int]$BackendPort = 8000,
  [int]$FrontendPort = 3000
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

function Step($text) { Write-Host "`n== $text" -ForegroundColor Cyan }
function Ok($text)   { Write-Host "   OK  $text" -ForegroundColor Green }
function Die($text)  { Write-Host "   실패: $text" -ForegroundColor Red; exit 1 }

# ---------------------------------------------------------------- 1. 도구
Step '필수 도구 확인'
foreach ($cmd in 'uv','node','npm','ollama') {
  if (-not (Get-Command $cmd -ErrorAction SilentlyContinue)) {
    Die "$cmd 가 없습니다. SETUP.md 0단계의 winget 명령으로 설치한 뒤 PowerShell을 새로 여세요."
  }
}
Ok 'uv / node / npm / ollama'

# ---------------------------------------------------------------- 2. .env
Step '.env 확인'
if (-not (Test-Path .env)) {
  Copy-Item .env.example .env
  Write-Host '   .env 를 새로 만들었습니다. QDRANT_URL / QDRANT_API_KEY / LANGFUSE_* 를 채운 뒤 다시 실행하세요.' -ForegroundColor Yellow
  Write-Host '   Qdrant를 로컬 도커로 쓸 거면 QDRANT_URL=http://127.0.0.1:6333 그대로 두면 됩니다.' -ForegroundColor Yellow
  exit 1
}
Ok '.env'

# ---------------------------------------------------------------- 3. 의존성
Step 'Python 의존성 설치 (uv sync)'
uv sync --locked --extra memory --extra dev
if ($LASTEXITCODE -ne 0) { Die 'uv sync 실패' }
Ok 'python packages'

Step '프런트엔드 의존성 확인'
if (-not (Test-Path frontend/node_modules)) {
  npm.cmd --prefix frontend ci
  if ($LASTEXITCODE -ne 0) { Die 'npm ci 실패 (네트워크/레지스트리 확인)' }
}
Ok 'node_modules'

# ---------------------------------------------------------------- 4. Ollama
Step 'Ollama 서버와 모델 확인'
try { $null = Invoke-RestMethod "http://127.0.0.1:11434/api/tags" -TimeoutSec 5 }
catch { Die 'Ollama 서버가 응답하지 않습니다. Ollama 앱을 실행하거나 별도 창에서 ollama serve 를 켜세요.' }

$installed = (ollama list | Select-Object -Skip 1 | ForEach-Object { ($_ -split '\s+')[0] })
foreach ($model in 'bge-m3','qwen3-embedding:0.6b','qwen3:4b') {
  if ($installed -notcontains $model -and $installed -notcontains "${model}:latest") {
    Write-Host "   $model 내려받는 중..." -ForegroundColor Yellow
    ollama pull $model
    if ($LASTEXITCODE -ne 0) { Die "ollama pull $model 실패" }
  }
}
Ok '모델 3종'

# ---------------------------------------------------------------- 5. 색인
Step '색인 확인'
$runtime = (uv run python -c "from backend.app.config import get_settings;print(get_settings().runtime_dir.resolve())").Trim()
$bge  = Join-Path $runtime 'indexes/active_bge.json'
$qwen = Join-Path $runtime 'indexes/active_qwen.json'

if ($Bootstrap -or -not (Test-Path $bge) -or -not (Test-Path $qwen)) {
  if (-not $Bootstrap) {
    Die "색인이 없습니다. 처음이라면 -Bootstrap 을 붙여 다시 실행하세요:`n       powershell -ExecutionPolicy Bypass -File scripts\serve.ps1 -Bootstrap"
  }
  Write-Host '   청킹 시작 (문서 수에 따라 수 분~수 시간)' -ForegroundColor Yellow
  $chunkArgs = if ($Full) { @() } else { @('--books','AS007') }
  uv run python -m scripts.01_chunk_documents @chunkArgs | Tee-Object -Variable lines
  if ($LASTEXITCODE -ne 0) { Die '01_chunk_documents 실패' }
  $corpusLine = $lines | Select-String '^CORPUS_DIR=' | Select-Object -Last 1
  $corpus = if ($corpusLine) { $corpusLine.Line -replace '^CORPUS_DIR=', '' } else { '' }
  if (-not $corpus) { Die '청킹 결과에서 CORPUS_DIR 경로를 찾지 못했습니다.' }
  Ok "corpus: $corpus"

  foreach ($embeddingProfile in 'bge','qwen') {
    Write-Host "   색인 생성: $embeddingProfile" -ForegroundColor Yellow
    uv run python -m scripts.02_build_indexes --corpus "$corpus" --embedding $embeddingProfile
    if ($LASTEXITCODE -ne 0) { Die "02_build_indexes --embedding $embeddingProfile 실패" }
  }
}
Ok '색인'

# ---------------------------------------------------------------- 6. 빌드
Step 'Next.js 빌드'
npm.cmd --prefix frontend run build
if ($LASTEXITCODE -ne 0) { Die 'next build 실패' }
Ok 'build'

# ---------------------------------------------------------------- 7. 실행
Step '서버 시작'
$backend = Start-Process -PassThru -NoNewWindow uv `
  -ArgumentList @('run','uvicorn','backend.app.main:app','--host','127.0.0.1','--port',"$BackendPort")

Write-Host '   백엔드 준비 대기 중...' -ForegroundColor Yellow
$ready = $false
foreach ($i in 1..90) {
  Start-Sleep -Seconds 2
  try { $null = Invoke-RestMethod "http://127.0.0.1:$BackendPort/health" -TimeoutSec 3; $ready = $true; break } catch { }
}
if (-not $ready) { Die "백엔드가 뜨지 않았습니다. 위 로그를 확인하세요." }
Ok "backend  http://127.0.0.1:$BackendPort/docs"

$env:BACKEND_URL = "http://127.0.0.1:$BackendPort"
$env:PORT = "$FrontendPort"
$frontend = Start-Process -PassThru -NoNewWindow npm.cmd `
  -ArgumentList @('--prefix','frontend','run', $(if ($Public) {'start:public'} else {'start'}))
Ok "frontend http://127.0.0.1:$FrontendPort"

$tunnel = $null
if ($Public) {
  Step '공개 URL 발급 (cloudflared)'
  if (-not (Get-Command cloudflared -ErrorAction SilentlyContinue)) {
    Write-Host '   cloudflared 가 없습니다: winget install --id Cloudflare.cloudflared -e' -ForegroundColor Yellow
  } else {
    $tunnel = Start-Process -PassThru -NoNewWindow cloudflared `
      -ArgumentList @('tunnel','--url',"http://127.0.0.1:$FrontendPort")
    Write-Host '   위 cloudflared 로그에 나오는 https://....trycloudflare.com 주소가 공개 URL입니다.' -ForegroundColor Green
  }
}

Write-Host "`n준비 완료. 종료하려면 Ctrl+C." -ForegroundColor Cyan
try { Wait-Process -Id $frontend.Id }
finally {
  foreach ($p in $backend, $frontend, $tunnel) {
    if ($p -and -not $p.HasExited) { Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue }
  }
}
