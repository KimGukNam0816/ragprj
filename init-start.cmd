@echo off
REM init-start.cmd: first-time setup and start of ragprj (Windows cmd). Run from the project root: init-start
REM Re-runnable: finished steps are skipped, running servers are left alone. Does not write to Qdrant.
setlocal EnableExtensions
set "ROOT=%CD%"
if not exist "%ROOT%\pyproject.toml" (echo [FAIL] Run this from the ragprj project root. & exit /b 1)
if not exist "%ROOT%\scripts\check_integration.py" (echo [FAIL] Run this from the ragprj project root. & exit /b 1)
set "UV_SYNC=--locked --python 3.12 --extra memory --extra dev"
set "UV_PY=uv run %UV_SYNC% python -X utf8"
set "CHECK=%UV_PY% -m scripts.check_integration --stage"
REM Ports: set BACKEND_PORT / FRONTEND_PORT before running when 8000 or 3000 is taken by another program.
if not defined BACKEND_PORT set "BACKEND_PORT=8088"
if not defined FRONTEND_PORT set "FRONTEND_PORT=3033"
set "API=http://127.0.0.1:%BACKEND_PORT%"
set "WEB=http://127.0.0.1:%FRONTEND_PORT%"
REM The web server reads the API address from the process environment, not from .env.
set "BACKEND_URL=%API%"

echo === 1/5 Tools, uv and Python packages
call :need uv astral-sh.uv || exit /b 1
call :need npm OpenJS.NodeJS.LTS || exit /b 1
call :need ollama Ollama.Ollama || exit /b 1
uv python install 3.12 || exit /b 1
uv sync %UV_SYNC% || exit /b 1
uv sync %UV_SYNC% --check || (echo [FAIL] .venv does not match uv.lock & exit /b 1)
if not exist "%ROOT%\frontend\node_modules" (call npm --prefix "%ROOT%\frontend" ci || exit /b 1)

echo === 2/5 .env
if not exist "%ROOT%\.env" (
  copy "%ROOT%\.env.example" "%ROOT%\.env" >nul
  echo Fill in .env ^(SETUP.md section 4^), save, then run init-start again.
  start "" notepad "%ROOT%\.env"
  exit /b 1
)
%CHECK% env || exit /b 1

echo === 3/5 Ollama server, models and Cloud connections
curl -sf -o nul http://127.0.0.1:11434/api/tags || start "ollama" /min ollama serve
call :wait_url http://127.0.0.1:11434/api/tags 12 || (echo [FAIL] Ollama server did not start. & exit /b 1)
REM Pull only what is missing: pulling again can move a model to a newer digest that no longer matches the shared index.
for %%M in (bge-m3 qwen3-embedding:0.6b qwen3:4b) do (ollama show %%M >nul 2>nul || ollama pull %%M || exit /b 1)
%CHECK% connections || exit /b 1

echo === 4/5 Index pointers to shared-index ^(BM25 files committed in git^)
REM A pointer to an index built on this PC is kept; see shared-index\README.md.
%UV_PY% -m scripts.use_shared_index || exit /b 1
REM Read-only: the pointers, the shared Qdrant point count and this PC's embedding model digest must agree.
%CHECK% indexes || exit /b 1

echo === 5/5 Start API and web
REM shortcut: reuses an existing frontend build; delete frontend\.next after pulling new frontend code.
if not exist "%ROOT%\frontend\.next" (call npm --prefix "%ROOT%\frontend" run build || exit /b 1)
REM A port that answers is reused only when it is ragprj (its health JSON has local_only); anything else is a conflict.
curl -sf -m 5 %API%/health 2>nul | findstr /c:"local_only" >nul && (echo API already running on port %BACKEND_PORT%, reusing it. & goto :api_started)
curl -s -m 5 -o nul %API%/ && (echo [FAIL] Port %BACKEND_PORT% is used by another program. Run: set BACKEND_PORT=8010 ^& init-start & exit /b 1)
start "ragprj-api" /d "%ROOT%" cmd /k "%UV_PY% -m uvicorn backend.app.main:app --host 127.0.0.1 --port %BACKEND_PORT%"
:api_started
echo Waiting for the API ^(the first start downloads the reranker model, up to 10 minutes^) ...
call :wait_url %API%/health 120 || (echo [FAIL] API did not answer: see the "ragprj-api" window. & exit /b 1)
curl -sf -m 5 %WEB%/api/backend/health 2>nul | findstr /c:"local_only" >nul && (echo Web already running on port %FRONTEND_PORT%, reusing it. & goto :web_started)
curl -s -m 5 -o nul %WEB%/ && (echo [FAIL] Port %FRONTEND_PORT% is used by another program. Run: set FRONTEND_PORT=3010 ^& init-start & exit /b 1)
start "ragprj-web" /d "%ROOT%\frontend" cmd /k "npm run start -- --port %FRONTEND_PORT%"
:web_started
call :wait_url %WEB% 24 || (echo [FAIL] Web did not answer: see the "ragprj-web" window. & exit /b 1)
%CHECK% web --api-url %API% --web-url %WEB% || exit /b 1
start "" %WEB%
echo Done. Open %WEB% . Servers run in the "ragprj-api" and "ragprj-web" windows; close them to stop.
exit /b 0

:need
REM %1 command, %2 winget id. A tool installed earlier can be missing from this window's PATH, so try the default folders first.
where %1 >nul 2>nul && exit /b 0
set "PATH=%PATH%;%ProgramFiles%\nodejs;%LOCALAPPDATA%\Programs\Ollama;%LOCALAPPDATA%\Microsoft\WinGet\Links;%USERPROFILE%\.local\bin"
where %1 >nul 2>nul && exit /b 0
where winget >nul 2>nul || (echo [FAIL] %1 not found and winget is missing: install %2 by hand. & exit /b 1)
echo Installing %2 with winget ...
REM <nul: without it winget can wait forever for console input (seen in testing).
winget install --id %2 -e --silent --accept-package-agreements --accept-source-agreements <nul
where %1 >nul 2>nul && exit /b 0
echo [FAIL] %1 still not found after installing %2: open a new cmd and run init-start again.
exit /b 1

:wait_url
REM %1 url, %2 tries, 5 s apart.
for /l %%I in (1,1,%2) do (curl -sf -o nul %1 && exit /b 0 & ping -n 6 127.0.0.1 >nul)
exit /b 1
