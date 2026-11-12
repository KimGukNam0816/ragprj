#!/usr/bin/env bash
# 설치 → 색인 확인 → 백엔드 → 프런트엔드까지 한 번에 띄운다. (Linux / macOS / WSL)
#
#   처음 한 번:  ./scripts/serve.sh --bootstrap
#   평소 실행:    ./scripts/serve.sh
#   공개 URL:     ./scripts/serve.sh --public      (cloudflared 필요)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

BOOTSTRAP=0; FULL=0; PUBLIC=0; BACKEND_PORT=8000; FRONTEND_PORT=3000
while [ $# -gt 0 ]; do
  case "$1" in
    --bootstrap) BOOTSTRAP=1 ;;
    --full) FULL=1 ;;
    --public) PUBLIC=1 ;;
    --backend-port) BACKEND_PORT="$2"; shift ;;
    --frontend-port) FRONTEND_PORT="$2"; shift ;;
    *) echo "알 수 없는 옵션: $1" >&2; exit 2 ;;
  esac
  shift
done

step() { printf '\n== %s\n' "$1"; }
ok()   { printf '   OK  %s\n' "$1"; }
die()  { printf '   실패: %s\n' "$1" >&2; exit 1; }

step '필수 도구 확인'
for cmd in uv node npm ollama; do
  command -v "$cmd" >/dev/null 2>&1 || die "$cmd 가 없습니다. SETUP.md 0단계를 먼저 진행하세요."
done
ok 'uv / node / npm / ollama'

step '.env 확인'
if [ ! -f .env ]; then
  cp .env.example .env
  echo '   .env 를 새로 만들었습니다. QDRANT_URL / QDRANT_API_KEY / LANGFUSE_* 를 채운 뒤 다시 실행하세요.'
  exit 1
fi
ok '.env'

step 'Python 의존성 설치 (uv sync)'
uv sync --locked --extra memory --extra dev \
  || die 'uv sync 실패. 네트워크(특히 download.pytorch.org)와 디스크 여유 공간을 확인하세요.'
ok 'python packages'

step '프런트엔드 의존성 확인'
[ -d frontend/node_modules ] || npm --prefix frontend ci \
  || die 'npm ci 실패. 네트워크와 Node 버전(24 LTS 권장)을 확인하세요.'
ok 'node_modules'

step 'Ollama 서버와 모델 확인'
curl -fsS http://127.0.0.1:11434/api/tags >/dev/null 2>&1 \
  || die 'Ollama 서버가 응답하지 않습니다. ollama serve 를 먼저 실행하세요.'
installed="$(ollama list | tail -n +2 | awk '{print $1}')"
for model in bge-m3 qwen3-embedding:0.6b qwen3:4b; do
  if ! grep -qx -e "$model" -e "$model:latest" <<<"$installed"; then
    echo "   $model 내려받는 중..."
    ollama pull "$model"
  fi
done
ok '모델 3종'

step '색인 확인'
RUNTIME="$(uv run python -c 'from backend.app.config import get_settings;print(get_settings().runtime_dir.resolve())')"
if [ "$BOOTSTRAP" = 1 ] || [ ! -f "$RUNTIME/indexes/active_bge.json" ] || [ ! -f "$RUNTIME/indexes/active_qwen.json" ]; then
  [ "$BOOTSTRAP" = 1 ] || die "색인이 없습니다. 처음이라면 ./scripts/serve.sh --bootstrap 으로 실행하세요."
  echo '   청킹 시작 (문서 수에 따라 수 분~수 시간)'
  if [ "$FULL" = 1 ]; then
    CORPUS="$(uv run python -m scripts.01_chunk_documents | tee /dev/stderr | sed -n 's/^CORPUS_DIR=//p' | tail -1)"
  else
    CORPUS="$(uv run python -m scripts.01_chunk_documents --books AS007 | tee /dev/stderr | sed -n 's/^CORPUS_DIR=//p' | tail -1)"
  fi
  [ -n "$CORPUS" ] || die '청킹 결과에서 CORPUS_DIR 경로를 찾지 못했습니다.'
  ok "corpus: $CORPUS"
  for profile in bge qwen; do
    echo "   색인 생성: $profile"
    uv run python -m scripts.02_build_indexes --corpus "$CORPUS" --embedding "$profile"
  done
fi
ok '색인'

step 'Next.js 빌드'
npm --prefix frontend run build || die 'next build 실패. 위 TypeScript 오류를 확인하세요.'
ok 'build'

step '서버 시작'
BIND=127.0.0.1
[ "$PUBLIC" = 1 ] && BIND=127.0.0.1   # 공개는 터널이 담당한다. 서버 자체는 로컬에만 바인딩한다.
uv run uvicorn backend.app.main:app --host "$BIND" --port "$BACKEND_PORT" &
BACKEND_PID=$!

cleanup() { kill "$BACKEND_PID" "${FRONTEND_PID:-}" "${TUNNEL_PID:-}" 2>/dev/null || true; }
trap cleanup EXIT INT TERM

echo '   백엔드 준비 대기 중...'
for _ in $(seq 1 90); do
  sleep 2
  curl -fsS "http://127.0.0.1:$BACKEND_PORT/health" >/dev/null 2>&1 && break
done
curl -fsS "http://127.0.0.1:$BACKEND_PORT/health" >/dev/null 2>&1 \
  || die '백엔드가 뜨지 않았습니다. 위 로그를 확인하세요.'
ok "backend  http://127.0.0.1:$BACKEND_PORT/docs"

BACKEND_URL="http://127.0.0.1:$BACKEND_PORT" PORT="$FRONTEND_PORT" npm --prefix frontend run start &
FRONTEND_PID=$!
ok "frontend http://127.0.0.1:$FRONTEND_PORT"

if [ "$PUBLIC" = 1 ]; then
  step '공개 URL 발급 (cloudflared)'
  if command -v cloudflared >/dev/null 2>&1; then
    cloudflared tunnel --url "http://127.0.0.1:$FRONTEND_PORT" &
    TUNNEL_PID=$!
    echo '   위 cloudflared 로그의 https://....trycloudflare.com 주소가 공개 URL입니다.'
  else
    echo '   cloudflared 가 없습니다. docs/DEPLOY.md 설치 안내를 참고하세요.'
  fi
fi

printf '\n준비 완료. 종료하려면 Ctrl+C.\n'
wait "$FRONTEND_PID"
