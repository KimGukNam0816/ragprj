# 웹에 서비스 띄우기

이 프로젝트는 **생성·임베딩 추론을 전부 로컬 Ollama로** 합니다. 그래서 "웹에 띄운다"는 말이
두 가지로 갈립니다. 먼저 둘 중 어느 쪽인지 정하고 아래 해당 절만 보세요.

| | 방법 A · 터널 | 방법 B · Docker 서버 | 방법 C · 프런트만 클라우드 |
|---|---|---|---|
| 공개 주소 | `https://xxx.trycloudflare.com` | 내 서버 도메인 | `https://xxx.vercel.app` |
| 추론 위치 | 내 RTX 5060 PC | 그 서버 (GPU 없으면 매우 느림) | 내 PC (터널 경유) |
| 비용 | 무료 | 서버 비용 | 무료 ~ |
| PC를 꺼도 되나 | ✗ | ✓ | ✗ |
| 준비 시간 | 5분 | 30분~ | 20분 |
| 발표·데모용 추천 | **◎** | △ | ○ |

> GPU가 필요한 구조라 **방법 A가 거의 항상 정답**입니다. 클라우드 GPU 없이 B로 가면
> `qwen3:4b` 생성이 CPU에서 수십 초~수 분 걸립니다.

---

## 0. 공통 전제

로컬에서 README 0~5단계가 끝나 있어야 합니다. 아직이면 한 번에 처리하는 스크립트를 쓰세요.

```powershell
# Windows (프로젝트 루트에서). 처음이면 -Bootstrap 으로 청킹·색인까지 한 번에.
powershell -ExecutionPolicy Bypass -File scripts\serve.ps1 -Bootstrap
```

```bash
# Linux / macOS / WSL
./scripts/serve.sh --bootstrap
```

스크립트가 하는 일: 도구 확인 → `.env` 생성 → `uv sync` → `npm ci` → Ollama 모델 3종 pull →
(필요하면) AS007 청킹·두 프로필 색인 → `next build` → FastAPI + Next.js 실행.
중간에 실패하면 **무엇을 고쳐야 하는지 한 줄로 알려 주고 멈춥니다.**

---

## 방법 A. 내 PC를 그대로 공개하기 (Cloudflare Tunnel) — 추천

GPU가 있는 PC에서 서비스를 돌리고, 바깥에서 접속할 수 있는 https 주소만 빌려 옵니다.
공유기 포트포워딩, 고정 IP, 인증서 모두 필요 없습니다.

### A-1. cloudflared 설치

```powershell
winget install --id Cloudflare.cloudflared -e
```

```bash
# macOS
brew install cloudflared
# Linux (deb)
curl -fsSL https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb -o cloudflared.deb && sudo dpkg -i cloudflared.deb
```

### A-2. 접근 토큰 정하기 (공개 전 필수)

주소가 추측하기 어렵긴 해도 **링크를 아는 사람은 누구나 내 PC의 LLM을 호출**할 수 있습니다.
`.env`에 토큰을 넣으면 `/query`, `/memory`가 토큰 없이는 401을 돌려줍니다.

```dotenv
PUBLIC_ACCESS_TOKEN=아무도-모를-긴-문자열
```

Next.js 서버에 같은 값을 넘기면, 브라우저에는 토큰이 내려가지 않고 서버끼리만 주고받습니다.

```powershell
$env:BACKEND_ACCESS_TOKEN = '아무도-모를-긴-문자열'
```

> 데모 시간에만 잠깐 열 거라면 토큰 없이 띄우고 끝나면 Ctrl+C로 내려도 됩니다.
> 다만 **그 사이에는 누구나 호출할 수 있다**는 점은 알고 하세요.

### A-3. 실행

```powershell
powershell -ExecutionPolicy Bypass -File scripts\serve.ps1 -Public
```

```bash
./scripts/serve.sh --public
```

로그에 다음과 같은 줄이 나옵니다. 이 주소가 공개 URL입니다.

```text
+--------------------------------------------------------------------+
|  https://pleasant-stone-xxxx.trycloudflare.com                      |
+--------------------------------------------------------------------+
```

직접 띄우고 싶으면 서버를 켠 상태에서 터널만 따로 실행해도 됩니다.

```bash
cloudflared tunnel --url http://127.0.0.1:3000
```

### A-4. 고정 주소로 쓰기

`trycloudflare.com` 주소는 실행할 때마다 바뀝니다. 발표 자료에 넣을 고정 주소가 필요하면
Cloudflare 계정에 도메인을 연결하고 named tunnel을 만드세요.

```bash
cloudflared tunnel login
cloudflared tunnel create ragprj
cloudflared tunnel route dns ragprj chat.내도메인.com
cloudflared tunnel run --url http://127.0.0.1:3000 ragprj
```

---

## 방법 B. Docker Compose로 서버에 올리기

Qdrant + FastAPI + Next.js를 컨테이너 3개로 띄웁니다. **Ollama는 compose 밖**에서 돌립니다
(GPU를 컨테이너에 넘기는 설정이 환경마다 달라서, 호스트의 Ollama를 가리키는 쪽이 안전합니다).

```bash
cp .env.example .env     # QDRANT_URL은 compose가 덮어쓰므로 비워 둬도 됩니다
docker compose up -d --build
docker compose logs -f backend
```

- 채팅: <http://localhost:3000>
- API 문서: <http://localhost:8000/docs>
- Qdrant: <http://localhost:6333/dashboard>

### 색인 만들기

컨테이너의 `runtime/`은 호스트의 `./runtime`과 공유됩니다. 색인은 호스트에서 만들되,
Qdrant 주소만 compose의 Qdrant로 맞춰 주세요.

```bash
QDRANT_URL=http://127.0.0.1:6333 QDRANT_API_KEY= \
  uv run python -m scripts.01_chunk_documents --books AS007
QDRANT_URL=http://127.0.0.1:6333 QDRANT_API_KEY= \
  uv run python -m scripts.02_build_indexes --corpus <CORPUS_DIR> --embedding bge
QDRANT_URL=http://127.0.0.1:6333 QDRANT_API_KEY= \
  uv run python -m scripts.02_build_indexes --corpus <CORPUS_DIR> --embedding qwen
docker compose restart backend
```

### 리랭커(hybrid_rerank)까지 쓰려면

기본 이미지에는 Torch가 빠져 있어 `naive`/`hybrid`만 동작합니다. compose가
`RERANKER_WARMUP_FAIL_FAST=false`를 넣어 주므로 서버는 정상적으로 뜨고,
`/status`의 `reranker`에 이유가 표시됩니다. CPU에서라도 리랭커를 쓰려면:

```bash
WITH_RERANKER=1 docker compose up -d --build backend
```

CPU 리랭킹은 질문당 수십 초가 걸릴 수 있습니다. 실사용은 GPU 머신에서 하세요.

### 공개 서버로 쓸 때

`docker-compose.yml`의 `frontend` 포트는 `3000:3000`이라 외부에 열립니다.
그 앞에 Caddy/Nginx로 HTTPS를 붙이고, `.env`에 다음을 설정하세요.

```dotenv
PUBLIC_ACCESS_TOKEN=긴-토큰
ALLOWED_ORIGINS=https://chat.내도메인.com
```

---

## 방법 C. 프런트엔드만 Vercel, 백엔드는 내 PC

채팅 화면만 Vercel에 올리고, API는 터널로 연결된 내 PC가 처리합니다.

1. 내 PC에서 백엔드만 공개합니다. `.env`에 `PUBLIC_ACCESS_TOKEN`을 **반드시** 넣으세요.
   ```bash
   uv run uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
   cloudflared tunnel --url http://127.0.0.1:8000
   ```
2. Vercel에 `frontend/` 폴더를 Root Directory로 지정해 배포합니다.
3. Vercel 환경변수에 넣습니다.
   ```text
   BACKEND_URL=https://백엔드-터널-주소
   BACKEND_ACCESS_TOKEN=.env에 넣은 같은 토큰
   ```

`/api/backend/*` 프록시가 서버 사이드에서 호출하므로 토큰과 백엔드 주소는 브라우저에 노출되지 않습니다.

---

## 공개 전 점검표

- [ ] `PUBLIC_ACCESS_TOKEN`을 설정했다 (또는 공개 시간이 짧고 그 위험을 알고 있다)
- [ ] `.env`를 커밋하지 않았다 — `git status`로 확인
- [ ] Qdrant/Langfuse 키가 프런트엔드 환경변수에 들어가지 않았다
- [ ] Ollama(11434)와 Qdrant(6333) 포트를 인터넷에 직접 열지 않았다
- [ ] `ALLOWED_ORIGINS`에 실제 공개 도메인을 넣었다
- [ ] 데모가 끝나면 터널과 서버를 내렸다

## 관련 설정

| 변수 | 기본값 | 설명 |
|---|---|---|
| `PUBLIC_ACCESS_TOKEN` | 없음 | 값이 있으면 `/query`·`/memory`에 `X-Access-Token` 요구 |
| `ALLOWED_ORIGINS` | 없음 | 쉼표로 구분한 추가 CORS origin |
| `RERANKER_WARMUP_ON_STARTUP` | `true` | 시작할 때 리랭커 미리 적재 |
| `RERANKER_WARMUP_FAIL_FAST` | `true` | 리랭커 준비 실패 시 서버도 안 뜸. 웹 배포에서는 `false` |
| `BACKEND_URL` (프런트) | `http://127.0.0.1:8000` | Next.js가 호출할 백엔드 주소 |
| `BACKEND_ACCESS_TOKEN` (프런트) | 없음 | Next.js 서버가 붙여 보낼 토큰 |

## 자주 걸리는 곳

| 증상 | 원인과 조치 |
|---|---|
| 공개 주소는 열리는데 질문하면 503 | 백엔드가 안 떴거나 `BACKEND_URL`이 틀렸다. `/api/backend/health` 확인 |
| 질문하면 401 | 프런트에 `BACKEND_ACCESS_TOKEN`이 없거나 백엔드 토큰과 다르다 |
| `index_missing` | 해당 `RUNTIME_DIR`에 두 프로필 색인이 없다. `serve.ps1 -Bootstrap` |
| 서버가 시작하다 죽음 | 리랭커 적재(VRAM·HF 다운로드) 실패. `RERANKER_WARMUP_FAIL_FAST=false`로 두면 경고만 남기고 뜬다. `/status`의 `reranker` 확인 |
| 첫 질문이 2분 넘게 걸림 | Ollama가 모델을 적재 중. `LOW_VRAM_MODE=true`면 매번 내렸다 올린다 |
| `local_model_missing` | `ollama pull qwen3:4b` 후 `ollama list`로 이름 일치 확인 |
