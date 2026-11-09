# 다른 PC 설치 및 실행 가이드

대상: **Windows 11 + PowerShell + RTX 5060 8GB**. 모든 생성·최종 답변·평가·Mem0 추론은 **Ollama 로컬 모델**을 사용합니다. Qdrant Cloud와 Langfuse Cloud 연결 및 최초 모델/패키지 다운로드에는 인터넷이 필요합니다.

공식 설치 참고: [uv](https://docs.astral.sh/uv/getting-started/installation/), [Ollama Windows](https://github.com/ollama/ollama/blob/main/docs/windows.mdx).

각 코드 블록을 순서대로 실행합니다. 실패하면 다음 단계로 넘어가지 마세요. PowerShell은 명령 실패 후에도 다음 줄을 실행할 수 있습니다. `$LASTEXITCODE`가 `0`인지 확인하세요.

> 현재 앱은 개발 검증 버전입니다. 이 문서는 새 PC의 설치·연동 절차이며 5060 정식 품질 평가 완료를 의미하지 않습니다. 기존 360응답의 추가 채점도 자동 재개하지 않습니다.

## 0. 준비물

- NVIDIA GPU 드라이버, Git for Windows, Node.js **24 LTS**, Ollama, `uv`
- 원본 `data/advanced/Advanced_shipping_part01of09` ~ `part09of09` 폴더
- 본인 Qdrant Cloud URL/API 키, Langfuse Cloud 리전 URL/Public Key/Secret Key
- Python은 아래 `uv python install 3.12`로 설치합니다. 별도 CUDA Toolkit은 필수가 아니며 PyTorch CUDA wheel과 호환 NVIDIA 드라이버가 필요합니다.

도구가 없다면 아래 공식 설치 경로를 사용하고 **PowerShell을 새로 여세요**.

```powershell
winget install --id Git.Git -e
winget install --id OpenJS.NodeJS.LTS -e
winget install --id Ollama.Ollama -e
winget install --id astral-sh.uv -e
```

설치 확인:

```powershell
git --version
node --version
npm.cmd --version
uv --version
ollama --version
nvidia-smi
```

`node --version`은 v24 계열을 권장합니다. Winget의 최신 LTS가 다른 계열이면 [Node.js 공식 사이트](https://nodejs.org/en/download)에서 24 LTS를 선택하세요. `nvidia-smi`가 실패하면 드라이버부터 해결합니다.

## 1. Git에서 프로젝트 받기

```powershell
New-Item -ItemType Directory -Force C:\project | Out-Null
Set-Location C:\project
git clone --branch main https://github.com/KimGukNam0816/ragprj.git ragdev
Set-Location C:\project\ragdev
git status --short --branch
```

이미 clone했다면 새 clone 대신 아래를 실행합니다. 로컬 수정이 있으면 먼저 확인하고, 강제 reset하지 마세요.

```powershell
Set-Location C:\project\ragdev
git pull --ff-only origin main
```

이후 Python 명령은 모두 **프로젝트 루트**에서 실행합니다. 저장소가 비공개면 Git Credential Manager 로그인 창에서 인증합니다.

## 2. Python과 프런트 의존성 설치

```powershell
uv python install 3.12
uv sync --locked --python 3.12 --extra memory --extra dev
.\.venv\Scripts\python.exe -X utf8 -m scripts.check_integration --stage tools
```

프런트 설치:

```powershell
npm.cmd --prefix frontend ci
```

`frontend/.npmrc`에 다운로드 대기 시간 10분, 재시도 3회, 동시 연결 3개 및 캐시 우선 사용을 설정했습니다. 위 명령에도 자동 적용되어 느린 네트워크의 다운로드 시간 초과를 줄입니다. 네트워크 단절이나 레지스트리 장애가 계속되면 실패할 수 있습니다.

CUDA 설치 확인:

```powershell
.\.venv\Scripts\python.exe -c "import torch; print('torch:',torch.__version__); print('CUDA:',torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CUDA unavailable')"
```

기대값: `CUDA: True`와 RTX 5060. `uv.lock`과 `pyproject.toml`은 PyTorch CUDA 12.8 인덱스를 지정합니다. CPU wheel을 별도로 덮어 설치하지 마세요. 가상환경 activate나 PowerShell 실행 정책 변경은 필요 없습니다.

## 3. 원본 데이터 별도 복사

`data/`는 Git에서 제외되어 있습니다. 외장 디스크 등을 통해 아래 구조로 복사합니다. **part별 폴더는 합치거나 이름을 바꾸지 않습니다.**

```text
C:\project\ragdev\data\advanced\
  Advanced_shipping_part01of09\
    documents\AS001.pdf ...
    evaluation\development_questions.json
               additional_questions.json
               development.json
               additional.json
               security_tasks.json
  Advanced_shipping_part02of09\documents\...
  ...
  Advanced_shipping_part09of09\documents\AS120.pdf
```

총 AS001~AS120 **120개**가 필요합니다. `--books AS007`로 빠른 청킹을 하더라도 원본 목록 검사는 120개를 요구합니다. Basic/Intermediate는 필요 없습니다. 평가 파일은 part01의 한 세트를 사용합니다.

## 4. .env 작성 - 5060 연동검증용 설정

기존 `.env`가 없을 때만 복사합니다. 키가 입력된 기존 파일을 덮어쓰지 않습니다.

```powershell
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
notepad .env
```

아래 항목은 `.env`의 **기존 같은 이름의 줄을 수정**합니다. 중복으로 붙이지 마세요.

```dotenv
DATA_DIR=./data
EVALUATION_DIR=./data/advanced/Advanced_shipping_part01of09/evaluation
RUNTIME_DIR=./runtime/5060-check

EXECUTION_STAGE=local_smoke
FORCE_LOCAL_ONLY=true
ALLOW_EXTERNAL_LLM=false
GENERATION_LLM_PROVIDER=ollama
EVALUATION_LLM_PROVIDER=ollama
MEM0_LLM_PROVIDER=ollama
LOCAL_LLM_MODEL=qwen3:4b
LOCAL_EVALUATION_MODEL=qwen3:4b
LOCAL_MEM0_MODEL=qwen3:4b
OLLAMA_BASE_URL=http://127.0.0.1:11434

SMOKE_RUN=true
PUBLISH_METRICS=false
METRICS_DEVICE=rtx5060
LOW_VRAM_MODE=true
EMBEDDING_BATCH_SIZE=1
EMBEDDING_CONTEXT_TOKENS=2048
LOCAL_CONTEXT_TOKENS=8192
CONTEXT_MAX_TOKENS=6000
LLM_MAX_OUTPUT_TOKENS=2048

EMBEDDING_MODEL=bge-m3
EMBEDDING_DIMENSIONS=1024
COMPARISON_EMBEDDING_MODEL=qwen3-embedding:0.6b
COMPARISON_EMBEDDING_DIMENSIONS=1024
RERANKER_MODEL=Qwen/Qwen3-Reranker-0.6B
RERANKER_DEVICE=cuda
RERANKER_BATCH_SIZE=1
RERANKER_MAX_LENGTH=1024
DENSE_CANDIDATE_K=20
BM25_CANDIDATE_K=20
RRF_K=60
RERANK_CANDIDATE_K=20
FINAL_TOP_K=4

QDRANT_URL=https://실제-클러스터-주소:6333
QDRANT_API_KEY=본인-키
QDRANT_DOCS_BGE_COLLECTION=check5060_docs_bge_v1
QDRANT_DOCS_QWEN_COLLECTION=check5060_docs_qwen06_v1
QDRANT_MEMORY_COLLECTION=check5060_mem_users_bge_v1
MEM0_HISTORY_DB_PATH=./runtime/5060-check/mem0/history.db

LANGFUSE_ENABLED=true
LANGFUSE_BASE_URL=https://본인-리전-Langfuse-주소
LANGFUSE_PUBLIC_KEY=본인-public-key
LANGFUSE_SECRET_KEY=본인-secret-key
```

위 Cloud 주소/키 예시는 실행 가능한 값이 아닙니다. 각 서비스 콘솔의 **정확한 URL과 본인 키**로 대체하세요. OPENAI_*는 비워 두며 사용하지 않습니다. HF_TOKEN은 공개 모델 다운로드에는 보통 필요 없습니다.

5060 8GB에 맞춰 문맥·리랭커 길이를 제한한 초기 설정입니다. **정식 비교 조건은 아직 아닙니다.** OOM이 나면 병렬 요청과 대량 평가를 멈추고 적재 상태부터 확인합니다. 이후 임베딩 모델/digest/context를 변경하면 새 색인이 필요합니다. 실행 중 `.env`를 바꾸면 서버도 재시작해야 합니다.

```powershell
.\.venv\Scripts\python.exe -X utf8 -m scripts.check_integration --stage env
.\.venv\Scripts\python.exe -X utf8 -m scripts.check_integration --stage data
```

## 5. Ollama 모델 설치 및 실제 연결 점검

Ollama 앱을 실행해 둡니다. 서버가 이미 실행 중이면 `ollama serve`를 중복 실행하지 않습니다. 서버가 없을 때만 별도 창에서 `ollama serve`를 실행하세요.

```powershell
ollama pull bge-m3
ollama pull qwen3-embedding:0.6b
ollama pull qwen3:4b
ollama list
.\.venv\Scripts\python.exe -X utf8 -m scripts.check_integration --stage connections
```

연결 점검은 Ollama/Qdrant/Langfuse 인증, 두 임베딩 실제 실행, 짧은 로컬 JSON 생성까지 확인합니다. 문서 색인은 만들지 않습니다.

## 6. 먼저 AS007 한 문서 청킹

```powershell
.\.venv\Scripts\python.exe -X utf8 -m scripts.01_chunk_documents --books AS007
```

마지막 `CORPUS_DIR=...`에 출력된 경로를 아래 변수에 넣습니다. 해시 부분은 모델 tokenizer 버전에 따라 달라질 수 있으므로 예전 PC 경로를 사용하지 마세요.

```powershell
$Corpus = '여기에 CORPUS_DIR= 뒤의 실제 경로를 붙여넣기'
.\.venv\Scripts\python.exe -X utf8 -m scripts.check_integration --stage corpus --corpus $Corpus
```

기대값: scope=subset, document_count=1, failure_count=0. 이 PowerShell 창을 닫으면 `$Corpus`를 다시 지정해야 합니다.

## 7. 두 임베딩 색인 생성 및 실제 검색

```powershell
.\.venv\Scripts\python.exe -X utf8 -m scripts.02_build_indexes --corpus $Corpus --embedding bge
.\.venv\Scripts\python.exe -X utf8 -m scripts.02_build_indexes --corpus $Corpus --embedding qwen
.\.venv\Scripts\python.exe -X utf8 -m scripts.check_integration --stage indexes
```

Qdrant에는 `check5060_..._<해시>` 컬렉션이 생성됩니다. 문서 전체 120개 색인과는 다른 **AS007 단일 문서 smoke**입니다. 중단됐으면 같은 원본/설정/명령으로 재실행합니다. 성공 배치부터 이어가며 기존 Cloud 컬렉션을 자동 삭제하지 않습니다. `active_bge.json`/`active_qwen.json`에는 절대 경로가 들어가므로 다른 PC 파일을 복사하지 않습니다.

## 8. 리랭커 다운로드·점수 계산

```powershell
.\.venv\Scripts\python.exe -X utf8 -m scripts.check_integration --stage reranker
```

처음 실행할 때 Hugging Face에서 가중치/tokenizer를 자동 다운로드하고 캐시에 저장합니다. Ollama에 리랭커를 pull하는 방식이 아닙니다. `device=cuda`, 관련 문장 점수 > 무관 문장 점수가 기대값입니다. 최초 다운로드 중에는 시간이 더 걸릴 수 있습니다.

## 9. 질문 1개 검색·답변 및 Langfuse readback

```powershell
.\.venv\Scripts\python.exe -X utf8 -m scripts.check_integration --stage query
.\.venv\Scripts\python.exe -X utf8 -m scripts.check_integration --stage trace
```

query는 실제 Hybrid+Reranker 파이프라인으로 1개 질문을 실행하고 답변·인용을 검사합니다. trace는 **새 추론 없이** 방금 실행의 Langfuse root/embedding/retrieval/generation/validation과 입출력 저장을 조회합니다. query PASS, trace FAIL이면 생성 자체보다 관측/읽기 권한/반영 지연/429를 확인하세요.

Mem0도 확인하려면 별도 실행합니다. 테스트 사용자에 선호 기억 **1개를 실제 저장**합니다.

```powershell
.\.venv\Scripts\python.exe -X utf8 -m scripts.check_integration --stage memory
```

기대값: 본인 검색 있음, 다른 사용자 검색 0. 테스트 user_id는 보고서에 남습니다. 일반 비교는 기억 OFF입니다.

## 10. FastAPI / Uvicorn 실행 - 터미널 A

새 PowerShell 창에서 실행하고 **창을 켜 둡니다**.

```powershell
Set-Location C:\project\ragdev
.\.venv\Scripts\python.exe -X utf8 -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
```

기존 작업 창에서 점검:

```powershell
.\.venv\Scripts\python.exe -X utf8 -m scripts.check_integration --stage api
```

기대값: health=200, 빈 질문=422. API 문서는 <http://127.0.0.1:8000/docs>입니다. OOM을 막기 위해 여러 Uvicorn worker를 사용하지 않습니다.

## 11. Next.js / React 실행 - 터미널 B

다른 새 PowerShell 창에서 실행하고 **창을 켜 둡니다**.

```powershell
Set-Location C:\project\ragdev\frontend
npm.cmd test
npm.cmd run build
npm.cmd run start
```

백엔드 주소가 기본 `http://127.0.0.1:8000`과 다를 때만, 위 start 전에 이 창에서 `$env:BACKEND_URL='정확한-backend-주소'`를 설정합니다. 프로젝트 루트 `.env`는 Next가 자동으로 읽는 파일이 아닙니다. Cloud 키는 프런트에 넣지 않습니다.

기존 작업 창에서 점검:

```powershell
.\.venv\Scripts\python.exe -X utf8 -m scripts.check_integration --stage web
Start-Process 'http://127.0.0.1:3000'
```

브라우저에서 질문 → 답변 → 출처 클릭까지 직접 확인하세요. HTTP PASS만으로 React 렌더링을 보증하지는 않습니다. 추천 첫 질문은 **“일반 건화물로 예약 가능한 컨테이너 종류는?”** 입니다.

실제 GPU·색인 범위는 이 점검 보고서와 index manifest로 확인하세요. AS007 전 페이지 smoke는 기존 120문서 앞2페이지 smoke와 다릅니다. 수정 코드를 받았으면 Next를 다시 build/start하고 브라우저를 Ctrl+Shift+R로 새로고침합니다.

## 12. 서버가 모두 켜진 뒤 전체 연동 체크

```powershell
Set-Location C:\project\ragdev
$Corpus = '6단계에서 출력된 실제 CORPUS_DIR 경로'
.\.venv\Scripts\python.exe -X utf8 -m scripts.check_integration --stage all --corpus $Corpus
$LASTEXITCODE
```

순서: tools → env → data → connections → corpus → indexes → reranker → query → trace → api → web. **실패한 첫 단계에서 중단하고 후속 단계는 SKIP**합니다. 실제 임베딩·짧은 생성·질문1개·trace 전송이 발생합니다. Mem0 쓰기 점검은 all에 포함하지 않습니다. 벤치마크·RAGAS/Judge 채점·문서 대량 삽입은 실행하지 않습니다.

출력 예:

```text
[PASS] connections: ...
[RUN ] indexes
[FAIL] indexes: FileNotFoundError
       NEXT: Run 02_build_indexes for BOTH profiles in this RUNTIME_DIR. ...
REPORT=.../runtime/5060-check/checks/setup-날짜.json
```

exit code 0=선택 단계 PASS, 1=FAIL. `REPORT=` 파일에 단계·상태·시간·조치가 남으며 비밀키/원시 예외 문자열은 출력하지 않습니다. 현재 CLI 환경에서 조치 문구는 영문입니다.

## 13. 단일 문서 검증 후 전체120 문서로 확장

아래 작업은 연동 확인 후 **선택적으로** 실행합니다. 전체 코퍼스는 오래 걸리고 Qdrant 저장 용량도 확인해야 합니다.

```powershell
.\.venv\Scripts\python.exe -X utf8 -m scripts.01_chunk_documents
```

새로 출력된 전체 코퍼스 경로로 변경합니다.

```powershell
$FullCorpus = '방금 출력된 전체 CORPUS_DIR 경로'
.\.venv\Scripts\python.exe -X utf8 -m scripts.check_integration --stage corpus --corpus $FullCorpus --require-full
.\.venv\Scripts\python.exe -X utf8 -m scripts.02_build_indexes --corpus $FullCorpus --embedding bge
.\.venv\Scripts\python.exe -X utf8 -m scripts.02_build_indexes --corpus $FullCorpus --embedding qwen
.\.venv\Scripts\python.exe -X utf8 -m scripts.check_integration --stage indexes --require-full
```

`--max-page`나 `--books`를 넣지 않아야 전체 색인입니다. 두 프로필의 active 포인터가 새 전체 색인으로 바뀝니다. 본 가이드에서는 `PUBLISH_METRICS=false`를 유지합니다. 정식 평가는 코드·설정·질문을 별도로 동결하고 새 실행 기록으로 관리합니다.

## 문제별 확인과 종료

| 실패 단계 | 먼저 확인할 것 |
|---|---|
| tools | 설치 후 터미널을 새로 열었는지, 명령 PATH, NVIDIA 드라이버 |
| env | 올바른 .venv, uv sync 완료, CUDA=True, .env 중복 키/경로 |
| data | 120 PDF 이름·part 구조, 평가5파일, DATA_DIR/EVALUATION_DIR |
| connections | Ollama 앱/모델, Cloud 주소·리전·키·권한·네트워크 |
| corpus | 출력 CORPUS_DIR, manifest failure_count, PDF 추출 기록 |
| indexes | 두 프로필 색인, RUNTIME_DIR 일치, digest/context 변경, Cloud 용량 |
| reranker/query | VRAM 부족, 다른 GPU 작업, 모델 다운로드, 저장된 result.json |
| trace | 수집 지연, Langfuse 키의 읽기 권한, URL, 429; trace만 재실행 |
| memory | mem0 설치, 전용 컬렉션, 사용자 필터, 로컬 구조화 출력 |
| api | 터미널 A가 살아 있는지, 포트8000 충돌, 같은 환경설정 |
| web | 터미널 B/build, BACKEND_URL, API 프록시, 브라우저 Console |

서버 종료는 A/B 창에서 각각 **Ctrl+C**. Ollama 모델 적재 상태 확인/해제는 아래와 같습니다. 진행 중 질문이나 평가가 없을 때 실행하세요.

```powershell
ollama ps
ollama stop qwen3:4b
ollama stop bge-m3
ollama stop qwen3-embedding:0.6b
```

Transformers 리랭커 VRAM은 Ollama가 관리하지 않습니다. 리랭커를 적재한 Python/Uvicorn 프로세스가 종료되어야 해제됩니다. `.env`, data, runtime, 모델, output은 Git에 올리지 않습니다.
