# 개발자 가이드

신규 개발자가 설치·코드 탐색·검증·PR 제출 전에 확인할 작업 규칙입니다. 시스템 설계는 [ARCHITECTURE.md](ARCHITECTURE.md), 설치와 실제 연동은 [SETUP.md](SETUP.md), 기능 개요는 [README.md](README.md)를 읽습니다.

## 먼저 지킬 사항

- 모든 프로젝트 LLM 추론은 로컬만 사용합니다. 외부 fallback을 추가하지 않습니다.
- 기존 비교·품질·보안 평가는 별도 요청 없이 실행하거나 재개하지 않습니다. CI는 RAG 평가를 수행하지 않습니다.
- `.env`, 키·토큰, 개인정보, runtime·trace·사용자 기억·생성 결과를 커밋하지 않습니다.
- 공개 승인한 `data/advanced/Advanced_shipping_part*/documents/*.pdf`와 `evaluation/*.json`만 데이터로 추적합니다. 새로운 데이터도 공개 승인을 확인합니다.
- main 직접 push·force push로 보호 규칙을 우회하지 않습니다. 변경은 작업 브랜치와 PR로 반영합니다.

## 새 PC 시작

Windows 11, Git, Node 24 LTS, uv, Ollama와 호환 NVIDIA 드라이버를 준비합니다. RTX 5060 8GB의 단계별 설정은 SETUP을 따릅니다. Python은 3.12를 사용하고 의존성 lock을 유지합니다.

```powershell
git clone https://github.com/KimGukNam0816/ragprj.git ragdev
Set-Location ragdev
uv python install 3.12
uv sync --locked --python 3.12 --extra memory --extra dev
npm.cmd --prefix frontend ci
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
```

각 명령이 실패하면 중단하고 원인을 확인합니다. 실제 키는 본인 `.env`에서만 설정하며 파일 내용을 로그에 출력하지 않습니다. 원본 데이터는 clone에 포함됩니다. 모델 다운로드·색인 생성·실제 질의는 설치만으로 자동 실행되지 않으며 SETUP 단계별로 진행합니다.

`frontend/.npmrc`는 느린 다운로드에 대기 시간 10분, 재시도 3회, 동시 연결 3개, 캐시 우선 사용을 적용합니다. 전역 npm 설정을 바꾸지 않아도 됩니다.

## 코드 읽는 순서

1. `schemas.py`, `config.py`: 입력·응답·로컬 추론 정책과 실제 설정.
2. `main.py`, `pipeline.py`: HTTP 요청부터 공통 질의·실패 처리까지.
3. `ingestion.py`, `indexing.py`, `reranker.py`: 청킹·색인·검색·리랭크.
4. `providers.py`, `prompts.py`: 실제 모델 호출·생성 계약.
5. `memory.py`, `observability.py`, `artifacts.py`: 기억·추적·저장.
6. `frontend/app/page.tsx`, 서버 프록시: 채팅·출처 표시·Backend 연결.
7. 변경 기능과 관련된 `backend/tests/`, `frontend/tests/`: 회귀 조건.

환경·모델·Cloud 실연동이 필요한 코드를 문서나 타입 검사 때문에 실행하지 않습니다. `scripts/check_integration.py`의 일부 stage는 실제 로컬 추론·Cloud 쓰기를 수행하므로 `--stage all`을 단순 정적 점검으로 취급하지 않습니다.

## 개발 서버

의존성·환경·색인 점검을 마친 뒤 별도 터미널에서 실행합니다.

```powershell
# 프로젝트 루트: Backend
.\.venv\Scripts\python.exe -X utf8 -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
```

```powershell
# 프로젝트 루트: Frontend
npm.cmd --prefix frontend run build
npm.cmd --prefix frontend run start
```

채팅은 `http://127.0.0.1:3000`, API 문서는 `http://127.0.0.1:8000/docs`입니다. 프런트 개발 중에는 `npm.cmd --prefix frontend run dev`를 사용할 수 있습니다. 설정 변경 후 Backend를 재시작하고, production UI 변경 후 다시 build/start합니다. 기본 데모에 서버 인증은 없습니다.

## 변경별 검증

```powershell
# Python 문법 검사: import·모델 호출 없음
.\.venv\Scripts\python.exe -m compileall -q backend scripts evaluation

# 프런트 빌드 후 생성된 Next 타입을 포함해 검사
npm.cmd --prefix frontend run build
npm.cmd --prefix frontend run typecheck

# 관련 코드 변경 시 단위 테스트: RAG 품질 평가와 별개
.\.venv\Scripts\python.exe -m pytest backend/tests -q
npm.cmd --prefix frontend test

git diff --check
```

변경에 맞는 검사만 실행하고 외부 서비스·모델 실연동은 필요한 범위를 명시합니다. 문서만 바꾸었을 때 로컬 추론이나 기존 평가를 실행하지 않습니다. 문법·타입·빌드 통과는 업무 정답성·인용 의미·보안·GPU 적합성을 증명하지 않습니다.

## GitHub 협업

Public 저장소 clone에는 인증이 필요 없지만 push·PR에는 본인 계정 인증과 협업자 권한이 필요합니다. 소유자 토큰을 공유하지 않습니다. Git Credential Manager 브라우저 로그인, SSH 또는 본인 GitHub CLI 인증을 사용합니다.

```powershell
git switch main
git pull --ff-only origin main
git switch -c docs/describe-change
# 필요한 파일만 수정하고 검증
git status --short
git diff --check
git add -- ARCHITECTURE.md DEVELOPER.md
git commit -m "Document architecture and developer workflow"
git push -u origin docs/describe-change

# GitHub CLI 설치·본인 로그인 후 실행
gh pr create --base main --title "Document architecture and developer workflow" --body "Describe the final changes and validation."
gh pr checks --watch
```

`git add` 경로·브랜치·제목·본문은 실제 작업에 맞게 바꿉니다. PR 본문에는 문제·변경 결과·검증·미완료 범위를 기록합니다. 로컬 커밋 작성자 설정은 GitHub 로그인이나 저장소 권한 부여와 별개입니다.

### 필수 CI와 자동 병합

| Job | 검사 |
| --- | --- |
| Python syntax | Python 3.12 compileall |
| Frontend checks | Node 24, npm ci, build, typecheck |
| CI required | 위 두 job이 모두 성공해야 통과 |

main 보호 규칙은 PR과 `CI required` 통과를 요구하며 승인 필수 수는 현재 0명입니다. 직접 push·강제 push·삭제를 제한합니다. GitHub ruleset은 저장소 서버 설정으로, clone만으로 다른 저장소에 복제되지 않습니다.

자동 병합은 `.github/workflows/auto-merge.yml`이 성공한 PR CI 완료 후 처리합니다. main 대상, Draft 아님, 원본 저장소 내부 브랜치, 작성자의 write/maintain/admin 권한, 현재 head와 검사 SHA 일치를 확인합니다. clean 상태는 해당 SHA로 squash 병합하고 추가 조건이 남으면 auto-merge를 예약합니다. 외부 fork PR은 자동 대상이 아닙니다.

CI와 자동화 workflow 수정에는 계정 인증의 workflow 쓰기 권한도 필요할 수 있습니다. 병합이 안 되면 Actions의 자동화 실행 로그·충돌·권한·Draft·base·필수 검사를 확인합니다. auto-merge와 squash 허용 설정은 저장소 Settings에서 관리합니다.

## 데이터·설정 변경 시 주의

- PDF·원본 JSON의 바이트는 `.gitattributes`로 보존합니다. 원본을 포맷팅하거나 재저장해 해시를 불필요하게 바꾸지 않습니다.
- 평가 정답·보안 메타데이터를 질문 생성 입력이나 검색 색인에 넣지 않습니다.
- 임베딩 모델·digest·context 또는 청킹을 바꾸면 새 색인으로 검증합니다. 기존 동결 실행에 변경 설정을 섞지 않습니다.
- Mem0 `user_id`는 로컬 데모 식별자입니다. 클라이언트 입력을 신뢰하는 상태를 인증·권한 완성으로 간주하지 않습니다.
- Windows에서 manifest 교체가 PermissionError로 실패하면 중복 프로세스·파일 잠금·속성을 확인합니다. 재시도 처리는 아직 구현됐다고 가정하지 않습니다. 원본·체크포인트를 임의 삭제하지 않습니다.
- GPU OOM은 LOW_VRAM_MODE와 모델 상주·문맥 길이를 점검합니다. 설정을 바꾸면 새 실행 기록으로 구분하고 필요하면 색인을 다시 만듭니다.
- 현재 미완료 범위는 정식 전체 품질·자원 측정, 표 의미 검수, 독립적인 자동 채점 검수, 실서비스 인증·세션·접근통제입니다.
