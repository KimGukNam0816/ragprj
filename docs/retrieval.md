# 권오륜 담당: retrieval

구현은 backend/app/retrieval/, 테스트는 backend/tests/retrieval/에서 관리합니다. 상세 공통 입출력·연결 방식·기존 경로 호환은 [팀 작업 안내](TEAM_WORKFLOW.md)를 참고하세요.

이 문서는 담당 기능 개선 내용과 실행 방법을 추가하는 전용 문서입니다. 공통 설정이나 다른 담당 폴더 변경이 필요하면 별도 통합 작업으로 분리합니다.

## 검색 방식 비교 지표와 보고서

세 검색 방식(naive, hybrid, hybrid_rerank)의 비교 실행 결과를 비교표 하나로 모으고, 브라우저로 여는 정적 보고서를 만듭니다. 새 모델 호출은 하지 않으며 이미 저장된 실행 결과만 읽습니다.

### 실행

```powershell
# 1) 세 방식 비교 실행 (기존)
.\.venv\Scripts\python.exe -X utf8 -m scripts.04_run_comparison --set core10 --run-name <이름>
# 2) 답변 채점 (기존, 선택). 없으면 보고서의 답변 지표 열이 비어 있습니다.
.\.venv\Scripts\python.exe -X utf8 -m scripts.05_evaluate_quality --comparison runtime\...\comparisons\<이름>
# 3) 요약 JSON과 정적 보고서
.\.venv\Scripts\python.exe -X utf8 -m evaluation.retrieval.summary --comparison runtime\...\comparisons\<이름> --html
```

출력은 같은 비교 폴더의 `retrieval_summary.json`과 `retrieval_report.html`입니다. HTML은 외부 라이브러리 없이 파일 하나로 열립니다.

### 발제 필수 지표 (계산식 변경 없음)

| 지표 | 계산 |
|---|---|
| Hit@4 | evidence_groups의 book_id(중복 제거) 중 하나라도 최종 문맥 4개에 있으면 1 |
| Recall@4 | 찾은 참고 근거 문서 수 / 전체 참고 근거 문서 수 |
| Faithfulness, Judge 4항목 | quality.py 결과를 그대로 집계. Judge는 항목별 충족률, 합산하지 않음 |
| 응답 시간 | 정상 요청 평균과 건수. 실패 요청 시간은 기록 건수만 표시 |
| 오류율 | 실패 요청 수 / 전체 요청 수 × 100과 건수 |
| 입출력 토큰 | 확인된 사용량 합계와 확인한 요청 수. 확인 못 한 값은 0으로 채우지 않음 |

평균 옆에 평가 대상 수(n), 해당 없음, 평가 불가 건수를 함께 표시합니다. 해당 없음과 평가 불가는 평균에서 제외하고 0점은 포함합니다. 근거 목록이 빈 문항은 검색 지표 해당 없음, 서비스 오류로 검색 결과도 저장되지 않은 요청은 평가 불가입니다. 서비스 오류의 Judge는 판단 0점, 나머지 항목은 평가 불가로 셉니다.

### 추가 지표 (도전 실습 A)

필수 지표와 별도 키로 기록하며 비교표의 "추가 지표" 표에 나옵니다.

| 지표 | 적용 대상 | 입력 | 계산 기준 |
|---|---|---|---|
| 페이지 Hit@4 / Recall@4 | 근거가 등록된 문항 | 최종 문맥 | (book_id, page)가 모두 같아야 인정 |
| MRR (문서·페이지) | 근거가 등록된 문항 | 순위가 매겨진 후보 전체 | 1 / 첫 정답 순위, 못 찾으면 0 |
| 근거 커버리지 | 근거가 등록된 문항 | 최종 문맥 | 근거 묶음 중 덮은 비율. 묶음 안 페이지 하나라도 있으면 덮음 |
| 근거 전부 찾음 | 근거가 등록된 문항 | 최종 문맥 | 묶음을 모두 덮은 문항 비율 |
| Recall@k 곡선 | 근거가 등록된 문항 | 후보 상위 1~20개 | 문서 단위 Recall의 k별 평균 |
| 중복률 | 최종 문맥이 있는 요청 | 최종 문맥 | 1 - 고유 페이지 수 / 청크 수 |
| p50 / p95 응답 시간 | 정상 요청 | 요청별 응답 시간 | 최근접 순위 백분위수 |

### 웹 평가 패널

채팅 화면 상단 오른쪽의 막대그래프 아이콘을 누르면 세 방식의 지표 막대, Recall@k 곡선, 응답 시간·오류·토큰, 지표별 해석 코멘트, 문항별 악화 사례가 열립니다. 백엔드 `GET /evaluation/summary`(구현 `evaluation/retrieval/api.py`)가 `runtime/.../comparisons/` 아래 최근 실행의 요약을 읽어 주며, 요약 파일이 없으면 저장된 실행 결과로 만들어 둡니다. 모델은 호출하지 않습니다.

해석 코멘트는 `summary.build_insights`의 규칙으로 만듭니다(같은 결과면 같은 문장). 원인 문장은 해당 패턴이 데이터에 보일 때만 붙입니다. 정식 지표가 아닌 실행(smoke, 합성 예시)은 패널에 표시됩니다.

공통 파일 변경: `main.py`에 router 등록 2줄, 프런트 프록시 허용 경로 2개, `page.tsx`에 아이콘 버튼. 패널은 `frontend/app/evaluation-panel.tsx`로 분리했고 눌렀을 때만 불러옵니다.

### 사람 확인 기록

대표 사례를 원문과 대조하다 자동 채점을 고친 경우, 비교 폴더에 `human_review.jsonl`을 두면 보고서의 해당 문항에 표시됩니다. 원래 점수는 덮어쓰지 않습니다.

```json
{"profile": "bge", "mode": "naive", "question_id": "AS-DEV07", "metric": "judge_facts", "original": 1, "revised": 0, "reason": "정정 공지의 값을 놓침"}
```

### 테스트

`backend/tests/retrieval/test_retrieval_metrics.py`가 외부 연결 없이 발제 예시(A·B 중 A만 찾으면 Hit 1, Recall 0.5), 해당 없음·평가 불가 집계, 보고서 생성과 문자 이스케이프를 확인합니다.
