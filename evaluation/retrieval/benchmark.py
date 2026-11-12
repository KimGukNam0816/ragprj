"""비교 실행: 같은 평가 질문을 임베딩 2개 × 검색 방식 3개(= 6구성)로 앱 파이프라인에 그대로 넣어 결과를 저장한다.

여기서는 채점하지 않는다. 문항마다 검색 후보·최종 문맥·답변·시간·토큰이 result.json에 남고,
지표 계산(Hit·Recall, RAGAS·Judge, 운영 지표)은 이 저장 결과로 나중에 한다.
실행: python -m scripts.04_run_comparison --set final60 --run-name <이름>
"""
import json
from datetime import datetime, timezone
from pathlib import Path

from backend.app.artifacts import digest, read_jsonl, write_json, write_jsonl
from backend.app.config import ROOT, Settings
from backend.app.observability import pinned_prompt_version
from backend.app.pipeline import SOURCE_HASHES, run_query
from backend.app.schemas import QueryRequest

MODES = ['naive', 'hybrid', 'hybrid_rerank']


# ── 평가 질문 고르기 ─────────────────────────────────────────

def questions(s, selection):
    """질문 파일(정답 없이 id·question만)에서 실행할 문항을 고른다. final60 = 전체 60개, core10 = 선정 10개."""
    all_questions = []
    for name in ['development_questions.json', 'additional_questions.json']:
        all_questions += json.loads((s.evaluation_dir/name).read_text(encoding='utf-8'))
    if len(all_questions) != 60 or len({q['id'] for q in all_questions}) != 60:
        raise ValueError('Final evaluation must have exactly 60 unique questions')
    if selection == 'core10':
        path=Path(s.eval_question_ids_file) if s.eval_question_ids_file else ROOT/'evaluation/selection/core10.json'
        if not path.is_absolute():path=ROOT/path
        chosen = json.loads(path.read_text(encoding='utf-8'))
        if len(chosen) != 10 or not {q['id'] for q in chosen} <= {q['id'] for q in all_questions}:
            raise ValueError('Core10 selection mismatch')
        return chosen
    if selection == 'final60':
        return all_questions
    raise ValueError('Unknown question selection')


# ── 6구성 비교 실행 ──────────────────────────────────────────

def run_comparison(s: Settings, selection, run_name=None, limit=None):
    # ── 1. 준비: 정책 확인 · 문항 선택 · 실행 이름 ──
    s.check_inference_policy()
    qs = questions(s, selection)
    if limit and limit < len(qs):
        if not s.smoke_run:
            raise ValueError('Partial question runs are smoke-only')
        qs = qs[:limit]
    run_name = run_name or f'smoke-{selection}-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    if not run_name.replace('-','').replace('_','').isalnum():
        raise ValueError('Invalid comparison name')
    folder = s.runtime_dir/'comparisons'/run_name

    # ── 2. 동일 조건 기록 (manifest): 질문·색인·모델·k·후보 수·소스 해시 ──
    # 같은 실행 이름에 설정이 달라지면 막는다 → 비교 조건이 섞이지 않게 (발제: 공통 설정을 같게)
    index_state = {p: json.loads((s.runtime_dir/'indexes'/f'active_{p}.json').read_text(encoding='utf-8')) for p in ['bge','qwen']}
    config = {'selection':selection, 'questions':qs, 'indexes':index_state, 'provider':s.generation_llm_provider,'model':s.local_llm_model,'top_k':s.final_top_k,'dense_k':s.dense_candidate_k,'bm25_k':s.bm25_candidate_k,'rrf':s.rrf_k,'rerank_k':s.rerank_candidate_k,'max_output':s.llm_max_output_tokens,'source_sha256':SOURCE_HASHES}
    manifest = {'run_name':run_name,'config_hash':digest(config),'config':config,'scope':'smoke' if s.smoke_run else 'formal','metric_eligible':False,'expected_requests':len(qs)*6}
    if (folder/'manifest.json').exists():
        prior=json.loads((folder/'manifest.json').read_text(encoding='utf-8'))
        if prior['config_hash'] != manifest['config_hash']:
            raise ValueError('Frozen comparison configuration changed; use a new run name')
    write_json(folder/'manifest.json',manifest)
    # TODO: drop fetch=False once prepare_context(prompt=) uses the fetched template; until then run_query uses
    # the local template, so the pin must record it (a Langfuse fetch failure would also abort the run for nothing).
    prompt_version=pinned_prompt_version(folder,fetch=False)

    # ── 3. 임베딩 × 검색 방식 × 문항 순서로 실행 (바꾸는 것은 profile·mode 두 가지뿐) ──
    for profile in ['bge','qwen']:
        for mode in MODES:
            for q in qs:
                path=folder/'items'/f'{profile}-{mode}-{q["id"]}.json'
                # 이미 성공한 항목은 건너뜀 → 중단 후 같은 명령으로 이어하기
                if path.exists() and json.loads(path.read_text(encoding='utf-8')).get('status') == 'ok':
                    continue
                result={'profile':profile,'mode':mode,'question_id':q['id'],'question':q['question'],'metric_eligible':False}
                try:
                    # 채팅 앱과 같은 파이프라인(run_query)을 그대로 호출. 기억(Mem0)은 끔
                    answer, output=run_query(s,QueryRequest(question=q['question'],mode=mode,embedding_profile=profile),run_id=run_name,question_id=q['id'],run_kind='comparison',prompt_version=prompt_version)
                    result.update(status='ok',result_path=str(output/'result.json'),response=answer.model_dump())
                except Exception as exc:
                    # 실패도 결과에 포함 (발제: 좋은 결과만 고르지 않음). 오류율에 집계됨
                    result.update(status='error',error_type=type(exc).__name__)
                    # Locate the saved failed request to retain latency, usage, and search evidence.
                    candidates=[]
                    for f in (s.runtime_dir/'runs'/run_name).glob('*/manifest.json'):
                        m=json.loads(f.read_text(encoding='utf-8'))
                        if m.get('question_id')==q['id'] and m.get('config',{}).get('embedding_profile')==profile and m.get('config',{}).get('retrieval_mode')==mode:
                            candidates.append(f)
                    if candidates:
                        result['result_path']=str(max(candidates,key=lambda f:f.stat().st_mtime).parent/'result.json')
                write_json(path,result)
                print(f'{run_name} {profile}/{mode}/{q["id"]}: {result["status"]}',flush=True)

    # ── 4. 마무리: 항목을 runs.jsonl로 모으고 완료·오류 수 기록 ──
    rows=[json.loads(p.read_text(encoding='utf-8')) for p in sorted((folder/'items').glob('*.json'))]
    write_jsonl(folder/'runs.jsonl',rows)
    manifest.update(completed_requests=len(rows),errors=sum(r['status']=='error' for r in rows))
    write_json(folder/'manifest.json',manifest)
    return folder
