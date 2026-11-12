"""검색 지표·요약·정적 보고서 테스트 (외부 서비스 없이 가짜 실행 폴더로 확인)."""
import json

import pytest

from backend.app.artifacts import write_json
from evaluation.retrieval import metrics
from evaluation.retrieval.html_report import render
from evaluation.retrieval.summary import build_summary, percentile, stat


def chunk(book, page, cid=None):
    return {'book_id': book, 'page': page, 'chunk_id': cid or f'{book}-{page}', 'text': f'{book} {page}쪽 내용'}


# 정답 묶음 2개: A 3쪽(또는 C 9쪽) / B 5쪽
REFERENCE = {'id': 'Q1', 'question': '질문', 'expected_action': 'answer', 'reference_answer': '답',
             'evidence_groups': [[{'book_id': 'A', 'page': 3, 'quote': '근거1'}, {'book_id': 'C', 'page': 9, 'quote': '근거1 대체'}],
                                 [{'book_id': 'B', 'page': 5, 'quote': '근거2'}]]}


# ── 발제 필수 지표 ───────────────────────────────────────────

def test_required_scores_follow_assignment_example():
    """발제 예시: 참고 문서 A·B 중 A만 찾으면 Hit 1, Recall 0.5. 같은 문서 청크 여러 개는 한 번만 센다."""
    ref = {'evidence_groups': [[{'book_id': 'A', 'page': 1}], [{'book_id': 'B', 'page': 1}]]}
    scores = metrics.retrieval_scores(ref, [chunk('A', 1), chunk('A', 2, 'x'), chunk('D', 1)])
    assert scores == {'hit_at_4': 1, 'recall_at_4': 0.5, 'retrieval_status': 'ok'}


def test_required_scores_mark_empty_reference_not_applicable():
    assert metrics.retrieval_scores({'evidence_groups': []}, [chunk('A', 1)])['retrieval_status'] == 'not_applicable'


# ── 추가 지표 ────────────────────────────────────────────────

def test_page_level_does_not_accept_other_page_of_same_document():
    final = [chunk('A', 80), chunk('B', 5)]
    scores = metrics.extended_scores(REFERENCE, final, final)
    # 문서 단위로는 A·B 모두 맞지만 페이지 단위로는 B 5쪽만 맞음 (정답 페이지 3개 중 1개)
    assert metrics.retrieval_scores(REFERENCE, final)['recall_at_4'] == pytest.approx(2 / 3)
    assert scores['page_hit_at_4'] == 1 and scores['page_recall_at_4'] == pytest.approx(1 / 3)


def test_rank_metrics_use_candidate_order():
    candidates = [chunk('D', 1), chunk('A', 3), chunk('E', 1), chunk('B', 5)]
    scores = metrics.extended_scores(REFERENCE, candidates[:2], candidates)
    assert scores['first_gold_rank_doc'] == 2 and scores['mrr_doc'] == 0.5
    curve = scores['recall_curve_doc']
    assert len(curve) == 20 and curve[0] == 0 and curve[1] == pytest.approx(1 / 3) and curve[3] == pytest.approx(2 / 3)
    assert curve[-1] == curve[3]  # 후보가 끝나면 값 유지


def test_missing_gold_gives_zero_reciprocal_rank():
    assert metrics.reciprocal_rank({'A'}, [chunk('D', 1)], metrics.doc_key) == 0.0


def test_evidence_coverage_accepts_alternative_location_in_group():
    final = [chunk('C', 9), chunk('D', 1)]
    cov = metrics.evidence_coverage(REFERENCE, final)
    assert cov == {'evidence_groups': 2, 'evidence_covered': 1, 'evidence_coverage': 0.5, 'evidence_all_covered': 0}
    assert metrics.evidence_coverage(REFERENCE, final + [chunk('B', 5)])['evidence_all_covered'] == 1


def test_citation_precision_checks_doc_and_page():
    cites = [{'book_id': 'A', 'page': 3}, {'book_id': 'A', 'page': 80}, {'book_id': 'D', 'page': 1}]
    result = metrics.citation_precision(REFERENCE, cites)
    assert result['citation_precision_doc'] == pytest.approx(2 / 3) and result['citation_precision_page'] == pytest.approx(1 / 3)
    assert metrics.citation_precision(REFERENCE, [])['citation_status'] == 'not_applicable'


def test_duplicate_rate_counts_same_page_chunks():
    final = [chunk('A', 1, 'a'), chunk('A', 1, 'b'), chunk('A', 1, 'c'), chunk('B', 2)]
    assert metrics.duplicate_rate(final) == 0.5
    assert metrics.duplicate_rate([]) is None


# ── 집계 규칙 ────────────────────────────────────────────────

def test_stat_excludes_not_applicable_and_unscorable_but_keeps_zero():
    s = stat([1, 0, None, None], ['ok', 'ok', 'not_applicable', 'unscorable'])
    assert s == {'mean': 0.5, 'n': 2, 'na': 1, 'unscorable': 1}


def test_percentile_returns_observed_value():
    assert percentile([5, 1, 3, 2, 4], 50) == 3
    assert percentile(list(range(1, 101)), 95) == 95
    assert percentile([], 95) is None


# ── 실행 폴더 → 요약 → 보고서 ────────────────────────────────

def make_run(tmp_path):
    """bge 3방식 × 문항 2개. hybrid_rerank의 Q2는 검색 전에 실패한 요청."""
    evaluation = tmp_path / 'evaluation'
    write_json(evaluation / 'development.json', [REFERENCE, {'id': 'Q2', 'question': '<b>근거 없는 질문</b>', 'evidence_groups': []}])
    folder = tmp_path / 'comparisons' / 'run1'
    write_json(folder / 'manifest.json', {'run_name': 'run1', 'scope': 'smoke', 'metric_eligible': False,
                                          'config': {'selection': 'core10', 'top_k': 4, 'provider': 'ollama', 'model': 'qwen3:4b'}})
    finals = {'naive': [chunk('D', 1), chunk('A', 80)], 'hybrid': [chunk('A', 3), chunk('D', 1)], 'hybrid_rerank': [chunk('A', 3), chunk('B', 5)]}
    for mode, final in finals.items():
        for qid in ['Q1', 'Q2']:
            failed = mode == 'hybrid_rerank' and qid == 'Q2'
            result_path = folder / 'runs' / f'{mode}-{qid}.json'
            saved = {'latency_ms': 900} if failed else {
                'final_context': final, 'candidates': final + [chunk('E', 1)],
                'generation': {'attempts': [{'input_tokens': 100, 'output_tokens': 20}]}}
            write_json(result_path, saved)
            item = {'profile': 'bge', 'mode': mode, 'question_id': qid, 'status': 'error' if failed else 'ok', 'result_path': str(result_path)}
            if not failed:
                item['response'] = {'status': 'answered', 'answer': '답변', 'citations': [], 'request_id': f'r-{mode}-{qid}', 'latency_ms': 1000 + len(mode) * 100}
            write_json(folder / 'items' / f'bge-{mode}-{qid}.json', item)
            judge = ({k: {'score': None, 'reason': 'service_error'} for k in ['facts', 'conditions', 'evidence']} | {'decision': {'score': 0, 'reason': 'service_error'}}
                     if failed else {k: {'score': 1, 'reason': '충족'} for k in ['facts', 'conditions', 'evidence', 'decision']})
            write_json(folder / 'scores' / f'bge-{mode}-{qid}.json', {
                'profile': 'bge', 'mode': mode, 'question_id': qid, 'judge': judge,
                'judge_status': 'service_error' if failed else 'ok',
                'faithfulness': None if failed else 0.8, 'faithfulness_status': 'unscorable' if failed else 'ok'})
    (folder / 'human_review.jsonl').write_text(json.dumps({'profile': 'bge', 'mode': 'naive', 'question_id': 'Q1', 'metric': 'judge_facts',
                                                           'original': 1, 'revised': 0, 'reason': '정정 값 누락'}, ensure_ascii=False) + '\n', encoding='utf-8')
    return folder, evaluation


def test_summary_follows_assignment_counting_rules(tmp_path):
    folder, evaluation = make_run(tmp_path)
    summary = build_summary(folder, evaluation)
    rows = {c['mode']: c for c in summary['configs']}
    assert set(rows) == {'naive', 'hybrid', 'hybrid_rerank'}

    # Q1 참고 문서는 A·B·C 3개. Q2는 근거 목록이 비어 해당 없음, 실패 요청은 평가 불가
    rerank = rows['hybrid_rerank']
    assert rerank['recall_at_4'] == {'mean': pytest.approx(2 / 3), 'n': 1, 'na': 0, 'unscorable': 1}
    assert rows['naive']['recall_at_4'] == {'mean': pytest.approx(1 / 3), 'n': 1, 'na': 1, 'unscorable': 0}
    assert rows['naive']['page_recall_at_4']['mean'] == 0

    # 운영: 오류율 %와 건수, 정상 요청만 평균, 확인된 토큰만 합계
    assert rerank['errors'] == 1 and rerank['error_rate_pct'] == 50
    assert rerank['latency_ok_n'] == 1 and rerank['latency_failed_recorded_n'] == 1
    assert rerank['input_tokens_sum'] == 100 and rerank['input_tokens_known_requests'] == 1

    # 서비스 오류: 판단은 0점으로 포함, 나머지는 평가 불가 (발제 STEP 6)
    assert rerank['judge_decision'] == {'mean': 0.5, 'n': 2, 'na': 0, 'unscorable': 0}
    assert rerank['judge_facts'] == {'mean': 1.0, 'n': 1, 'na': 0, 'unscorable': 1}
    assert rerank['faithfulness']['unscorable'] == 1

    q1 = next(q for q in summary['questions']['bge'] if q['question_id'] == 'Q1')
    assert q1['change_vs_naive'] == 'improved'
    assert summary['human_reviews'][0]['reason'] == '정정 값 누락'


def test_insights_explain_metrics_from_results(tmp_path):
    folder, evaluation = make_run(tmp_path)
    insights = build_summary(folder, evaluation)['insights']['bge']
    by_metric = {i['metric']: i for i in insights}
    # 가짜 실행에서 Naive와 Hybrid의 Q1 Recall이 둘 다 1/3이라 "효과 미미"로 판정돼야 함
    assert by_metric['hybrid']['title'] == 'Hybrid 효과 미미'
    assert by_metric['recall_at_4']['tone'] == 'warn'  # Hit 1.0 대비 Recall 0.67
    assert any(i['metric'] == 'error_rate' for i in insights)  # 실패 요청이 있으면 경고
    assert all(i['tone'] in {'good', 'warn', 'info'} and i['text'] for i in insights)


def test_evaluation_api_lists_and_summarizes_runs(tmp_path, monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from backend.app.config import Settings
    from evaluation.retrieval import api
    folder, evaluation = make_run(tmp_path)
    settings = Settings(_env_file=None, runtime_dir=tmp_path, evaluation_dir=evaluation)
    monkeypatch.setattr(api, 'get_settings', lambda: settings)
    monkeypatch.setattr(api, 'SNAPSHOT_DIR', tmp_path / 'no-snapshots')
    app = FastAPI(); app.include_router(api.router)
    client = TestClient(app)
    assert client.get('/evaluation/runs').json() == [{'run': 'run1', 'has_summary': False, 'source': 'local'}]
    body = client.get('/evaluation/summary').json()
    assert body['run']['folder'] == 'run1' and len(body['configs']) == 3 and body['insights']['bge']
    assert 'final_context' not in json.dumps(body['questions'])  # 원문은 화면 응답에서 제외
    assert (folder / 'retrieval_summary.json').exists()
    assert client.get('/evaluation/summary', params={'run': '../x'}).status_code == 422
    assert client.get('/evaluation/summary', params={'run': 'nope'}).status_code == 404


def test_evaluation_api_falls_back_to_committed_snapshot(tmp_path, monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from backend.app.config import Settings
    from evaluation.retrieval import api
    folder, evaluation = make_run(tmp_path)
    settings = Settings(_env_file=None, runtime_dir=tmp_path, evaluation_dir=evaluation)
    monkeypatch.setattr(api, 'get_settings', lambda: settings)
    snapshot = api.make_snapshot(api._load(folder), 'run1')
    # 스냅샷에는 발제 필수 지표만 남고 경로·문맥 원문은 빠진다
    assert set(snapshot['configs'][0]) == set(api.REQUIRED_OPS) | set(api.REQUIRED_STATS)
    assert 'final_context' not in json.dumps(snapshot) and str(tmp_path) not in json.dumps(snapshot)
    monkeypatch.setattr(api, 'SNAPSHOT_DIR', tmp_path / 'snapshots')
    (tmp_path / 'snapshots').mkdir()
    (tmp_path / 'snapshots' / 'saved1.json').write_text(json.dumps({**snapshot, 'run': {**snapshot['run'], 'run_name': 'saved1'}}), encoding='utf-8')
    app = FastAPI(); app.include_router(api.router)
    client = TestClient(app)
    assert {'run': 'saved1', 'has_summary': True, 'source': 'snapshot'} in client.get('/evaluation/runs').json()
    body = client.get('/evaluation/summary', params={'run': 'saved1'}).json()
    assert body['run']['source'] == 'snapshot' and len(body['configs']) == 3


def test_static_report_renders_and_escapes_text(tmp_path):
    folder, evaluation = make_run(tmp_path)
    page = render(build_summary(folder, evaluation))
    assert '비교표 (발제 필수 지표)' in page and '실패 요청' in page and '정정 값 누락' in page
    assert '추가 지표' not in page  # 보고서에는 발제 지표만 싣는다
    assert '&lt;b&gt;근거 없는 질문&lt;/b&gt;' in page and '<b>근거 없는 질문</b>' not in page
