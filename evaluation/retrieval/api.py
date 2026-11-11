"""평가 화면용 읽기 전용 API.

웹의 "평가 지표" 패널이 비교 실행 요약(retrieval_summary.json)을 읽어 간다.
main.py는 router를 등록만 하고, 실행 결과를 바꾸거나 모델을 부르는 일은 하지 않는다.

  GET /evaluation/runs            비교 실행 목록 (최근 것 먼저)
  GET /evaluation/summary?run=..  실행 하나의 비교표·차트·코멘트용 요약 (run 생략 시 최근 실행)

runtime/은 git에 올라가지 않으므로, 확정된 실행의 요약은 snapshots/<실행 이름>.json으로 저장소에 둔다.
로컬에 같은 이름의 실행이 없으면 이 스냅샷을 보여 준다 → 팀원이 비교를 다시 돌리지 않아도 결과가 보인다.
  스냅샷 갱신: python -m evaluation.retrieval.api --snapshot runtime/5060-check/comparisons/<실행 이름>
"""
import argparse
import json
import re
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query

from backend.app.artifacts import write_json
from backend.app.config import get_settings

from .summary import build_insights, build_summary

router = APIRouter(prefix='/evaluation', tags=['evaluation'])
RUN_NAME = re.compile(r'^[A-Za-z0-9_-]{1,120}$')  # 폴더 이름만 허용 (경로 이동 방지)
SNAPSHOT_DIR = Path(__file__).parent / 'snapshots'

# 발제 필수 지표 (스냅샷·화면에 남기는 값)
REQUIRED_STATS = ['hit_at_4', 'recall_at_4', 'faithfulness', 'judge_facts', 'judge_conditions', 'judge_evidence', 'judge_decision']
REQUIRED_OPS = ['profile', 'mode', 'requests', 'errors', 'error_rate_pct', 'latency_mean_ms', 'latency_ok_n',
                'input_tokens_sum', 'input_tokens_known_requests', 'output_tokens_sum', 'output_tokens_known_requests']


def _comparisons_dir():
    return get_settings().runtime_dir / 'comparisons'


def _runs():
    root = _comparisons_dir()
    if not root.exists():
        return []
    folders = [p for p in root.iterdir() if p.is_dir() and ((p / 'retrieval_summary.json').exists() or (p / 'runs.jsonl').exists() or (p / 'items').exists())]
    return sorted(folders, key=lambda p: p.stat().st_mtime, reverse=True)


def _load(folder: Path):
    """요약 파일이 있으면 읽고, 없으면 실행 결과로 만들어 저장한다 (모델 호출 없음)."""
    path = folder / 'retrieval_summary.json'
    if path.exists():
        summary = json.loads(path.read_text(encoding='utf-8'))
    else:
        summary = build_summary(folder, get_settings().evaluation_dir)
        write_json(path, summary)
    if 'insights' not in summary:  # 이전 형식 파일 호환
        summary['insights'] = {p: build_insights([c for c in summary['configs'] if c['profile'] == p], qs)
                               for p, qs in summary.get('questions', {}).items()}
    return summary


MODES = ['naive', 'hybrid', 'hybrid_rerank']
CELL_KEYS = ['hit_at_4', 'recall_at_4', 'faithfulness', 'judge_facts', 'judge_conditions', 'judge_evidence', 'judge_decision',
             'status', 'answer_status', 'latency_ms', 'cause', 'error_code']


def _question_brief(questions):
    """발제 "질문별 결과와 세 방식의 평균을 같은 비교표에" + "개선·악화 대표 사례를 원문과 대조".

    table: 60문항 × 3방식의 발제 지표 값 (원문 제외, 가볍게)
    cases: 나빠진 문항의 정답 근거 · 실제 문맥 · 답변 · 채점 이유 (원문 대조용, 최대 8개)
    """
    counts = {k: sum(q['change_vs_naive'] == k for q in questions) for k in ['improved', 'worsened', 'same']}
    answer_counts = {k: sum(q.get('answer_change_vs_naive') == k for q in questions) for k in ['improved', 'worsened', 'same']}
    table = [{'question_id': q['question_id'], 'question': q.get('question'), 'expected_action': q.get('expected_action'),
              'gold_count': len(q['gold_books']), 'change': q['change_vs_naive'], 'answer_change': q.get('answer_change_vs_naive'),
              'modes': {m: {k: (q['modes'].get(m) or {}).get(k) for k in CELL_KEYS} for m in MODES}} for q in questions]
    worsened = [q for q in questions if 'worsened' in (q['change_vs_naive'], q.get('answer_change_vs_naive'))][:8]
    cases = []
    for q in worsened:
        modes = {}
        for m in MODES:
            cell = q['modes'].get(m) or {}
            modes[m] = {'answer': cell.get('answer'), 'answer_status': cell.get('answer_status'), 'recall_at_4': cell.get('recall_at_4'),
                        'cause': cell.get('cause'),
                        'judge': {k: {'score': cell.get('judge_' + k), 'reason': cell.get('judge_' + k + '_reason')} for k in ['facts', 'conditions', 'evidence', 'decision']},
                        'context': [{'book_id': c['book_id'], 'page': c['page'], 'text': (c.get('text') or '')[:280]} for c in cell.get('final_context', [])]}
        cases.append({'question_id': q['question_id'], 'question': q.get('question'), 'expected_action': q.get('expected_action'),
                      'reference_answer': q.get('reference_answer'), 'evidence': q.get('evidence'), 'gold_books': q['gold_books'],
                      'change': q['change_vs_naive'], 'answer_change': q.get('answer_change_vs_naive'), 'modes': modes})
    return {'counts': counts, 'answer_counts': answer_counts, 'table': table, 'cases': cases}


# ── 스냅샷: 저장소에 함께 올리는 확정 결과 ──────────────────────

def make_snapshot(data, run_name):
    """요약에서 발제 필수 지표·해석·선정 근거만 남긴다 (질문 원문·답변·경로 제외)."""
    configs = [{**{k: c.get(k) for k in REQUIRED_OPS}, **{k: c.get(k) for k in REQUIRED_STATS}} for c in data['configs']]
    run = {k: v for k, v in data['run'].items() if k != 'folder'}
    return {'run': {**run, 'run_name': run_name, 'source': 'snapshot'}, 'configs': configs,
            'insights': data.get('insights', {}), 'selection': data.get('selection', {}), 'cost': data.get('cost'),
            'metric_definitions': {k: v for k, v in data.get('metric_definitions', {}).items() if k in REQUIRED_STATS}}


def _snapshots():
    return sorted(SNAPSHOT_DIR.glob('*.json'), key=lambda p: p.name, reverse=True) if SNAPSHOT_DIR.exists() else []


# ── 엔드포인트 ──────────────────────────────────────────────

@router.get('/runs')
def runs():
    local = [{'run': p.name, 'has_summary': (p / 'retrieval_summary.json').exists(), 'source': 'local'} for p in _runs()]
    names = {r['run'] for r in local}
    return local + [{'run': p.stem, 'has_summary': True, 'source': 'snapshot'} for p in _snapshots() if p.stem not in names]


@router.get('/summary')
def summary(run: str | None = Query(default=None)):
    folders = _runs()
    snapshots = _snapshots()
    if run is not None:
        if not RUN_NAME.match(run):
            raise HTTPException(422, detail={'code': 'invalid_run', 'message': '실행 이름 형식이 올바르지 않습니다.'})
        folders = [p for p in folders if p.name == run]
        snapshots = [p for p in snapshots if p.stem == run]
    if not folders:
        # 로컬 실행이 없으면 저장소의 확정 결과(스냅샷)를 보여 준다
        if snapshots:
            return json.loads(snapshots[0].read_text(encoding='utf-8'))
        raise HTTPException(404, detail={'code': 'no_comparison', 'message': '비교 실행 결과가 없습니다. scripts.04_run_comparison 실행 후 다시 열어 주세요.'})
    data = _load(folders[0])
    return {
        'run': {**data['run'], 'folder': folders[0].name, 'source': 'local'},
        'configs': [{k: v for k, v in c.items()} for c in data['configs']],
        'insights': data.get('insights', {}),
        'questions': {p: _question_brief(qs) for p, qs in data.get('questions', {}).items()},
        'selection': data.get('selection', {}),
        'cost': data.get('cost'),
        'causes': data.get('causes', {}),
        'cause_labels': data.get('cause_labels', {}),
        'failures': data.get('failures', []),
        'human_reviews': data.get('human_reviews', []),
        'metric_definitions': data.get('metric_definitions', {}),
    }


# ── 스냅샷 만들기 (CLI) ──────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='비교 실행 요약을 저장소용 스냅샷으로 저장')
    parser.add_argument('--snapshot', required=True, type=Path, help='runtime/.../comparisons/<실행 이름>')
    folder = parser.parse_args().snapshot
    out = SNAPSHOT_DIR / f'{folder.name}.json'
    write_json(out, make_snapshot(_load(folder), folder.name))
    print(f'SNAPSHOT={out}')
