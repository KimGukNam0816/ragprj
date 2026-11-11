"""비교 실행 결과를 요약 JSON 하나로 모은다 (웹 화면·정적 보고서·발표 자료의 공통 입력).

입력: evaluation/retrieval/benchmark.py가 만든 runtime/comparisons/<run>/ 폴더
  - manifest.json, items/*.json (또는 runs.jsonl), 각 항목의 result.json
  - scores/*.json (또는 scores.jsonl): quality.py의 RAGAS·Judge 결과. 없으면 답변 지표는 비워 둔다.
  - human_review.jsonl (선택): 사람이 원문과 대조해 점수를 고친 기록
출력: <run>/retrieval_summary.json

발제 STEP 6·7의 집계 규칙을 따른다.
  - 평균 옆에 평가 대상 수(n), 해당 없음(na), 평가 불가(unscorable) 건수를 함께 둔다.
  - 해당 없음과 평가 불가는 평균에서 빼고, 0점은 포함한다. 빈 값을 0이나 1로 바꾸지 않는다.
  - 응답 시간은 정상 요청 평균과 건수, 오류율은 실패 수 / 전체 요청 수 × 100과 건수.
  - 토큰은 합계와 사용량을 확인한 요청 수. 확인 못 한 값은 0으로 채우지 않는다.

실행: python -X utf8 -m evaluation.retrieval.summary --comparison runtime/.../comparisons/<run> [--html]
"""
import argparse
import json
import math
from pathlib import Path
from statistics import mean, median

from backend.app.artifacts import read_jsonl, write_json

from .metrics import citation_precision, extended_scores, gold_books, retrieval_scores

PROFILES = ['bge', 'qwen']
MODES = ['naive', 'hybrid', 'hybrid_rerank']
JUDGE_ITEMS = ['facts', 'conditions', 'evidence', 'decision']
JUDGE_LABEL = {'facts': '사실', 'conditions': '조건', 'evidence': '근거', 'decision': '판단'}
TEXT_LIMIT = 600  # 상세 패널용 원문 길이 제한 (요약 파일이 너무 커지지 않게)


# ── 파일 읽기 ────────────────────────────────────────────────

def _load_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def load_references(evaluation_dir):
    """평가 정답(development.json + additional.json)을 문항 ID로 찾을 수 있게 만든다."""
    refs = {}
    for name in ['development.json', 'additional.json']:
        path = Path(evaluation_dir) / name
        if path.exists():
            for row in _load_json(path):
                refs[row['id']] = row
    return refs


def load_items(folder):
    """비교 실행의 항목(구성 × 문항). runs.jsonl이 없으면 items/ 폴더에서 읽는다."""
    if (folder / 'runs.jsonl').exists():
        return list(read_jsonl(folder / 'runs.jsonl'))
    return [_load_json(p) for p in sorted((folder / 'items').glob('*.json'))]


def load_scores(folder):
    """quality.py 채점 결과를 (profile, mode, question_id) 키로 묶는다."""
    if (folder / 'scores.jsonl').exists():
        rows = list(read_jsonl(folder / 'scores.jsonl'))
    else:
        rows = [_load_json(p) for p in sorted((folder / 'scores').glob('*.json'))] if (folder / 'scores').exists() else []
    return {(r['profile'], r['mode'], r['question_id']): r for r in rows}


def load_reviews(folder):
    """사람 확인 기록. 한 줄에 {profile, mode, question_id, metric, original, revised, reason}."""
    path = folder / 'human_review.jsonl'
    return list(read_jsonl(path)) if path.exists() else []


# ── 집계 도구 ────────────────────────────────────────────────

def stat(values, statuses=None):
    """평균과 건수. values의 None은 statuses로 해당 없음/평가 불가를 구분한다 (없으면 해당 없음으로 셈)."""
    statuses = statuses or ['not_applicable' if v is None else 'ok' for v in values]
    # 계산은 됐지만 값이 없는 경우(예: 근거 묶음이 비어 있음)는 해당 없음으로 센다
    statuses = ['not_applicable' if st == 'ok' and v is None else st for v, st in zip(values, statuses)]
    valid = [v for v, st in zip(values, statuses) if st == 'ok' and v is not None]
    result = {'mean': mean(valid) if valid else None, 'n': len(valid),
              'na': sum(st == 'not_applicable' for st in statuses),
              'unscorable': sum(st == 'unscorable' for st in statuses)}
    pending = sum(st == 'pending' for st in statuses)
    if pending:  # 아직 채점하지 않은 문항 (해당 없음과 구분)
        result['pending'] = pending
    return result


def percentile(values, q):
    """최근접 순위 백분위수: 정렬한 값의 ceil(q/100 × n)번째. 보간 없이 실제 관측값 하나를 돌려준다."""
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, math.ceil(q / 100 * len(ordered)) - 1)]


def _clip(text):
    text = text or ''
    return text if len(text) <= TEXT_LIMIT else text[:TEXT_LIMIT] + '…'


def _chunk_view(c):
    return {'book_id': c.get('book_id'), 'page': c.get('page'), 'chunk_id': c.get('chunk_id'), 'text': _clip(c.get('text'))}


# ── 문항 하나 평가 ───────────────────────────────────────────

CAUSE_LABEL = {
    'service_error': '시스템 오류 (답변 미생성)',
    'no_evidence': '정답 근거 미등록 문항',
    'retrieval_miss': '검색 실패 (후보에도 정답 문서 없음)',
    'ranked_out': '순위 밀림 (후보에는 있었으나 상위 4개 밖)',
    'context_budget': '문맥 예산으로 정답 청크 제외',
    'retrieval_partial': '근거 일부만 검색 → 답 틀림',
    'generation': '근거는 찾았으나 답이 틀림 (생성·판단)',
    'ok': '정상',
}


def classify_cause(ok, reference, retrieval, candidates, dropped_ids, answer_scores):
    """한 요청이 왜 실패했는지 한 가지로 분류한다. 판단 근거는 저장된 검색 후보·최종 문맥·채점뿐이다."""
    if not ok:
        return 'service_error'
    gold = gold_books(reference)
    if not gold:
        return 'no_evidence'
    recall = retrieval.get('recall_at_4')
    if recall == 0:
        if any(c.get('chunk_id') in dropped_ids and c.get('book_id') in gold for c in candidates):
            return 'context_budget'
        return 'ranked_out' if any(c.get('book_id') in gold for c in candidates) else 'retrieval_miss'
    if answer_scores.get('judge_facts') == 0:
        return 'retrieval_partial' if (recall or 0) < 1 else 'generation'
    return 'ok'


def evaluate_item(item, reference, score):
    """구성 하나 × 문항 하나의 검색·답변·운영 값을 모은다."""
    saved = _load_json(item['result_path']) if item.get('result_path') and Path(item['result_path']).exists() else {}
    final_context = saved.get('final_context', [])
    candidates = saved.get('candidates', [])
    ok = item.get('status') == 'ok'

    # 1. 검색 지표: 서비스가 실패해도 검색 결과가 저장됐으면 계산한다 (quality.py와 같은 규칙)
    retrieval = retrieval_scores(reference, final_context)
    extended = extended_scores(reference, final_context, candidates)
    if not ok and 'final_context' not in saved:
        retrieval = {'hit_at_4': None, 'recall_at_4': None, 'retrieval_status': 'unscorable'}
        extended = {'extended_status': 'unscorable'}

    # 2. 운영 지표: 시도별 기록에서 확인된 토큰만 더한다
    attempts = (saved.get('generation') or {}).get('attempts') or (saved.get('error') or {}).get('attempts') or []
    tokens = {}
    for key in ['input_tokens', 'output_tokens']:
        known = [a.get(key) for a in attempts if a.get(key) is not None]
        tokens[key] = sum(known) if known else None
    response = item.get('response') or {}
    latency = response.get('latency_ms') if ok else saved.get('latency_ms')
    # 인용 정확도: 답변이 만든 출처를 정답 근거와 대조 (서비스 실패면 평가 불가)
    citation = citation_precision(reference, response.get('citations') or []) if ok else {
        'citation_precision_doc': None, 'citation_precision_page': None, 'citation_status': 'unscorable'}

    # 3. 답변 지표 (quality.py 결과가 있을 때만)
    answer_scores = {}
    if score:
        answer_scores['faithfulness'] = score.get('faithfulness')
        answer_scores['faithfulness_status'] = score.get('faithfulness_status')
        judge = score.get('judge') or {}
        for name in JUDGE_ITEMS:
            grade = judge.get(name) or {}
            value = grade.get('score')
            if score.get('judge_status') == 'unscorable':
                status = 'unscorable'
            elif value is None:
                # 서비스 오류로 출력이 없으면 평가 불가, 평가할 대상이 없으면 해당 없음 (발제 STEP 6)
                status = 'unscorable' if grade.get('reason') == 'service_error' else 'not_applicable'
            else:
                status = 'ok'
            answer_scores['judge_' + name] = value
            answer_scores['judge_' + name + '_status'] = status
            answer_scores['judge_' + name + '_reason'] = grade.get('reason')

    # 4. 원인 분류 (발제: 주요 질문이 나빠지면 원인 제시, 정정 공지를 못 찾은 경우와 문맥에서 빠뜨린 경우를 구분)
    error = saved.get('error') or {}
    cause = classify_cause(ok, reference, retrieval, candidates, saved.get('dropped_chunk_ids') or [], answer_scores)

    return {
        'question_id': item['question_id'], 'profile': item['profile'], 'mode': item['mode'],
        'status': item.get('status'), 'error_type': item.get('error_type'),
        'error_code': error.get('code'), 'error_message': error.get('message'), 'cause': cause,
        'answer_status': response.get('status'), 'answer': _clip(response.get('answer')),
        'citations': response.get('citations', []), 'request_id': response.get('request_id') or saved.get('request_id'),
        'trace_id': response.get('trace_id') or saved.get('trace_id'),
        'latency_ms': latency, **tokens, **retrieval, **extended, **citation, **answer_scores,
        'final_context': [_chunk_view(c) for c in final_context],
    }


# ── 구성별 요약 ──────────────────────────────────────────────

def summarize_config(rows, max_k=20):
    """구성 하나(임베딩 × 방식)의 모든 문항을 비교표 한 줄로 만든다."""
    def column(key, status_key):
        return stat([r.get(key) for r in rows], [r.get(status_key) or 'not_applicable' for r in rows])

    ok_rows = [r for r in rows if r['status'] == 'ok']
    latency_ok = [r['latency_ms'] for r in ok_rows if r['latency_ms'] is not None]
    errors = sum(r['status'] != 'ok' for r in rows)
    summary = {
        'requests': len(rows),
        # ── 발제 필수 ──
        'hit_at_4': column('hit_at_4', 'retrieval_status'),
        'recall_at_4': column('recall_at_4', 'retrieval_status'),
        'latency_mean_ms': mean(latency_ok) if latency_ok else None,
        'latency_ok_n': len(latency_ok),
        'latency_failed_recorded_n': sum(r['latency_ms'] is not None for r in rows if r['status'] != 'ok'),
        'errors': errors,
        'error_rate_pct': errors / len(rows) * 100 if rows else None,
    }
    for key in ['input_tokens', 'output_tokens']:
        known = [r[key] for r in rows if r.get(key) is not None]
        summary[key + '_sum'] = sum(known) if known else None
        summary[key + '_known_requests'] = len(known)
    if any('faithfulness_status' in r for r in rows):
        # 채점 결과가 없는 문항은 '대기'로 센다 (해당 없음으로 섞지 않음)
        def answer_column(key, status_key):
            return stat([r.get(key) for r in rows], [r.get(status_key) or ('pending' if 'faithfulness_status' not in r else 'not_applicable') for r in rows])
        summary['faithfulness'] = answer_column('faithfulness', 'faithfulness_status')
        for name in JUDGE_ITEMS:
            summary['judge_' + name] = answer_column('judge_' + name, 'judge_' + name + '_status')

    # ── 추가 지표 (도전 실습 A) ──
    for key in ['page_hit_at_4', 'page_recall_at_4', 'mrr_doc', 'mrr_page', 'evidence_coverage', 'evidence_all_covered', 'duplicate_rate']:
        summary[key] = column(key, 'extended_status')
    for key in ['citation_precision_doc', 'citation_precision_page']:
        summary[key] = column(key, 'citation_status')
    curves = [r['recall_curve_doc'] for r in rows if r.get('extended_status') == 'ok' and r.get('recall_curve_doc')]
    summary['recall_curve_doc'] = [mean(c[k] for c in curves) for k in range(max_k)] if curves else None
    summary['latency_p50_ms'] = median(latency_ok) if latency_ok else None
    summary['latency_p95_ms'] = percentile(latency_ok, 95)
    return summary


# ── 문항별 비교 (개선·악화 사례 찾기) ──────────────────────────

def question_rows(per_item, references, profile):
    """임베딩 하나에서 문항별로 세 방식의 값을 나란히 놓고, Naive 대비 변화를 표시한다."""
    output = []
    for qid in sorted({r['question_id'] for r in per_item if r['profile'] == profile}):
        reference = references.get(qid, {})
        cells = {r['mode']: r for r in per_item if r['profile'] == profile and r['question_id'] == qid}
        def compare(key):
            base, best = (cells.get('naive') or {}).get(key), (cells.get('hybrid_rerank') or {}).get(key)
            return None if base is None or best is None else ('improved' if best > base else 'worsened' if best < base else 'same')
        change = compare('recall_at_4')            # 검색: Naive → Hybrid+Reranker의 Recall@4 변화
        answer_change = compare('judge_facts')     # 답변: 같은 비교의 Judge "사실" 변화
        output.append({
            'question_id': qid,
            'question': reference.get('question'),
            'expected_action': reference.get('expected_action'),
            'gold_books': sorted(gold_books(reference)),
            'evidence': [[{'book_id': x['book_id'], 'page': x['page'], 'quote': _clip(x.get('quote'))} for x in g] for g in reference.get('evidence_groups', [])],
            'reference_answer': _clip(reference.get('reference_answer')),
            'change_vs_naive': change,
            'answer_change_vs_naive': answer_change,
            'modes': cells,
        })
    order = {'worsened': 0, 'improved': 1, 'same': 2, None: 3}
    return sorted(output, key=lambda q: (min(order[q['change_vs_naive']], order[q['answer_change_vs_naive']]), q['question_id']))


# ── 최종 구성 선정 근거 (발제: 품질이 비슷하면 시간·토큰으로) ───

def selection_rationale(configs):
    """Judge(사실·조건·근거·판단) 평균과 Recall@4로 후보를 정하고, 차이가 작으면 응답 시간이 짧은 쪽을 고른다.

    점수를 하나로 합쳐 발표하려는 것이 아니라 선택 순서를 정하기 위한 내부 기준이다(발제는 네 항목 합산을 요구하지 않음).
    """
    def quality(c):
        judge = [_m(c, 'judge_' + k) for k in JUDGE_ITEMS]
        judge = [v for v in judge if v is not None]
        return (mean(judge) if judge else None, _m(c, 'recall_at_4'))
    ranked = sorted(configs, key=lambda c: tuple(-(v if v is not None else -1) for v in quality(c)))
    if not ranked:
        return None
    best = ranked[0]
    close = [c for c in ranked[1:] if all(a is not None and b is not None and abs(a - b) <= 0.03 for a, b in zip(quality(c), quality(best)))]
    faster = min([best] + close, key=lambda c: c.get('latency_mean_ms') or float('inf'))
    jq, rc = quality(best)
    text = (f'{best["profile"]} / {best["mode"]}가 가장 높은 답변 품질(Judge 평균 {jq:.2f})과 Recall@4({rc:.2f})를 보였습니다.'
            if jq is not None else f'{best["profile"]} / {best["mode"]}가 가장 높은 Recall@4({rc:.2f})를 보였습니다. 답변 채점 전이라 검색 기준으로만 고른 후보입니다.')
    if faster is not best:
        text += (f' 다만 {faster["profile"]} / {faster["mode"]}와 품질 차이가 0.03 이내이고 평균 응답이 '
                 f'{(best["latency_mean_ms"] - faster["latency_mean_ms"]) / 1000:.1f}초 짧아, 발제 기준(품질이 비슷하면 시간·토큰)으로 이쪽을 우선 후보로 둡니다.')
    return {'profile': faster['profile'], 'mode': faster['mode'], 'text': text}


# ── 지표별 짧은 해석 (화면의 코멘트) ─────────────────────────
# 규칙 기반이라 같은 결과면 항상 같은 코멘트가 나온다. LLM을 부르지 않는다.

def _m(config, key):
    value = (config or {}).get(key)
    return value.get('mean') if isinstance(value, dict) else value


def _recall_by_gold_count(questions, mode):
    """정답 문서 수별 Recall@4 평균. 여러 기록철에 흩어진 문항이 약한지 보기 위함."""
    buckets = {}
    for q in questions:
        r = (q['modes'].get(mode) or {}).get('recall_at_4')
        if r is not None and q['gold_books']:
            buckets.setdefault(len(q['gold_books']), []).append(r)
    return {n: mean(v) for n, v in sorted(buckets.items())}


def build_insights(configs, questions):
    """임베딩 하나의 세 방식 결과를 보고 지표별로 "어떻게 나왔고 왜 그런지" 한두 문장씩 만든다."""
    by = {c['mode']: c for c in configs}
    naive, hybrid, rerank = by.get('naive'), by.get('hybrid'), by.get('hybrid_rerank')
    best = max(configs, key=lambda c: _m(c, 'recall_at_4') or -1) if configs else None
    out = []

    def add(metric, tone, title, text):
        out.append({'metric': metric, 'tone': tone, 'title': title, 'text': text})

    # 1. Hit 대비 Recall
    if best and _m(best, 'hit_at_4') is not None and _m(best, 'recall_at_4') is not None:
        hit, rec = _m(best, 'hit_at_4'), _m(best, 'recall_at_4')
        buckets = _recall_by_gold_count(questions, best['mode'])
        many = [v for n, v in buckets.items() if n >= 4]
        detail, cause = '', ''
        if 1 in buckets and many:
            detail = f' 정답 문서 1개 문항은 {buckets[1]:.2f}, 4개 이상 문항은 {mean(many):.2f}.'
            if buckets[1] - mean(many) > 0.2:
                cause = ' 같은 건이 여러 기록철에 흩어진 문항에서 4자리가 부족한 것이 주원인입니다.'
        tone = 'good' if rec >= 0.9 else 'warn' if hit - rec > 0.15 else 'info'
        add('recall_at_4', tone, f'Recall@4 {rec:.2f} · Hit@4 {hit:.2f}',
            f'정답 문서를 하나는 찾지만 전부는 못 가져옵니다.{detail}{cause}'
            if hit - rec > 0.15 else f'찾아야 할 문서를 대부분 가져옵니다.{detail}')

    # 2. Hybrid 효과 (키워드 검색 추가)
    if naive and hybrid and _m(naive, 'recall_at_4') is not None and _m(hybrid, 'recall_at_4') is not None:
        d = _m(hybrid, 'recall_at_4') - _m(naive, 'recall_at_4')
        if d > 0.02:
            add('hybrid', 'good', f'Hybrid Recall +{d:.2f}', '예약번호·장비코드 같은 정확한 단어를 BM25가 잡아, 의미 검색이 놓친 문서를 보충했습니다.')
        elif d < -0.02:
            add('hybrid', 'warn', f'Hybrid Recall {d:.2f}', 'BM25 후보가 비슷한 양식의 다른 건 문서를 섞어 오히려 자리를 빼앗았습니다.')
        else:
            add('hybrid', 'info', 'Hybrid 효과 미미', '키워드 검색을 더해도 상위 4개 구성이 거의 같습니다. 두 검색이 같은 문서를 찾고 있습니다.')

    # 3. 리랭커 효과 (Recall@4) 와 비용 (응답 시간)
    if hybrid and rerank and _m(hybrid, 'recall_at_4') is not None and _m(rerank, 'recall_at_4') is not None:
        d = _m(rerank, 'recall_at_4') - _m(hybrid, 'recall_at_4')
        dt = (rerank.get('latency_mean_ms') or 0) - (hybrid.get('latency_mean_ms') or 0)
        cost = ''
        if hybrid.get('latency_mean_ms') and rerank.get('latency_mean_ms'):
            cost = f' 대신 평균 응답이 {abs(dt) / 1000:.1f}초 {"늘었습니다" if dt > 0 else "줄었습니다"}.'
        if d > 0.02:
            add('rerank', 'good', f'리랭커 Recall +{d:.2f}', f'Hybrid 후보 20개 안의 정답 문서를 상위 4개로 끌어올렸습니다.{cost}')
        elif d < -0.02:
            add('rerank', 'warn', f'리랭커 Recall {d:.2f}', f'리랭커가 문장이 비슷한 다른 건 문서를 위로 올렸습니다.{cost}')
        else:
            add('rerank', 'info', '리랭커 효과 미미', f'상위 4개에 드는 정답 문서 수가 거의 같습니다.{cost}')

    # 4. Judge 항목 중 가장 약한 것 (어느 능력이 부족한지)
    for c in [best] if best else []:
        judge = {k: _m(c, 'judge_' + k) for k in JUDGE_ITEMS}
        judge = {k: v for k, v in judge.items() if v is not None}
        if len(judge) == len(JUDGE_ITEMS):
            weakest = min(judge, key=judge.get)
            reason = {'facts': '필요한 사실·수치를 빠뜨리거나 틀립니다. 근거 문서가 문맥에 다 들어오지 않은 경우가 많습니다.',
                      'conditions': '기준일·대상·정정 범위를 잘못 적용합니다. 여러 기록의 변경 내용을 함께 봐야 하는 Advanced 데이터의 특성입니다.',
                      'evidence': '인용한 쪽이 주장을 뒷받침하지 못합니다.',
                      'decision': '답변·추가 확인·보류 판단이 기대 행동과 다릅니다.'}[weakest]
            add('judge_' + weakest, 'warn' if judge[weakest] < 0.6 else 'info',
                f'Judge 가장 낮은 항목: {JUDGE_LABEL[weakest]} {judge[weakest]:.2f}', reason)

    # 5. 답변: 충실도 대 사실
    for c in configs:
        f, facts = _m(c, 'faithfulness'), _m(c, 'judge_facts')
        if f is not None and facts is not None and f - facts > 0.2:
            add('faithfulness', 'warn', f'{c["mode"]}: Faithfulness {f:.2f} > 사실 {facts:.2f}',
                '답은 받은 문맥에 충실하지만 그 문맥에 정답 근거가 부족합니다. 생성보다 검색 쪽 문제입니다.')
            break

    # 6. 입력 토큰 (검색 방식이 문맥 길이를 바꾸는지)
    if naive and rerank and naive.get('input_tokens_sum') and rerank.get('input_tokens_sum'):
        ratio = rerank['input_tokens_sum'] / naive['input_tokens_sum'] - 1
        if abs(ratio) > 0.05:
            add('tokens', 'info', f'입력 토큰 {ratio:+.0%} (Reranker vs Naive)', '최종 청크 수(k=4)는 같지만 고른 청크의 길이가 달라 문맥 크기가 달라졌습니다.')

    # 7. 오류 (발제: 시스템 오류만 셈, 보류·추가 확인은 오류 아님)
    for c in configs:
        if c.get('errors'):
            add('error_rate', 'warn', f'{c["mode"]} 오류 {c["errors"]}건 ({c["error_rate_pct"]:.1f}%)',
                '재시도 후에도 시스템 오류로 끝난 요청입니다. 이 요청의 검색·답변 지표는 평가 불가로 빠지고 평균 응답 시간에서도 제외됩니다.')
    return out


# ── 전체 요약 ────────────────────────────────────────────────

def build_summary(folder, evaluation_dir):
    folder = Path(folder)
    manifest = _load_json(folder / 'manifest.json') if (folder / 'manifest.json').exists() else {}
    references = load_references(evaluation_dir)
    scores = load_scores(folder)
    per_item = [evaluate_item(item, references.get(item['question_id'], {}), scores.get((item['profile'], item['mode'], item['question_id'])))
                for item in load_items(folder)]
    config = manifest.get('config', {})
    configs = []
    for profile in PROFILES:
        for mode in MODES:
            rows = [r for r in per_item if r['profile'] == profile and r['mode'] == mode]
            if rows:
                configs.append({'profile': profile, 'mode': mode, **summarize_config(rows)})
    questions = {p: question_rows(per_item, references, p) for p in PROFILES if any(r['profile'] == p for r in per_item)}
    insights = {p: build_insights([c for c in configs if c['profile'] == p], questions[p]) for p in questions}
    # 원인 분류 집계와 실패 요청 목록 (발제: 실패한 실행도 포함하고 원인을 남김)
    causes = {f'{c["profile"]}/{c["mode"]}': {k: sum(r['cause'] == k for r in per_item if r['profile'] == c['profile'] and r['mode'] == c['mode']) for k in CAUSE_LABEL}
              for c in configs}
    failures = [{'question_id': r['question_id'], 'profile': r['profile'], 'mode': r['mode'], 'error_code': r.get('error_code'),
                 'error_message': r.get('error_message'), 'latency_ms': r.get('latency_ms')} for r in per_item if r['status'] != 'ok']
    return {
        'run': {
            'run_name': manifest.get('run_name', folder.name), 'scope': manifest.get('scope'),
            'metric_eligible': manifest.get('metric_eligible', False), 'selection': config.get('selection'),
            'question_count': len({r['question_id'] for r in per_item}), 'provider': config.get('provider'),
            'model': config.get('model'),
            'settings': {k: config.get(k) for k in ['top_k', 'dense_k', 'bm25_k', 'rrf', 'rerank_k', 'max_output']},
            'answer_scores_available': bool(scores),
            # 채점 모델·프롬프트 기록 (scoring.json, 있을 때만)
            'scoring': _load_json(folder / 'scoring.json') if (folder / 'scoring.json').exists() else None,
        },
        'metric_definitions': METRIC_DEFINITIONS,
        'configs': configs,
        'insights': insights,
        'selection': {p: selection_rationale([c for c in configs if c['profile'] == p]) for p in questions},
        'cost': {'status': 'unavailable', 'reason': '로컬 추론(Ollama)이라 모델 호출 단가가 없습니다. 발제 기준에 따라 비용은 "확인 불가"로 두고 토큰만 비교합니다.'},
        'causes': causes,
        'cause_labels': CAUSE_LABEL,
        'failures': failures,
        'questions': questions,
        'human_reviews': load_reviews(folder),
    }


# 비교표에 함께 적는 정의 (발제: 추가 지표는 적용 대상·입력·계산 기준을 정의해 기록)
METRIC_DEFINITIONS = {
    'hit_at_4': '발제 필수. evidence_groups의 book_id(중복 제거) 중 하나라도 최종 문맥 4개에 있으면 1',
    'recall_at_4': '발제 필수. 찾은 참고 근거 문서 수 / 전체 참고 근거 문서 수 (문서 단위)',
    'page_hit_at_4': '추가. 문서 ID와 페이지가 모두 같아야 인정하는 Hit@4',
    'page_recall_at_4': '추가. (문서 ID, 페이지) 단위 Recall@4',
    'mrr_doc': '추가. 순위가 매겨진 후보 전체에서 첫 참고 근거 문서의 순위 역수 (못 찾으면 0)',
    'mrr_page': '추가. mrr_doc과 같되 페이지까지 일치해야 인정',
    'evidence_coverage': '추가. 근거 묶음 중 최종 문맥이 덮은 비율 (묶음 안 페이지 하나라도 있으면 덮음)',
    'evidence_all_covered': '추가. 근거 묶음을 전부 덮은 문항 비율',
    'duplicate_rate': '추가. 1 - 최종 문맥의 고유 페이지 수 / 청크 수',
    'recall_curve_doc': '추가. 후보 상위 k개(k=1..20)의 문서 단위 Recall 평균',
    'citation_precision_doc': '추가(발제 예시 "인용 정확도"). 답변이 인용한 출처 중 참고 근거 문서와 맞는 비율. 인용이 없으면 해당 없음',
    'citation_precision_page': '추가. 위와 같되 문서 ID와 페이지가 모두 맞아야 인정',
    'latency_p50_ms': '추가. 정상 요청 응답 시간 중앙값',
    'latency_p95_ms': '추가. 정상 요청 응답 시간 95번째 백분위수',
}


def main():
    from backend.app.config import get_settings
    parser = argparse.ArgumentParser(description='Summarize a comparison run for the web view and static report')
    parser.add_argument('--comparison', type=Path, required=True)
    parser.add_argument('--html', action='store_true', help='Also write retrieval_report.html')
    args = parser.parse_args()
    summary = build_summary(args.comparison, get_settings().evaluation_dir)
    write_json(args.comparison / 'retrieval_summary.json', summary)
    print('SUMMARY=' + str(args.comparison / 'retrieval_summary.json'))
    if args.html:
        from .html_report import write_report
        print('REPORT=' + str(write_report(summary, args.comparison / 'retrieval_report.html')))


if __name__ == '__main__':
    main()
