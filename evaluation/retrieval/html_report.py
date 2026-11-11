"""retrieval_summary.json을 브라우저로 여는 정적 HTML 보고서로 만든다.

웹 화면(/evaluation) 합의 전에도 발표·원문 대조에 쓸 수 있게, 외부 라이브러리와 서버 없이
파일 하나로 열리도록 CSS와 SVG 차트를 모두 안에 넣는다.

화면 순서: ① 비교표(발제 필수) → ② 추가 지표 → ③ 근거 차트 → ④ 문항별 원문 대조 → ⑤ 지표 정의
"""
import html
from pathlib import Path

MODE_LABEL = {'naive': 'Naive', 'hybrid': 'Hybrid · RRF', 'hybrid_rerank': 'Hybrid + Reranker'}
MODE_COLOR = {'naive': 'var(--m-naive)', 'hybrid': 'var(--m-hybrid)', 'hybrid_rerank': 'var(--m-rerank)'}
PROFILE_LABEL = {'bge': 'bge-m3', 'qwen': 'qwen3-embedding:0.6b'}
JUDGE_LABEL = {'facts': '사실', 'conditions': '조건', 'evidence': '근거', 'decision': '판단'}


def esc(value):
    return html.escape('' if value is None else str(value))


# ── 숫자 표시 ────────────────────────────────────────────────

def fmt(value, kind='ratio'):
    if value is None:
        return '–'
    if kind == 'ratio':
        return f'{value:.2f}'
    if kind == 'pct':
        return f'{value:.1f}%'
    if kind == 'ms':
        return f'{value / 1000:.1f}s'
    if kind == 'int':
        return f'{value:,}'
    return esc(value)


def delta(value, base, higher_is_better=True):
    """Naive 대비 증감 화살표. 값이 없으면 빈칸."""
    if value is None or base is None or value == base:
        return ''
    better = (value > base) == higher_is_better
    return f' <span class="{"up" if better else "down"}">{"▲" if value > base else "▼"}</span>'


def stat_cell(s, base=None):
    """평균 + 발제 요구 건수(n · 해당 없음 · 평가 불가)."""
    if not s:
        return '<td class="n">–</td>'
    sub = f'n={s["n"]}'
    if s.get('na'):
        sub += f' · 해당없음 {s["na"]}'
    if s.get('unscorable'):
        sub += f' · 평가불가 {s["unscorable"]}'
    return f'<td class="n">{fmt(s["mean"])}{delta(s["mean"], (base or {}).get("mean"))}<div class="sub">{sub}</div></td>'


# ── 표 ───────────────────────────────────────────────────────

def required_table(configs, has_answers):
    base = next((c for c in configs if c['mode'] == 'naive'), {})
    head = '<th>방식</th><th>Hit@4</th><th>Recall@4</th>'
    if has_answers:
        head += '<th>Faithfulness</th>' + ''.join(f'<th>Judge {JUDGE_LABEL[k]}</th>' for k in JUDGE_LABEL)
    else:
        head += '<th>Faithfulness · Judge</th>'
    head += '<th>평균 응답 시간</th><th>오류율</th><th>입력 토큰</th><th>출력 토큰</th>'
    rows = []
    for c in configs:
        cells = f'<td><span class="dot" style="background:{MODE_COLOR[c["mode"]]}"></span>{MODE_LABEL[c["mode"]]}</td>'
        cells += stat_cell(c['hit_at_4'], base.get('hit_at_4')) + stat_cell(c['recall_at_4'], base.get('recall_at_4'))
        if has_answers:
            cells += stat_cell(c.get('faithfulness'), base.get('faithfulness'))
            cells += ''.join(stat_cell(c.get('judge_' + k), base.get('judge_' + k)) for k in JUDGE_LABEL)
        else:
            cells += '<td class="n">채점 보류</td>'
        cells += (f'<td class="n">{fmt(c["latency_mean_ms"], "ms")}{delta(c["latency_mean_ms"], base.get("latency_mean_ms"), False)}'
                  f'<div class="sub">정상 {c["latency_ok_n"]}건</div></td>')
        cells += f'<td class="n">{fmt(c["error_rate_pct"], "pct")}<div class="sub">{c["errors"]}/{c["requests"]}건 · 실패 시간 기록 {c.get("latency_failed_recorded_n", 0)}</div></td>'
        for key in ['input_tokens', 'output_tokens']:
            cells += f'<td class="n">{fmt(c[key + "_sum"], "int")}<div class="sub">확인 {c[key + "_known_requests"]}/{c["requests"]}</div></td>'
        rows.append(f'<tr>{cells}</tr>')
    return f'<div class="tbl"><table><thead><tr>{head}</tr></thead><tbody>{"".join(rows)}</tbody></table></div>'


def extended_table(configs):
    base = next((c for c in configs if c['mode'] == 'naive'), {})
    cols = [('page_hit_at_4', '페이지 Hit@4'), ('page_recall_at_4', '페이지 Recall@4'), ('mrr_doc', 'MRR (문서)'),
            ('mrr_page', 'MRR (페이지)'), ('evidence_coverage', '근거 커버리지'), ('evidence_all_covered', '근거 전부 찾음')]
    head = '<th>방식</th>' + ''.join(f'<th>{label}</th>' for _, label in cols) + '<th>중복률</th><th>p50 / p95</th>'
    rows = []
    for c in configs:
        cells = f'<td><span class="dot" style="background:{MODE_COLOR[c["mode"]]}"></span>{MODE_LABEL[c["mode"]]}</td>'
        cells += ''.join(stat_cell(c.get(key), base.get(key)) for key, _ in cols)
        dup = c.get('duplicate_rate') or {}
        cells += f'<td class="n">{fmt(dup.get("mean"))}{delta(dup.get("mean"), (base.get("duplicate_rate") or {}).get("mean"), False)}</td>'
        cells += f'<td class="n">{fmt(c.get("latency_p50_ms"), "ms")} / {fmt(c.get("latency_p95_ms"), "ms")}</td>'
        rows.append(f'<tr>{cells}</tr>')
    return f'<div class="tbl"><table><thead><tr>{head}</tr></thead><tbody>{"".join(rows)}</tbody></table></div>'


# ── 차트 (SVG) ───────────────────────────────────────────────

def recall_curve_svg(configs, current_k):
    width, height, left, right, top, bottom = 420, 220, 40, 12, 12, 30
    plot_w, plot_h = width - left - right, height - top - bottom
    curves = [(c['mode'], c['recall_curve_doc']) for c in configs if c.get('recall_curve_doc')]
    if not curves:
        return '<p class="note">후보 순위 정보가 없어 곡선을 그릴 수 없습니다.</p>'
    max_k = len(curves[0][1])
    x = lambda k: left + (k - 1) / (max_k - 1) * plot_w
    y = lambda v: top + (1 - v) * plot_h
    parts = [f'<svg viewBox="0 0 {width} {height}" width="100%" role="img" aria-label="Recall@k 곡선">']
    for v in [0, 0.25, 0.5, 0.75, 1]:
        parts.append(f'<line x1="{left}" y1="{y(v):.1f}" x2="{width - right}" y2="{y(v):.1f}" class="grid"/>'
                     f'<text x="{left - 6}" y="{y(v) + 3:.1f}" text-anchor="end">{v:g}</text>')
    ticks = {1, 5, 10, 15, max_k}
    if current_k:
        # 현재 k와 너무 가까운 기본 눈금은 빼서 숫자가 겹치지 않게
        ticks = {t for t in ticks if abs(t - current_k) > 1 or t in (1, max_k)} | {current_k}
    for k in sorted(ticks):
        if 1 <= k <= max_k:
            parts.append(f'<text x="{x(k):.1f}" y="{height - 10}" text-anchor="middle">{k}</text>')
    if current_k and 1 <= current_k <= max_k:
        parts.append(f'<line x1="{x(current_k):.1f}" y1="{top}" x2="{x(current_k):.1f}" y2="{top + plot_h}" class="mark"/>')
    for mode, curve in curves:
        points = ' '.join(f'{x(k + 1):.1f},{y(v):.1f}' for k, v in enumerate(curve))
        parts.append(f'<polyline points="{points}" fill="none" stroke="{MODE_COLOR[mode]}" stroke-width="2.2"/>')
    parts.append('</svg>')
    return ''.join(parts)


def tradeoff_svg(configs):
    """세로: Recall@4, 가로: 평균 응답 시간. 왼쪽 위가 좋음."""
    points = [(c['mode'], c['latency_mean_ms'], c['recall_at_4']['mean']) for c in configs
              if c.get('latency_mean_ms') is not None and c['recall_at_4']['mean'] is not None]
    if not points:
        return '<p class="note">응답 시간 또는 Recall 값이 없어 그릴 수 없습니다.</p>'
    width, height, left, right, top, bottom = 420, 220, 44, 20, 14, 34
    xs, ys = [p[1] for p in points], [p[2] for p in points]
    x_lo, x_hi = min(xs) * 0.9, max(xs) * 1.1 or 1
    y_lo, y_hi = max(0, min(ys) - 0.1), min(1, max(ys) + 0.1)
    if y_hi == y_lo:
        y_hi = y_lo + 0.1
    x = lambda v: left + (v - x_lo) / ((x_hi - x_lo) or 1) * (width - left - right)
    y = lambda v: top + (1 - (v - y_lo) / (y_hi - y_lo)) * (height - top - bottom)
    parts = [f'<svg viewBox="0 0 {width} {height}" width="100%" role="img" aria-label="정확도 대 응답 시간">',
             f'<line x1="{left}" y1="{height - bottom}" x2="{width - right}" y2="{height - bottom}" class="grid"/>',
             f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height - bottom}" class="grid"/>',
             f'<text x="{left - 6}" y="{y(y_hi) + 4:.1f}" text-anchor="end">{y_hi:.2f}</text>',
             f'<text x="{left - 6}" y="{y(y_lo):.1f}" text-anchor="end">{y_lo:.2f}</text>',
             f'<text x="{left}" y="{height - 14}">{x_lo / 1000:.1f}s</text>',
             f'<text x="{width - right}" y="{height - 14}" text-anchor="end">{x_hi / 1000:.1f}s</text>',
             f'<text x="{(left + width) / 2:.0f}" y="{height - 2}" text-anchor="middle">평균 응답 시간 →</text>']
    last_label_y = None
    for mode, lat, rec in sorted(points, key=lambda p: y(p[2])):
        label_y = y(rec) + 4
        if last_label_y is not None and label_y - last_label_y < 13:
            label_y = last_label_y + 13  # 점이 가까우면 라벨만 아래로 밀어 겹치지 않게
        last_label_y = label_y
        # 오른쪽 끝 점은 라벨을 점 왼쪽에 붙여 잘리지 않게
        right_side = x(lat) > width * 0.6
        parts.append(f'<circle cx="{x(lat):.1f}" cy="{y(rec):.1f}" r="6" fill="{MODE_COLOR[mode]}"/>'
                     f'<text x="{x(lat) + (-9 if right_side else 9):.1f}" y="{label_y:.1f}" class="lbl" text-anchor="{"end" if right_side else "start"}">{MODE_LABEL[mode]}</text>')
    parts.append('</svg>')
    return ''.join(parts)


# ── 문항별 원문 대조 ─────────────────────────────────────────

def _context_list(chunks, gold):
    if not chunks:
        return '<p class="note">모델에 전달된 문맥 없음</p>'
    items = []
    for c in chunks:
        mark = ' class="gold"' if c.get('book_id') in gold else ''
        items.append(f'<li{mark}><b>{esc(c.get("book_id"))} p.{esc(c.get("page"))}</b> {esc(c.get("text"))}</li>')
    return f'<ol class="ctx">{"".join(items)}</ol>'


def question_section(questions, reviews):
    review_map = {}
    for r in reviews:
        review_map.setdefault(r.get('question_id'), []).append(r)
    change_label = {'improved': ('개선', 'up'), 'worsened': ('악화', 'down'), 'same': ('동일', ''), None: ('–', '')}
    blocks = []
    for q in questions:
        gold = set(q['gold_books'])
        label, cls = change_label[q['change_vs_naive']]
        cells = ''
        for mode in MODE_LABEL:
            m = q['modes'].get(mode)
            if not m:
                cells += '<span class="cell c0">–</span>'
                continue
            rec, rank = m.get('recall_at_4'), m.get('first_gold_rank_doc')
            tone = 'c0' if rec is None else 'c1' if rec == 1 else 'c2' if rec > 0 else 'c3'
            cells += f'<span class="cell {tone}" title="첫 정답 순위 {rank or "없음"}">{fmt(rec)}</span>'
        detail = [f'<div class="ref"><b>질문</b> {esc(q["question"])}<br><b>기대 행동</b> {esc(q["expected_action"])} · <b>참고 답변</b> {esc(q["reference_answer"])}</div>']
        evidence = ''.join(f'<li><b>{esc(x["book_id"])} p.{esc(x["page"])}</b> {esc(x["quote"])}</li>' for g in q['evidence'] for x in g)
        detail.append(f'<h5>정답 근거</h5><ul class="ctx">{evidence or "<li>등록된 근거 없음 (검색 지표 해당 없음)</li>"}</ul>')
        for mode, label_mode in MODE_LABEL.items():
            m = q['modes'].get(mode)
            if not m:
                continue
            judge = ' · '.join(f'{JUDGE_LABEL[k]} {fmt(m.get("judge_" + k), "raw")}' for k in JUDGE_LABEL if ('judge_' + k + '_status') in m)
            reasons = ''.join(f'<li>{JUDGE_LABEL[k]}: {esc(m.get("judge_" + k + "_reason"))}</li>' for k in JUDGE_LABEL if m.get('judge_' + k + '_reason'))
            detail.append(
                f'<div class="mode"><h5><span class="dot" style="background:{MODE_COLOR[mode]}"></span>{label_mode}'
                f' <span class="note">상태 {esc(m.get("answer_status") or m.get("status"))} · Recall@4 {fmt(m.get("recall_at_4"))}'
                f' · 첫 정답 {esc(m.get("first_gold_rank_doc") or "없음")}등 · request {esc(m.get("request_id"))}</span></h5>'
                f'<p class="ans">{esc(m.get("answer"))}</p>'
                + (f'<p class="note">Judge {judge}</p><ul class="reasons">{reasons}</ul>' if judge else '')
                + _context_list(m.get('final_context'), gold) + '</div>')
        for r in review_map.get(q['question_id'], []):
            detail.append(f'<p class="review">사람 확인 · {esc(r.get("profile"))}/{esc(r.get("mode"))} {esc(r.get("metric"))}: '
                          f'{esc(r.get("original"))} → {esc(r.get("revised"))} · {esc(r.get("reason"))}</p>')
        blocks.append(
            f'<details class="q"><summary><span class="qid">{esc(q["question_id"])}</span>'
            f'<span class="qtext">{esc((q["question"] or "")[:70])}</span>'
            f'<span class="cells">{cells}</span><span class="chg {cls}">{label}</span></summary>'
            f'<div class="qbody">{"".join(detail)}</div></details>')
    return ''.join(blocks)


# ── 페이지 조립 ──────────────────────────────────────────────

CSS = """
:root{--bg:#f5f7f8;--surface:#fff;--ink:#14212b;--muted:#566673;--line:#d8e0e5;--accent:#0d6e78;--accent-soft:#e1f0f1;
--amber:#b86e00;--amber-soft:#fbf0dc;--good:#1f7a4a;--bad:#b23a3a;--m-naive:#7a8a96;--m-hybrid:#2f7fb8;--m-rerank:#0d6e78;
--font:"Malgun Gothic","Apple SD Gothic Neo",system-ui,sans-serif;--mono:ui-monospace,Consolas,monospace}
@media (prefers-color-scheme:dark){:root{--bg:#0f171d;--surface:#16222a;--ink:#e3eaee;--muted:#9aabb6;--line:#2a3a45;--accent:#4fb8c2;
--accent-soft:#163238;--amber:#e3a447;--amber-soft:#3a2c15;--good:#5cc08a;--bad:#e27a7a;--m-naive:#8d9ca7;--m-hybrid:#6aaee0;--m-rerank:#4fb8c2;color-scheme:dark}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.6 var(--font);padding:28px max(16px,3vw) 60px}
main{max-width:1100px;margin:0 auto;display:flex;flex-direction:column;gap:28px}h1{font-size:24px;margin:0}h2{font-size:18px;margin:0 0 10px}
h3{font-size:15px;margin:18px 0 8px;color:var(--accent)}h5{margin:12px 0 4px;font-size:13px}.note{color:var(--muted);font-size:12.5px}
.meta{display:flex;flex-wrap:wrap;gap:6px 16px;color:var(--muted);font-size:13px}.badge{background:var(--amber-soft);color:var(--amber);font-weight:600;padding:2px 10px;border-radius:999px}
.tbl{overflow-x:auto;border:1px solid var(--line);border-radius:6px;background:var(--surface)}table{border-collapse:collapse;width:100%;min-width:720px}
th,td{padding:8px 10px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}th{font-size:12px;color:var(--muted);background:var(--bg)}
tr:last-child td{border-bottom:0}td:first-child{white-space:nowrap}td.n{font-family:var(--mono);font-variant-numeric:tabular-nums;white-space:nowrap}.sub{font-size:11px;color:var(--muted)}
.dot{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:6px}.up{color:var(--good)}.down{color:var(--bad)}
.charts{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,560px));gap:12px}.panel{background:var(--surface);border:1px solid var(--line);border-radius:6px;padding:12px 14px}
svg text{font:10px var(--mono);fill:var(--muted)}svg .lbl{fill:var(--ink);font-size:11px}svg .grid{stroke:var(--line)}svg .mark{stroke:var(--muted);stroke-dasharray:3 3}
.legend{display:flex;gap:14px;flex-wrap:wrap;font-size:12px;color:var(--muted)}
details.q{background:var(--surface);border:1px solid var(--line);border-radius:6px;margin-bottom:6px}
details.q summary{display:grid;grid-template-columns:90px 1fr auto 44px;gap:10px;align-items:center;padding:8px 12px;cursor:pointer;list-style:none}
details.q summary::-webkit-details-marker{display:none}.qid{font-family:var(--mono);font-size:12px}.qtext{overflow:hidden;text-overflow:ellipsis;white-space:nowrap;min-width:0}
.cells{display:flex;gap:4px}.cell{font-family:var(--mono);font-size:11.5px;width:46px;text-align:center;border-radius:3px;padding:1px 0}
.c1{background:var(--accent);color:var(--surface)}.c2{background:var(--accent-soft);color:var(--accent)}.c3{background:var(--amber-soft);color:var(--amber)}.c0{background:var(--line);color:var(--muted)}
.chg{font-size:12px;font-weight:600}.qbody{padding:4px 14px 14px;border-top:1px solid var(--line)}.ref{font-size:13px;margin-top:8px}
.mode{border-top:1px dashed var(--line);margin-top:10px}.ans{margin:4px 0;white-space:pre-wrap}.ctx{margin:4px 0;padding-left:20px;font-size:12.5px}
.ctx li{margin-bottom:4px}.ctx li.gold{background:var(--accent-soft)}.reasons{margin:0;padding-left:20px;font-size:12px;color:var(--muted)}
.review{background:var(--amber-soft);padding:6px 10px;border-radius:4px;font-size:12.5px}dl{display:grid;grid-template-columns:auto 1fr;gap:4px 14px;font-size:13px}dt{font-family:var(--mono)}dd{margin:0}
.cond{background:var(--surface);border:1px solid var(--line);border-radius:6px;padding:10px 14px;margin-top:12px;font-size:13px}.cond ul{margin:4px 0 0;padding-left:18px}
.hold{background:var(--amber-soft);color:var(--ink);border-radius:6px;padding:10px 14px;font-size:13px;margin-top:10px}
.ins{margin:0;padding-left:0;list-style:none;display:flex;flex-direction:column;gap:6px;font-size:13px}.ins li{background:var(--surface);border:1px solid var(--line);border-left:4px solid var(--muted);border-radius:6px;padding:8px 12px}
.ins li.good{border-left-color:var(--good)}.ins li.warn{border-left-color:var(--amber)}.sel{background:var(--accent-soft);border-radius:6px;padding:10px 14px;font-size:13px;margin:10px 0}
@media (max-width:640px){details.q summary{grid-template-columns:70px 1fr;}.cells,.chg{grid-column:1/-1}}
"""


# 발제 STEP 6·7의 필수 지표 정의 (보고서에는 이것만 싣는다)
REQUIRED_DEFINITIONS = {
    'Hit@4': '참고 근거 문서(evidence_groups의 book_id, 중복 제거) 중 하나라도 최종 문맥 4개에 있으면 1. 문항별 값을 평균',
    'Recall@4': '최종 문맥 4개에서 찾은 참고 근거 문서 수 ÷ 전체 참고 근거 문서 수. 같은 문서의 청크가 여러 개여도 한 번만 셈',
    'RAGAS Faithfulness': '답변의 주장 중 실제 최종 문맥이 뒷받침하는 비율. 사실 주장 없이 되묻기만 한 답은 해당 없음',
    'LLM Judge 사실·조건·근거·판단': '참고 답변·필수 사실·기대 행동과 대조해 항목별 0/1. 항목별 충족률 = 1점 수 ÷ 평가 대상 수 (네 항목을 합산하지 않음)',
    '응답 시간': '요청 수신부터 응답까지(검색·리랭크·생성·재시도 포함). 정상 처리 요청의 평균과 건수. 실패 요청 시간은 따로 기록',
    '오류율': '재시도 후에도 시스템 오류로 끝난 요청 수 ÷ 전체 요청 수 × 100. 보류·추가 확인은 오류가 아님',
    '입출력 토큰': '요청별로 확인된 토큰의 방식별 합계와 확인한 요청 수. 확인 못 한 값은 0으로 채우지 않음',
    '비용': '단가를 확인할 수 있을 때만 계산. 로컬 추론이라 확인 불가',
    '해당 없음 · 평가 불가': '평균에서 제외하고 건수를 따로 표시. 0점은 평균에 포함',
}


def _conditions_box(run):
    s = run.get('settings') or {}
    return ('<div class="cond"><b>평가 조건</b><ul>'
            f'<li><b>동일:</b> 문항 {run["question_count"]}개 · 생성 {esc(run.get("model"))} · 최종 청크 k={esc(s.get("top_k"))} · '
            f'후보 의미 {esc(s.get("dense_k"))} / BM25 {esc(s.get("bm25_k"))} / 리랭커 {esc(s.get("rerank_k"))} · RRF {esc(s.get("rrf"))} · 출력 최대 {esc(s.get("max_output"))} · 같은 프롬프트</li>'
            '<li><b>바꾼 것:</b> 임베딩 2개(bge-m3, qwen3-embedding:0.6b) × 검색 방식 3개(Naive, Hybrid·RRF, Hybrid+Reranker)</li>'
            '<li><b>정답:</b> 제공된 평가 파일(development.json · additional.json)의 참고 근거·참고 답변과 코드·채점 모델로 대조</li></ul></div>')


def _cause_table(causes, labels, profile):
    keys = [k for k in labels if k != 'ok' and any(causes.get(f'{profile}/{m}', {}).get(k) for m in MODE_LABEL)]
    if not keys:
        return '<p class="note">분류할 실패가 없습니다.</p>'
    head = '<th>원인</th>' + ''.join(f'<th>{MODE_LABEL[m]}</th>' for m in MODE_LABEL)
    rows = ''.join(f'<tr><td>{esc(labels[k])}</td>' + ''.join(f'<td class="n">{causes.get(f"{profile}/{m}", {}).get(k, 0)}</td>' for m in MODE_LABEL) + '</tr>' for k in keys)
    return f'<div class="tbl"><table><thead><tr>{head}</tr></thead><tbody>{rows}</tbody></table></div>'


def render(summary):
    run = summary['run']
    has_answers = run.get('answer_scores_available')
    scope = run.get('scope') or 'unknown'
    badge = '' if run.get('metric_eligible') else f'<span class="badge">{esc(scope)} · 정식 지표 아님</span>'
    body = [f'<header><h1>검색 방식 3종 비교 보고서</h1><div class="meta"><span>실행 {esc(run["run_name"])}</span>'
            f'<span>생성 {esc(run.get("provider"))} / {esc(run.get("model"))}</span>{badge}</div>'
            + _conditions_box(run)
            + ('' if has_answers else '<p class="hold">답변 채점(RAGAS Faithfulness · LLM Judge)은 <b>보류</b> 중입니다. 로컬 채점 모델이 점수를 제대로 매기지 못해 결과를 제외했습니다. 아래 수치는 검색·운영 지표입니다.</p>')
            + '</header>']
    configs_by_profile = {}
    for c in summary['configs']:
        configs_by_profile.setdefault(c['profile'], []).append(c)
    cost = summary.get('cost') or {}
    for profile, configs in configs_by_profile.items():
        insights = ''.join(f'<li class="{esc(i["tone"])}"><b>{esc(i["title"])}</b> {esc(i["text"])}</li>' for i in summary.get('insights', {}).get(profile, []))
        selection = (summary.get('selection') or {}).get(profile)
        body.append(f'<section><h2>임베딩 {esc(PROFILE_LABEL.get(profile, profile))}</h2>'
                    '<h3>① 비교표 (발제 필수 지표)</h3>'
                    '<p class="note">평균 아래: 평가 대상 수(n) · 해당 없음 · 평가 불가. 해당 없음·평가 불가는 평균에서 제외, 0점은 포함. 화살표는 Naive 대비.'
                    f' 비용: {esc(cost.get("reason", "확인 불가"))}</p>'
                    + required_table(configs, has_answers)
                    + '<h3>② 결과 해석과 선정 근거</h3>'
                    f'<ul class="ins">{insights}</ul>'
                    + (f'<p class="sel"><b>선정 후보:</b> {esc(selection["text"])}</p>' if selection else '')
                    + '<div class="charts"><div class="panel"><b>정확도(Recall@4) 대 평균 응답 시간</b><p class="note">왼쪽 위일수록 빠르고 정확. 품질이 비슷하면 시간·토큰으로 고른다는 발제 기준을 그림으로</p>'
                    f'{tradeoff_svg(configs)}</div></div>'
                    '<div class="legend">' + ''.join(f'<span><span class="dot" style="background:{MODE_COLOR[m]}"></span>{l}</span>' for m, l in MODE_LABEL.items()) + '</div>'
                    '<h3>③ 실패 원인 분류 (문항 수)</h3><p class="note">저장된 검색 후보·최종 문맥으로 분류. 정답 문서를 못 찾은 경우와 후보에는 있었으나 4개 안에 못 든 경우를 구분합니다.</p>'
                    + _cause_table(summary.get('causes', {}), summary.get('cause_labels', {}), profile)
                    + '<h3>④ 질문별 결과와 원문 대조</h3><p class="note">칸 = 문항별 Recall@4 (Naive · Hybrid · +Reranker). 나빠진 문항이 위. 행을 누르면 정답 근거와 실제 문맥(정답 문서 강조), 답변'
                    + (', 채점 이유' if has_answers else '') + '를 나란히 봅니다.</p>'
                    + question_section(summary['questions'].get(profile, []), summary.get('human_reviews', [])) + '</section>')
    failures = summary.get('failures') or []
    if failures:
        rows = ''.join(f'<tr><td>{esc(f["profile"])} / {esc(MODE_LABEL.get(f["mode"], f["mode"]))}</td><td>{esc(f["question_id"])}</td><td>{esc(f.get("error_code"))}</td>'
                       f'<td>{esc(f.get("error_message"))}</td><td class="n">{fmt(f.get("latency_ms"), "ms")}</td></tr>' for f in failures)
        body.append('<section><h2>⑤ 실패 요청 (발제: 실패한 실행도 포함)</h2><div class="tbl"><table><thead><tr><th>구성</th><th>문항</th><th>오류</th><th>메시지</th><th>걸린 시간</th></tr></thead>'
                    f'<tbody>{rows}</tbody></table></div><p class="note">실패 요청의 검색·답변 지표는 평가 불가이며 평균 응답 시간에서 제외합니다.</p></section>')
    defs = ''.join(f'<dt>{esc(k)}</dt><dd>{esc(v)}</dd>' for k, v in REQUIRED_DEFINITIONS.items())
    body.append(f'<section><h2>지표 정의 (발제 STEP 6·7)</h2><dl>{defs}</dl></section>')
    return (f'<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
            f'<title>검색 비교 보고서 · {esc(run["run_name"])}</title><style>{CSS}</style></head><body><main>{"".join(body)}</main></body></html>')


def write_report(summary, path):
    path = Path(path)
    path.write_text(render(summary), encoding='utf-8')
    return path
