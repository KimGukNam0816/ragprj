"""비교 실행 진행 상황을 터미널에 실시간으로 보여 준다 (읽기 전용, 모델 호출 없음).

단계: ① 비교 실행(60문항 × 임베딩 2 × RAG 3) → ② 답변 채점(RAGAS·LLM Judge) → ③ 요약·보고서
구성별 완료 수, 오류, 지금까지의 중간 지표(Hit@4·Recall@4·평균 응답 시간), 예상 남은 시간을 표시한다.

실행: python -X utf8 -m evaluation.retrieval.progress --comparison runtime/5060-check/comparisons/<실행이름>
"""
import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path
from statistics import mean

from .metrics import retrieval_scores
from .summary import MODES, PROFILES, load_references

LABEL = {'naive': 'Naive', 'hybrid': 'Hybrid·RRF', 'hybrid_rerank': 'Hybrid+Reranker'}


def _read(path):
    try:
        return json.loads(Path(path).read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return None  # 쓰는 중인 파일은 다음 갱신 때 다시 읽는다


def snapshot(folder, refs):
    manifest = _read(folder / 'manifest.json') or {}
    expected = manifest.get('expected_requests') or 360
    items = [x for x in (_read(p) for p in sorted((folder / 'items').glob('*.json'))) if x] if (folder / 'items').exists() else []
    scores = list((folder / 'scores').glob('*.json')) if (folder / 'scores').exists() else []
    rows = {}
    for profile in PROFILES:
        for mode in MODES:
            group = [x for x in items if x['profile'] == profile and x['mode'] == mode]
            hits, recalls, latency = [], [], []
            for x in group:
                if x.get('status') == 'ok':
                    latency.append(x['response']['latency_ms'])
                saved = _read(x['result_path']) if x.get('result_path') else None
                ref = refs.get(x['question_id'], {})
                if saved and 'final_context' in saved and ref.get('evidence_groups'):
                    s = retrieval_scores(ref, saved['final_context'])
                    hits.append(s['hit_at_4']); recalls.append(s['recall_at_4'])
            rows[(profile, mode)] = {'done': len(group), 'errors': sum(x.get('status') != 'ok' for x in group),
                                     'hit': mean(hits) if hits else None, 'recall': mean(recalls) if recalls else None,
                                     'latency': mean(latency) / 1000 if latency else None}
    latest = max(items, key=lambda x: Path(x['result_path']).stat().st_mtime if x.get('result_path') and Path(x['result_path']).exists() else 0, default=None)
    return manifest, expected, items, scores, rows, latest


def eta(paths, remaining):
    """최근 20건의 완료 간격으로 남은 시간을 어림한다."""
    times = sorted(p.stat().st_mtime for p in paths)[-21:]
    if len(times) < 2 or remaining <= 0:
        return None
    per_item = (times[-1] - times[0]) / (len(times) - 1)
    return remaining * per_item


def fmt_minutes(seconds):
    return '–' if seconds is None else f'{seconds / 60:.0f}분'


def render(folder, refs):
    manifest, expected, items, scores, rows, latest = snapshot(folder, refs)
    cfg = manifest.get('config', {})
    done = len(items)
    stage = '① 비교 실행' if done < expected else ('② 답변 채점' if len(scores) < expected else '③ 요약·보고서 생성 가능')
    lines = [f'검색 방식 비교 평가 · {folder.name}   ({datetime.now():%H:%M:%S} 갱신)', '',
             f'동일 조건  질문 {len(cfg.get("questions", [])) or "–"}개 · 생성 {cfg.get("model")} · k={cfg.get("top_k")} · 후보 의미{cfg.get("dense_k")}/BM25 {cfg.get("bm25_k")}/리랭커 {cfg.get("rerank_k")} · RRF {cfg.get("rrf")} · 범위 {manifest.get("scope")}',
             f'바꾼 것    임베딩 2(bge-m3, qwen3-embedding) × RAG 3(Naive, Hybrid·RRF, Hybrid+Reranker) = 6구성', '',
             f'현재 단계  {stage}',
             f'  ① 비교 실행  {done:>3}/{expected}  {"■" * int(done / expected * 30):<30}  남은 시간 약 {fmt_minutes(eta([Path(x["result_path"]) for x in items if x.get("result_path") and Path(x["result_path"]).exists()], expected - done))}',
             f'  ② 답변 채점  {len(scores):>3}/{expected}  {"■" * int(len(scores) / expected * 30):<30}  남은 시간 약 {fmt_minutes(eta(scores, expected - len(scores)))}', '',
             f'{"구성":<24}{"완료":>7}{"오류":>6}{"Hit@4":>8}{"Recall@4":>10}{"평균 응답":>10}',
             '-' * 66]
    for profile in PROFILES:
        for mode in MODES:
            r = rows[(profile, mode)]
            f = lambda v, p=2: '–' if v is None else f'{v:.{p}f}'
            lines.append(f'{profile + " / " + LABEL[mode]:<24}{r["done"]:>4}/60{r["errors"]:>6}{f(r["hit"]):>8}{f(r["recall"]):>10}{f(r["latency"], 1) + "s":>10}')
    if latest:
        lines += ['', f'마지막 처리  {latest["profile"]}/{latest["mode"]}/{latest["question_id"]} → {latest.get("status")}']
    lines += ['', '※ 중간 지표는 지금까지 끝난 문항 기준이며 최종 값과 다를 수 있습니다. Ctrl+C로 이 화면만 닫힙니다(실행은 계속).']
    return '\n'.join(lines)


def main():
    from backend.app.config import get_settings
    p = argparse.ArgumentParser(description='Show live progress of a comparison run')
    p.add_argument('--comparison', type=Path, required=True)
    p.add_argument('--interval', type=float, default=10)
    p.add_argument('--once', action='store_true')
    a = p.parse_args()
    refs = load_references(get_settings().evaluation_dir)
    while True:
        text = render(a.comparison, refs)
        if a.once:
            print(text)
            return
        os.system('cls' if os.name == 'nt' else 'clear')
        print(text, flush=True)
        time.sleep(a.interval)


if __name__ == '__main__':
    main()
