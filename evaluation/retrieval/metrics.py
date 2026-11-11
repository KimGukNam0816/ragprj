"""검색 지표 계산.

- retrieval_scores: 발제 필수 지표(문서 단위 Hit@4·Recall@4). quality.py가 사용하며 계산식은 바꾸지 않는다.
- extended_scores: 발제 "도전 실습 A"의 추가 지표. 필수 지표와 섞지 않고 별도 키로 기록한다.

입력 형식
  reference: development.json / additional.json의 한 문항 (evidence_groups 사용)
  final_context: 답변 생성 직전 모델에 넘긴 청크 목록 (result.json의 final_context)
  candidates: 검색 방식이 순위를 매긴 후보 전체 (result.json의 candidates)
  청크는 book_id, page, chunk_id를 가진 dict.
"""


# ── 발제 필수: 문서 단위 Hit@k · Recall@k ─────────────────────

def retrieval_scores(reference, contexts):
    """evidence_groups의 book_id를 중복 없이 모아 상위 k개 청크의 문서 ID와 대조한다 (발제 STEP 6)."""
    gold={x['book_id'] for group in reference.get('evidence_groups',[]) for x in group}
    if not gold:
        return {'hit_at_4':None,'recall_at_4':None,'retrieval_status':'not_applicable'}
    found={x['book_id'] for x in contexts}
    return {'hit_at_4':int(bool(gold&found)),'recall_at_4':len(gold&found)/len(gold),'retrieval_status':'ok'}


# ── 정답 위치 꺼내기 ─────────────────────────────────────────

def gold_books(reference):
    """참고 근거 문서 ID 집합 (발제 기준과 동일)."""
    return {x['book_id'] for group in reference.get('evidence_groups', []) for x in group}


def gold_pages(reference):
    """참고 근거 (문서 ID, 페이지) 집합. 페이지 단위 추가 지표용."""
    return {(x['book_id'], int(x['page'])) for group in reference.get('evidence_groups', []) for x in group}


def doc_key(chunk):
    return chunk['book_id']


def page_key(chunk):
    return (chunk['book_id'], int(chunk['page']))


# ── 추가 지표: 개별 계산 함수 ─────────────────────────────────

def hit_recall(gold, chunks, key):
    """상위 청크들의 key(문서 또는 페이지)를 중복 없이 모아 Hit·Recall 계산. 정답이 없으면 (None, None)."""
    if not gold:
        return None, None
    found = {key(c) for c in chunks} & gold
    return int(bool(found)), len(found) / len(gold)


def first_gold_rank(gold, ranked, key):
    """순위 목록에서 정답이 처음 나오는 청크 순위(1부터). 없으면 None."""
    for rank, chunk in enumerate(ranked, 1):
        if key(chunk) in gold:
            return rank
    return None


def reciprocal_rank(gold, ranked, key):
    """MRR의 문항별 값: 1 / 첫 정답 순위. 정답을 못 찾으면 0, 정답 목록이 없으면 None."""
    if not gold:
        return None
    rank = first_gold_rank(gold, ranked, key)
    return 1 / rank if rank else 0.0


def recall_curve(gold, ranked, key, max_k=20):
    """k=1..max_k 각각의 Recall. 후보가 k개보다 적으면 마지막 값을 유지한다(더 가져올 청크가 없음)."""
    if not gold:
        return None
    curve, found = [], set()
    for k in range(1, max_k + 1):
        if k <= len(ranked) and key(ranked[k - 1]) in gold:
            found.add(key(ranked[k - 1]))
        curve.append(len(found) / len(gold))
    return curve


def evidence_coverage(reference, chunks):
    """근거 묶음 중 몇 개를 덮었나. 묶음 안 위치(문서+페이지) 중 하나라도 청크에 있으면 그 묶음은 덮은 것.

    여러 근거를 모두 봐야 답할 수 있는 문항에서 검색 방식 차이를 보기 위한 지표.
    """
    groups = [g for g in reference.get('evidence_groups', []) if g]
    if not groups:
        return {'evidence_groups': 0, 'evidence_covered': None, 'evidence_coverage': None, 'evidence_all_covered': None}
    pages = {page_key(c) for c in chunks}
    covered = sum(any((x['book_id'], int(x['page'])) in pages for x in g) for g in groups)
    return {'evidence_groups': len(groups), 'evidence_covered': covered,
            'evidence_coverage': covered / len(groups), 'evidence_all_covered': int(covered == len(groups))}


def duplicate_rate(chunks):
    """상위 청크 중 같은 페이지에서 나온 청크 비율: 1 - 고유 페이지 수 / 청크 수. 청크가 없으면 None."""
    if not chunks:
        return None
    return 1 - len({page_key(c) for c in chunks}) / len(chunks)


def citation_precision(reference, citations):
    """답변이 인용한 출처 중 정답 근거와 맞는 비율 (발제 도전 실습 A의 "인용 정확도").

    인용이 없거나 정답 근거가 없으면 해당 없음(None). 문서 단위와 페이지 단위를 함께 돌려준다.
    """
    books, pages = gold_books(reference), gold_pages(reference)
    if not citations or not books:
        return {'citation_precision_doc': None, 'citation_precision_page': None, 'citation_status': 'not_applicable'}
    doc_hits = sum(c['book_id'] in books for c in citations)
    page_hits = sum((c['book_id'], int(c['page'])) in pages for c in citations)
    return {'citation_precision_doc': doc_hits / len(citations), 'citation_precision_page': page_hits / len(citations),
            'citation_status': 'ok'}


# ── 추가 지표 묶음 ───────────────────────────────────────────

def extended_scores(reference, final_context, candidates, max_k=20):
    """문항 하나의 추가 지표를 한 번에 계산한다.

    - 페이지 단위 Hit/Recall·근거 커버리지·중복률: 최종 문맥(final_context) 기준.
    - MRR·첫 정답 순위·Recall@k 곡선: 순위가 있는 후보 전체(candidates) 기준. 없으면 최종 문맥으로 대신한다.
    """
    books, pages = gold_books(reference), gold_pages(reference)
    if not books:
        return {'extended_status': 'not_applicable'}
    ranked = candidates or final_context
    page_hit, page_recall = hit_recall(pages, final_context, page_key)
    result = {
        'extended_status': 'ok',
        'page_hit_at_4': page_hit,
        'page_recall_at_4': page_recall,
        'first_gold_rank_doc': first_gold_rank(books, ranked, doc_key),
        'first_gold_rank_page': first_gold_rank(pages, ranked, page_key),
        'mrr_doc': reciprocal_rank(books, ranked, doc_key),
        'mrr_page': reciprocal_rank(pages, ranked, page_key),
        'recall_curve_doc': recall_curve(books, ranked, doc_key, max_k),
        'duplicate_rate': duplicate_rate(final_context),
        'candidate_count': len(ranked),
    }
    result.update(evidence_coverage(reference, final_context))
    return result
