"""정김기보: Mem0 검색 결과를 프롬프트용 표현 선호 문장으로 변환한다.

`contracts.MemoryResult`는 Mem0 버전별 응답을 그대로 보존한다. 프롬프트에는
원본 JSON(id·hash·score·created_at·metadata)을 넣지 않고 사용자가 밝힌
표현 선호 문장만 넣는다는 계약을 이 모듈이 담당한다.

이 모듈은 순수 변환만 수행한다. Mem0·Qdrant·모델·네트워크를 호출하지 않고
설정을 읽지도 않으므로 단위 테스트에서 그대로 실행할 수 있다.
"""
import math
import re
from typing import Any

from ..contracts import MemoryResult
from .records import rows as _records

# builder.prepare_context 와 pipeline 이 공유하는 '기억 미사용' 표시값.
NO_PREFERENCE = '사용하지 않음'

# 기억 본문에서 제거할 제어문자. 줄바꿈·탭은 아래에서 공백으로 접는다.
_CONTROL = re.compile(r'[\x00-\x08\x0b-\x1f\x7f-\x9f]')
# 모델 chat template 경계를 흉내 내는 토큰.
_CHAT_BOUNDARY = re.compile(r'<\|[^|>]{0,64}\|>|\[/?INST\]|<<\s*/?SYS\s*>>', re.IGNORECASE)
_WHITESPACE = re.compile(r'\s+')
_REDACTED = '[제거됨]'

# RAG_PROMPT 의 human 메시지 구획 라벨. 기억이 이 라벨을 재현해 가짜 구획을
# 만들지 못하도록 콜론을 분리한다.
_SECTION_LABELS = ('질문:', 'PDF 검색 자료:', '사용자 표현 선호')


def _text(record: dict[str, Any]) -> str:
    for key in ('memory', 'text', 'content'):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return ''


def _score(record: dict[str, Any]) -> float | None:
    """점수를 제공하지 않는 Mem0 설정·고정 응답은 None으로 둔다."""
    value = record.get('score')
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def sanitize(text: str, max_chars: int) -> str:
    """기억 한 건을 한 줄 평문으로 만든다.

    줄 구조를 모두 접기 때문에 기억 안에서 새 프롬프트 구획이나 역할 줄을
    만들 수 없다. 내용 자체를 판단하지는 않으며, 지시문 무시 책임은 계속
    RAG_SYSTEM 과 DEFENSE_ADDITION 이 진다.
    """
    text = _CONTROL.sub(' ', text)
    text = _CHAT_BOUNDARY.sub(_REDACTED, text)
    text = _WHITESPACE.sub(' ', text).strip()
    for label in _SECTION_LABELS:
        text = text.replace(label, label.rstrip(':') + ' ')
    text = _WHITESPACE.sub(' ', text).strip()
    if max_chars > 0 and len(text) > max_chars:
        text = text[:max_chars].rstrip() + '…'
    return text


def estimate_tokens(text: str, tokenizer=None) -> int:
    """토크나이저가 있으면 그대로 세고, 없으면 넉넉한 쪽으로 어림한다.

    한글 1자를 1토큰으로 보는 어림이라 실제보다 크게 나온다. 예산을 넘기는
    쪽보다 기억을 한 건 덜 넣는 쪽이 안전하다.
    """
    if tokenizer is not None:
        encoded = tokenizer.encode(text)
        return len(getattr(encoded, 'ids', encoded))
    wide = sum(1 for ch in text if ord(ch) > 127)
    return wide + math.ceil((len(text) - wide) / 4)


def select_preferences(s, result: MemoryResult | None, *, tokenizer=None) -> dict[str, Any]:
    """프롬프트에 넣을 선호 문장과 그 선별 근거를 함께 돌려준다.

    반환값의 kept/dropped 는 실행 기록과 trace 에 남겨 어떤 기억이 왜
    빠졌는지 나중에 확인하기 위한 것이다.
    """
    threshold = getattr(s, 'memory_min_relevance_score', 0.0) or 0.0
    max_chars = getattr(s, 'memory_preference_max_chars', 200)
    budget = getattr(s, 'memory_context_max_tokens', 0) or 0
    limit = getattr(s, 'memory_top_k', 0) or 0

    kept: list[str] = []
    dropped: list[dict[str, Any]] = []
    seen: set[str] = set()
    used = 0
    for record in _records(result):
        raw = _text(record)
        identity = {'id': record.get('id'), 'score': _score(record)}
        if not raw:
            dropped.append({**identity, 'reason': 'empty'})
            continue
        score = _score(record)
        # 점수를 주지 않는 응답까지 버리지 않는다. 측정값이 있을 때만 거른다.
        if score is not None and score < threshold:
            dropped.append({**identity, 'reason': 'below_threshold'})
            continue
        text = sanitize(raw, max_chars)
        if not text:
            dropped.append({**identity, 'reason': 'empty_after_sanitize'})
            continue
        key = _WHITESPACE.sub(' ', text).strip().casefold()
        if key in seen:
            dropped.append({**identity, 'reason': 'duplicate'})
            continue
        if limit and len(kept) >= limit:
            dropped.append({**identity, 'reason': 'top_k'})
            continue
        cost = estimate_tokens('- ' + text + '\n', tokenizer)
        if budget and used + cost > budget:
            dropped.append({**identity, 'reason': 'token_budget'})
            continue
        seen.add(key)
        kept.append(text)
        used += cost
    return {'preferences': kept, 'dropped': dropped, 'estimated_tokens': used,
            'token_budget': budget, 'min_relevance_score': threshold}


def format_preferences(s, result: MemoryResult | None, *, tokenizer=None) -> str:
    """프롬프트의 {memory} 자리에 넣을 문자열. 남는 선호가 없으면 미사용 표시."""
    return render(select_preferences(s, result, tokenizer=tokenizer)['preferences'])


def render(preferences: list[str]) -> str:
    return '\n'.join('- ' + p for p in preferences) if preferences else NO_PREFERENCE
