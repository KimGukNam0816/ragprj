"""질문·답변 시맨틱 캐시(Redis Stack, redisvl SemanticCache).

비슷한 질문이 들어오면 문서 검색과 LLM 호출을 건너뛰고 저장된 답을 돌려준다.
redis·redisvl 패키지는 memory 묶음(uv sync --extra memory)에만 있어서 함수 안에서 import한다.
"""

import json
import re
from collections.abc import Callable

from ..config import Settings
from ..providers import Providers

CACHE_NAME = "shipping_answer_cache"
SCOPE_FIELD = "scope"  # 임베딩 프로필·검색 모드·인덱스 지문이 같은 답만 돌려준다
NUMBER_TOKEN_PATTERN = re.compile(
    r"\d+(?:[.,:/-]\d+)*"
)  # 50, 2028-05-20, 09:00, 1,000 같은 값
NEAREST_SEARCH_DISTANCE = (
    1.0  # 코사인 거리 0~2. 1.0 안쪽이면 "비슷한 편"이라 miss 기록용으로 충분
)
RESPONSE_FIELDS_NOT_CACHED = {
    "request_id",
    "trace_id",
    "latency_ms",
    "warnings",
    "cache_hit",
}

# (주소, 임계값, TTL) → 준비가 끝난 캐시. 프로세스당 1개만 만들어 재사용
_caches_by_target: dict[tuple[str, float, int], object] = {}


def build_cache_scope(profile: str, mode: str, index_fingerprint: str) -> str:
    """캐시 항목을 묶는 범위 값. 인덱스를 다시 만들면 지문이 바뀌어 옛 답이 안 나온다."""
    return f"{profile}:{mode}:{index_fingerprint[:12]}"


def find_cached_answer(
    settings: Settings, question: str, question_vector: list[float], scope: str
) -> dict | None:
    """같은 범위에서 질문 벡터와 가장 가까운 저장 답. 저장된 것이 없으면 None, Redis 실패는 예외.

    돌려주는 dict: {"is_hit": 임계값 안이고 숫자·날짜도 같으면 True, "response": 저장했던 응답 dict,
    "question": 저장 당시 질문, "distance": 코사인 거리, "miss_reason": None | "distance" | "numbers_differ"}.
    임계값 밖이어도 거리를 돌려줘 miss 이유를 기록할 수 있다.
    """
    return _find_in_cache(get_answer_cache(settings), question, question_vector, scope)


def save_answer(
    settings: Settings,
    question: str,
    question_vector: list[float],
    scope: str,
    response: dict,
) -> None:
    """응답을 캐시에 넣는다. 돌아오면 Redis 반영이 끝난 것이다."""
    _store_in_cache(
        get_answer_cache(settings), question, question_vector, scope, response
    )


def get_answer_cache(settings: Settings):
    """프로세스당 하나만 만들어 재사용한다. 처음 부를 때 인덱스가 없으면 만든다."""
    target = (
        settings.redis_vector_url.get_secret_value(),
        settings.semantic_cache_distance_threshold,
        settings.semantic_cache_ttl_seconds,
    )
    if target not in _caches_by_target:
        _caches_by_target[target] = _build_cache(
            *target, _build_question_embedder(settings)
        )
    return _caches_by_target[target]


def _build_cache(
    redis_url: str,
    distance_threshold: float,
    ttl_seconds: int,
    embed_question: Callable[[str], list[float]],
    name: str = CACHE_NAME,
):
    """redisvl 캐시 객체. 벡터는 밖에서 넣지만 차원을 알아내려고 임베딩 함수가 필요하다."""
    from redisvl.extensions.cache.llm import SemanticCache
    from redisvl.utils.vectorize import CustomVectorizer

    return SemanticCache(
        name=name,
        redis_url=redis_url,
        distance_threshold=distance_threshold,
        ttl=ttl_seconds,
        vectorizer=CustomVectorizer(embed=embed_question),
        filterable_fields=[{"name": SCOPE_FIELD, "type": "tag"}],
    )


def _build_question_embedder(settings: Settings) -> Callable[[str], list[float]]:
    provider = Providers(settings)

    def embed_question(text: str) -> list[float]:
        return provider.embed([text], "bge", query=True)[0]

    return embed_question


def _find_in_cache(
    cache, question: str, question_vector: list[float], scope: str
) -> dict | None:
    from redisvl.query.filter import Tag

    nearest = cache.check(
        vector=question_vector,
        filter_expression=Tag(SCOPE_FIELD) == scope,
        distance_threshold=NEAREST_SEARCH_DISTANCE,  # 임계값 밖이어도 가장 가까운 것을 받아 거리를 기록
    )
    if not nearest:
        return None
    distance = float(nearest[0]["vector_distance"])
    cached_question = nearest[0]["prompt"]
    miss_reason = _choose_miss_reason(
        distance, cache.distance_threshold, question, cached_question
    )
    return {
        "is_hit": miss_reason is None,
        "response": json.loads(nearest[0]["response"]),
        "question": cached_question,
        "distance": distance,
        "miss_reason": miss_reason,
    }


def _choose_miss_reason(distance, threshold, question, cached_question) -> str | None:
    """hit이면 None. 숫자·날짜만 다른 질문(50개 vs 70개)은 벡터가 거의 같아서 따로 걸러낸다."""
    if distance > threshold:
        return "distance"
    if not _have_same_numbers(question, cached_question):
        return "numbers_differ"
    return None


def _have_same_numbers(question: str, cached_question: str) -> bool:
    """두 질문에 나오는 숫자·날짜 토큰이 (개수까지) 같은지."""
    return sorted(NUMBER_TOKEN_PATTERN.findall(question)) == sorted(
        NUMBER_TOKEN_PATTERN.findall(cached_question)
    )


def _store_in_cache(cache, question, question_vector, scope, response: dict) -> None:
    cacheable = {
        key: value
        for key, value in response.items()
        if key not in RESPONSE_FIELDS_NOT_CACHED
    }
    cache.store(
        prompt=question,
        response=json.dumps(cacheable, ensure_ascii=False),
        vector=question_vector,
        filters={SCOPE_FIELD: scope},
    )
