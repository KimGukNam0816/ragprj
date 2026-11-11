"""Qdrant 사용자 기억 저장소.

QdrantMemoryStore의 메서드는 redis_client.py의 RedisMemoryStore와 같다(정의서 docs/storage.md 3절).
문서 검색용 Qdrant(retrieval/dense.py)와는 별개의 컬렉션을 쓴다.
"""

import time
from functools import lru_cache

from qdrant_client import QdrantClient, models
from qdrant_client.http.exceptions import UnexpectedResponse

from ..config import Settings
from .schema import (
    CATEGORIES,
    HNSW_EF_CONSTRUCTION,
    HNSW_M,
    VECTOR_SIZE,
    build_memory_id,
    build_update_fields,
    check_record,
    check_user_id,
    check_vector,
    is_uuid,
    parse_kinds,
    parse_top_k,
    to_epoch_seconds,
    to_iso_time,
)

COLLECTION_NAME = "shipping_memory_v2"
MAX_LIST_SIZE = 100  # 기획상 사용자당 최대 40건
KEYWORD_FIELDS = ("type", "category", "source")
TIME_FIELDS = ("created_at", "updated_at", "last_used_at")
KIND_FILTERS = {
    "preference": models.FieldCondition(
        key="type", match=models.MatchValue(value="preference")
    ),
    "category": models.FieldCondition(
        key="category", match=models.MatchAny(any=list(CATEGORIES))
    ),
    "note": models.Filter(
        must=[
            models.FieldCondition(key="type", match=models.MatchValue(value="context")),
            models.IsEmptyCondition(
                is_empty=models.PayloadField(key="category")
            ),  # category가 없는 context
        ]
    ),
}


def get_qdrant(settings: Settings) -> QdrantClient:
    api_key = settings.qdrant_api_key.get_secret_value()
    return _create_client(
        settings.qdrant_url, api_key, int(settings.connection_timeout_seconds)
    )


class QdrantMemoryStore:
    """사용자 기억 저장소(Qdrant). 기억 1건 = point 1개(벡터 + payload).

    last_used_at 수정은 payload만 바꾸므로 벡터 색인을 다시 만들지 않는다.
    연결·이름은 만들 때 한 번 준비하고 메서드에서 재사용한다.
    """

    def __init__(
        self,
        client: QdrantClient,
        collection_name: str = COLLECTION_NAME,
        hnsw_m: int = HNSW_M,
        hnsw_ef_construction: int = HNSW_EF_CONSTRUCTION,
    ):
        self.client = client
        self.collection_name = collection_name
        self.hnsw_m = hnsw_m  # 부하 테스트에서 바꿔 가며 비교
        self.hnsw_ef_construction = hnsw_ef_construction

    @classmethod
    def from_settings(cls, settings: Settings) -> "QdrantMemoryStore":
        return cls(get_qdrant(settings))

    def ensure(self) -> bool:
        """컬렉션이 없으면 만들고 True, 이미 있으면 건드리지 않고 False."""
        if self.client.collection_exists(self.collection_name):
            return False
        if not self._create_collection():
            return False
        self._create_payload_indexes()
        return True

    def save_memory(
        self,
        user_id,
        memory,
        vector,
        memory_type,
        category=None,
        source="auto_extracted",
    ) -> str:
        """새 기억을 저장하고 id를 돌려준다. category는 같은 항목(user_id, category)이 있으면 그 자리를 덮어쓴다."""
        check_record(user_id, memory, vector, memory_type, category, source)
        memory_id = build_memory_id(user_id, category)
        now_seconds = time.time()
        payload = _build_payload(
            user_id, memory, memory_type, category, source, now_seconds
        )
        if category is not None:
            payload.update(self._load_times_to_keep(memory_id, now_seconds))
        point = models.PointStruct(id=memory_id, vector=list(vector), payload=payload)
        self.client.upsert(self.collection_name, points=[point], wait=True)
        return memory_id

    def list_memories(self, user_id, kinds=None) -> list[dict]:
        """사용자의 기억(최대 100건). kinds로 고른 종류만 Qdrant 안에서 거른다(기본: 모든 종류)."""
        kind_filters = [KIND_FILTERS[kind] for kind in parse_kinds(kinds)]
        query_filter = models.Filter(
            must=[_build_user_condition(user_id)], should=kind_filters
        )  # should = 하나라도 맞으면
        points, _ = self.client.scroll(
            self.collection_name,
            scroll_filter=query_filter,
            limit=MAX_LIST_SIZE,
            with_payload=True,
            with_vectors=False,
        )
        return [_to_memory_record(point) for point in points]

    def search_notes(self, user_id, vector, top_k=3, min_similarity=0.5) -> list[dict]:
        """질문 벡터와 비슷한 note만 비슷한 순으로 최대 top_k건. similarity(0~1)가 min_similarity 이상인 것만."""
        check_vector(vector)
        note_filter = models.Filter(
            must=[_build_user_condition(user_id), KIND_FILTERS["note"]]
        )
        response = self.client.query_points(
            self.collection_name,
            query=list(vector),
            query_filter=note_filter,
            limit=parse_top_k(top_k),
            with_payload=True,
        )
        notes = [_to_note_record_with_similarity(point) for point in response.points]
        return [
            note for note in notes if note["similarity"] >= min_similarity
        ]  # 하한 비교는 Redis와 같게 여기서

    def update_memory(self, memory_id, memory=None, vector=None, source=None) -> bool:
        """준 값만 바꾸고 나머지는 그대로 둔다. memory를 바꾸면 vector도 같이 줘야 한다. 없는 id면 False.

        type·category·user_id는 바꿀 수 없다(바꾸려면 지우고 새로 저장).
        """
        payload = build_update_fields(memory, vector, source)
        if not self._memory_exists(memory_id):
            return False
        payload["updated_at"] = time.time()
        if vector is not None:
            self._replace_vector(memory_id, vector)
        self.client.set_payload(
            self.collection_name, payload=payload, points=[memory_id], wait=True
        )  # 준 칸만 바뀜
        return True

    def touch(self, user_id, memory_ids, used_at=None) -> int:
        """이 사용자 기억의 사용 시각만 used_at(ISO 글자, 기본: 지금)으로 바꾸고 바꾼 개수를 돌려준다.

        없는 id와 다른 사용자의 id는 건너뛴다.
        """
        user_condition = _build_user_condition(user_id)
        valid_ids = [memory_id for memory_id in memory_ids if is_uuid(memory_id)]
        if not valid_ids:
            return 0
        own_points = models.Filter(
            must=[user_condition, models.HasIdCondition(has_id=valid_ids)]
        )
        touched_count = self._count_points(own_points)
        self.client.set_payload(
            self.collection_name,
            payload={"last_used_at": to_epoch_seconds(used_at)},
            points=models.FilterSelector(
                filter=own_points
            ),  # 이 사용자의 있는 id만 골라짐
            wait=True,
        )
        return touched_count

    def delete_memory(self, memory_id) -> bool:
        """1건 삭제. 없는 id면 False."""
        if not self._memory_exists(memory_id):
            return False
        self.client.delete(
            self.collection_name,
            points_selector=models.PointIdsList(points=[memory_id]),
            wait=True,
        )
        return True

    def purge_user(self, user_id) -> int:
        """사용자의 기억 전부를 지우고 지운 개수를 돌려준다(퇴사 처리). 복구할 수 없다."""
        user_points = models.Filter(must=[_build_user_condition(user_id)])
        deleted_count = self._count_points(user_points)
        self.client.delete(
            self.collection_name,
            points_selector=models.FilterSelector(filter=user_points),
            wait=True,
        )
        return deleted_count

    def _memory_exists(self, memory_id: str) -> bool:
        if not is_uuid(memory_id):  # Qdrant id는 UUID만 허용
            return False
        found_points = self.client.retrieve(
            self.collection_name, [memory_id], with_payload=False
        )
        return bool(found_points)

    def _count_points(self, query_filter: models.Filter) -> int:
        return self.client.count(
            self.collection_name, count_filter=query_filter, exact=True
        ).count

    def _replace_vector(self, memory_id: str, vector) -> None:
        new_vector = models.PointVectors(id=memory_id, vector=list(vector))
        self.client.update_vectors(self.collection_name, points=[new_vector], wait=True)

    def _load_times_to_keep(self, memory_id: str, now_seconds: float) -> dict:
        """같은 category를 다시 저장할 때 유지할 시각. 처음 만든 시각·사용 시각은 그대로, 수정 시각만 지금(Redis와 같게)."""
        found_points = self.client.retrieve(
            self.collection_name,
            [memory_id],
            with_payload=["created_at", "last_used_at"],
        )
        if not found_points:
            return {}
        return {**found_points[0].payload, "updated_at": now_seconds}

    def _create_collection(self) -> bool:
        try:
            self.client.create_collection(
                self.collection_name,
                vectors_config=models.VectorParams(
                    size=VECTOR_SIZE, distance=models.Distance.COSINE
                ),
                hnsw_config=models.HnswConfigDiff(
                    m=self.hnsw_m, ef_construct=self.hnsw_ef_construction
                ),
            )
        except UnexpectedResponse:
            if self.client.collection_exists(
                self.collection_name
            ):  # 다른 프로세스가 방금 만든 경우만 정상
                return False
            raise
        return True

    def _create_payload_indexes(self) -> None:
        # 사용자별로 나눠 저장·검색하도록 user_id를 tenant 인덱스로 지정(사용자가 많을 때 Qdrant 권장 방식)
        tenant_index = models.KeywordIndexParams(
            type=models.KeywordIndexType.KEYWORD, is_tenant=True
        )
        self.client.create_payload_index(self.collection_name, "user_id", tenant_index)
        for field_name in KEYWORD_FIELDS:
            self.client.create_payload_index(
                self.collection_name, field_name, models.PayloadSchemaType.KEYWORD
            )
        for field_name in TIME_FIELDS:
            self.client.create_payload_index(
                self.collection_name, field_name, models.PayloadSchemaType.FLOAT
            )


@lru_cache(maxsize=4)  # 같은 주소면 연결 객체를 재사용
def _create_client(qdrant_url: str, api_key: str, timeout_seconds: int) -> QdrantClient:
    return QdrantClient(
        url=qdrant_url, api_key=api_key or None, timeout=timeout_seconds
    )  # REST 사용(gRPC 아님)


def _build_user_condition(user_id: str) -> models.FieldCondition:
    check_user_id(user_id)
    return models.FieldCondition(key="user_id", match=models.MatchValue(value=user_id))


def _build_payload(user_id, memory, memory_type, category, source, now_seconds) -> dict:
    """새 기억의 payload. 사용 시각 시작값 = 저장 시각(저장 직후 '가장 오래 안 쓴 것'으로 지워지지 않게)."""
    payload = {
        "user_id": user_id,
        "memory": memory,
        "type": memory_type,
        "source": source,
        "created_at": now_seconds,
        "last_used_at": now_seconds,
    }
    if category is not None:
        payload["category"] = category
    return payload


def _to_memory_record(point) -> dict:
    """Qdrant point → 정의서 모양 dict. 없는 칸(note의 category 등)은 None."""
    payload = point.payload
    return {
        "id": str(point.id),
        "user_id": payload.get("user_id"),
        "memory": payload.get("memory"),
        "type": payload.get("type"),
        "category": payload.get("category"),
        "source": payload.get("source"),
        "created_at": to_iso_time(payload.get("created_at")),
        "updated_at": to_iso_time(payload.get("updated_at")),
        "last_used_at": to_iso_time(payload.get("last_used_at")),
    }


def _to_note_record_with_similarity(point) -> dict:
    return {
        **_to_memory_record(point),
        "similarity": point.score,
    }  # Qdrant cosine 점수는 그대로 유사도
