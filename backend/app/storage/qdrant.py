"""Qdrant 사용자 기억 저장소.

공개 메서드와 값 검사는 schema.py의 MemoryStore에 있고, 여기는 Qdrant 읽기·쓰기만 둔다(정의서 docs/storage.md 3절).
문서 검색용 Qdrant(retrieval/dense.py)와는 별개의 컬렉션을 쓴다.
"""

import time
from functools import lru_cache

from qdrant_client import QdrantClient, models
from qdrant_client.http.exceptions import ApiException, UnexpectedResponse

from ..config import Settings
from .schema import (
    CATEGORIES,
    HNSW_EF_CONSTRUCTION,
    HNSW_M,
    VECTOR_SIZE,
    MemoryStore,
    build_memory_id,
    check_user_id,
    is_uuid,
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


class QdrantMemoryStore(MemoryStore):
    """사용자 기억 저장소(Qdrant). 기억 1건 = point 1개(벡터 + payload).

    사용 시각 수정은 payload만 바꾸므로 벡터 색인을 다시 만들지 않는다.
    연결·이름은 만들 때 한 번 준비하고 메서드에서 재사용한다.
    """

    db_error_types = (ApiException,)  # 연결 실패·서버 오류 응답의 공통 부모

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
        """설정의 Qdrant 주소로 저장소를 만든다."""
        return cls(get_qdrant(settings))

    def _create_index_if_missing(self) -> bool:
        if self.client.collection_exists(self.collection_name):
            return False
        if not self._create_collection():
            return False
        self._create_payload_indexes()
        return True

    def _save_memory(self, user_id, text, vector, memory_type, category, source) -> str:
        """category를 다시 저장하면 처음 만든 시각·사용 시각은 그대로 둔다(Redis와 같게)."""
        memory_id = build_memory_id(user_id, category)
        now_seconds = time.time()
        payload = _build_payload(
            user_id, text, memory_type, category, source, now_seconds
        )
        if category is not None:
            payload.update(self._load_times_to_keep(memory_id, now_seconds))
        point = models.PointStruct(id=memory_id, vector=list(vector), payload=payload)
        self.client.upsert(self.collection_name, points=[point], wait=True)
        return memory_id

    def _list_memories(self, user_id, kinds) -> list[dict]:
        """고른 종류만 Qdrant 안에서 걸러 최대 100건."""
        kind_filters = [KIND_FILTERS[kind] for kind in kinds]
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

    def _find_similar_notes(self, user_id, question_vector, top_k) -> list[dict]:
        note_filter = models.Filter(
            must=[_build_user_condition(user_id), KIND_FILTERS["note"]]
        )
        response = self.client.query_points(
            self.collection_name,
            query=list(question_vector),
            query_filter=note_filter,
            limit=top_k,
            with_payload=True,
        )
        return [_to_note_record_with_similarity(point) for point in response.points]

    def _update_memory_fields(self, user_id, memory_id, fields, vector) -> bool:
        if not self._is_own_memory(user_id, memory_id):
            return False
        if vector is not None:
            self._replace_vector(memory_id, vector)
        self.client.set_payload(
            self.collection_name, payload=fields, points=[memory_id], wait=True
        )  # 준 칸만 바뀜
        return True

    def _mark_memories_used(self, user_id, memory_ids, used_at_seconds) -> int:
        valid_ids = [memory_id for memory_id in memory_ids if is_uuid(memory_id)]
        if not valid_ids:
            return 0
        own_points = _build_own_points_filter(user_id, valid_ids)
        used_count = self._count_points(own_points)
        self.client.set_payload(
            self.collection_name,
            payload={"last_used_at": used_at_seconds},
            points=models.FilterSelector(filter=own_points),  # 이 사용자의 있는 id만
            wait=True,
        )
        return used_count

    def _delete_memory(self, user_id, memory_id) -> bool:
        if not self._is_own_memory(user_id, memory_id):
            return False
        own_point = _build_own_points_filter(user_id, [memory_id])
        self.client.delete(
            self.collection_name,
            points_selector=models.FilterSelector(filter=own_point),
            wait=True,
        )
        return True

    def _delete_all_memories_of_user(self, user_id) -> int:
        user_points = models.Filter(must=[_build_user_condition(user_id)])
        deleted_count = self._count_points(user_points)
        self.client.delete(
            self.collection_name,
            points_selector=models.FilterSelector(filter=user_points),
            wait=True,
        )
        return deleted_count

    def _is_own_memory(self, user_id: str, memory_id: str) -> bool:
        if not is_uuid(memory_id):  # Qdrant id는 UUID만 허용
            return False
        return self._count_points(_build_own_points_filter(user_id, [memory_id])) > 0

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


def _build_own_points_filter(user_id: str, memory_ids: list[str]) -> models.Filter:
    """이 사용자의 기억 중 memory_ids에 든 것만 고르는 조건(남의 id는 빠짐)."""
    return models.Filter(
        must=[_build_user_condition(user_id), models.HasIdCondition(has_id=memory_ids)]
    )


def _build_payload(user_id, text, memory_type, category, source, now_seconds) -> dict:
    """새 기억의 payload. 사용 시각 시작값 = 저장 시각(저장 직후 '가장 오래 안 쓴 것'으로 지워지지 않게)."""
    payload = {
        "user_id": user_id,
        "memory": text,
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
