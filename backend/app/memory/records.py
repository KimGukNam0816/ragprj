"""Mem0 응답 행을 읽는 공통 도우미. 외부 연결 없음."""
from typing import Any

from ..contracts import MemoryResult

# Mem0가 직접 관리하는 필드. 커스텀 metadata와 구분한다.
CORE_FIELDS = frozenset({'id', 'memory', 'hash', 'score', 'created_at', 'updated_at', 'user_id'})


def rows(result: MemoryResult | None) -> list[dict[str, Any]]:
    """Mem0는 버전에 따라 {'results': [...]} 또는 리스트를 돌려준다."""
    if result is None:
        return []
    if isinstance(result, dict):
        found = result.get('results')
        if found is None:
            # 단일 기억을 그대로 돌려주는 호출 경로도 허용한다.
            found = [result] if result.get('memory') or result.get('text') else []
    else:
        found = result
    if not isinstance(found, list):
        return []
    return [row for row in found if isinstance(row, dict)]


def field(row: dict[str, Any], name: str, default=None):
    """커스텀 metadata는 최상위로 승격되기도 하고 metadata 안에 남기도 한다."""
    if name in row:
        return row[name]
    meta = row.get('metadata')
    if isinstance(meta, dict):
        return meta.get(name, default)
    return default
