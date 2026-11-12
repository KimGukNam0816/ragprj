"""Redis 접근 코드를 모아 둔 파일.

- 연결: get_redis, check_redis

redis 패키지는 memory 묶음(uv sync --extra memory)에만 있어서, 패키지 없이도 서버가 뜨도록 함수 안에서 import한다.
"""

from functools import lru_cache

from ..config import Settings


@lru_cache(maxsize=4)  # 같은 주소·타임아웃이면 연결 객체를 새로 만들지 않고 재사용
def _create_client(redis_url: str, timeout_seconds: float):
    import redis

    return redis.Redis.from_url(
        redis_url,
        decode_responses=True,  # 응답을 bytes 대신 str로 받음
        socket_connect_timeout=timeout_seconds,  # 서버가 없을 때 무한 대기하지 않게
        socket_timeout=timeout_seconds,
    )


def get_redis(settings: Settings):
    """다 같이 쓰는 Redis 연결. redis-py가 안에 연결 풀을 두고 재사용한다."""
    redis_url = (
        settings.redis_vector_url.get_secret_value()
    )  # SecretStr에서 실제 주소를 꺼냄
    return _create_client(redis_url, settings.connection_timeout_seconds)


def check_redis(settings: Settings) -> dict:
    """Redis 연결과 검색 기능(FT.*)을 확인한다. 주소에 비밀번호가 있을 수 있어 메시지에 주소를 넣지 않는다."""
    if not settings.semantic_cache_enabled:
        return _build_check_result("disabled", "비활성")
    if not settings.redis_vector_url.get_secret_value():
        return _build_check_result("missing", "설정 필요: REDIS_VECTOR_URL")
    try:
        import redis
    except ImportError:
        return _build_check_result("missing", "uv sync --extra memory 필요")
    return _check_connection_and_search(get_redis(settings), redis)


def _check_connection_and_search(client, redis_module) -> dict:
    """1단계 연결, 2단계 검색 기능(FT.*)을 확인한다."""
    try:
        client.ping()
    except redis_module.RedisError:
        return _build_check_result(
            "error", "연결 실패. Redis 실행 상태·주소를 확인하세요."
        )
    try:
        index_names = client.execute_command(
            "FT._LIST"
        )  # 일반 Redis면 unknown command 오류
    except redis_module.ResponseError:
        return _build_check_result(
            "error", "검색 기능 없음. Redis Stack 이미지인지 확인하세요."
        )
    return {
        **_build_check_result("ok", "연결·검색 기능 확인"),
        "indexes": sorted(index_names),
    }


def _build_check_result(status: str, message: str) -> dict:
    return {"status": status, "message": message}
