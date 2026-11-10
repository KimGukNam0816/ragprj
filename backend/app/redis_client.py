"""Redis 접근 코드는 모두 이 파일에 둔다. redis 패키지는 memory 묶음에만 있어서 함수 안에서 import한다."""
from functools import lru_cache

from .config import Settings


@lru_cache(maxsize=4)  # 같은 주소·타임아웃이면 연결 객체를 새로 만들지 않고 재사용
def _client(url: str, timeout: float):
    import redis  # memory 묶음을 안 깐 환경에서도 서버가 import 오류 없이 뜨게 여기서 import
    return redis.Redis.from_url(
        url,
        decode_responses=True,  # 응답을 bytes 대신 str로 받음
        socket_connect_timeout=timeout,  # 서버가 없을 때 무한 대기하지 않게
        socket_timeout=timeout,
    )


def get_redis(s: Settings):
    """다 같이 쓰는 Redis 연결. redis-py가 안에 연결 풀을 두고 재사용한다."""
    return _client(s.redis_vector_url.get_secret_value(), s.connection_timeout_seconds)  # SecretStr에서 실제 주소를 꺼냄


def check_redis(s: Settings) -> dict:
    """Redis 연결과 Mem0에 필요한 검색 기능(FT.*)을 확인한다.
    주소에 비밀번호가 들어갈 수 있어서 메시지에 주소를 넣지 않는다."""
    if not s.redis_enabled:
        return {'status': 'disabled', 'message': '비활성'}
    if not s.redis_vector_url.get_secret_value():
        return {'status': 'missing', 'message': '설정 필요: REDIS_VECTOR_URL'}
    try:
        import redis
    except ImportError:  # redis 패키지가 설치 안 됨
        return {'status': 'missing', 'message': 'uv sync --extra memory 필요'}
    client = get_redis(s)
    try:
        client.ping()  # 1단계: 서버에 연결되는지
    except redis.RedisError:
        return {'status': 'error', 'message': '연결 실패. Redis 실행 상태·주소를 확인하세요.'}
    try:
        indexes = client.execute_command('FT._LIST')  # 2단계: 검색 기능 확인. 일반 Redis면 unknown command 오류
    except redis.ResponseError:
        return {'status': 'error', 'message': '검색 기능 없음. Redis Stack 이미지인지 확인하세요.'}
    return {'status': 'ok', 'message': '연결·검색 기능 확인', 'indexes': sorted(indexes)}


def mem0_vector_store(s: Settings) -> dict:
    """Mem0의 'redis' 저장소 설정. 저장 구조는 Mem0가 정한다:
    키 mem0:<인덱스 이름>:<id>, 벡터는 FLAT·cosine, 필터는 user_id/agent_id/run_id만 가능."""
    return {'provider': 'redis', 'config': {
        'redis_url': s.redis_vector_url.get_secret_value(),
        'collection_name': s.redis_memory_index,  # Redis 인덱스 이름이자 키 접두사(mem0:<이름>)
        'embedding_model_dims': s.mem0_embedding_dimensions,  # 임베딩 모델 차원(bge-m3=1024)과 같아야 함
    }}
