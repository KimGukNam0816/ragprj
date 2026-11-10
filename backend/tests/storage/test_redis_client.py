import pytest

from backend.app.config import Settings
from backend.app.storage import check_redis, mem0_vector_store


def test_redis_disabled_by_default_and_mem0_config_is_valid():
    """Redis와 Mem0 연결 설정이 맞는지 Redis를 띄우지 않고 몇 초 만에 확인하는 자동 테스트"""
    assert check_redis(Settings(_env_file=None, redis_enabled=False))['status'] == 'disabled'
    assert check_redis(Settings(_env_file=None, redis_enabled=True, redis_vector_url=''))['status'] == 'missing'
    cfg = mem0_vector_store(Settings(_env_file=None, redis_vector_url='redis://127.0.0.1:6379', redis_memory_index='idx:test', mem0_embedding_dimensions=1024))
    assert cfg == {'provider': 'redis', 'config': {'redis_url': 'redis://127.0.0.1:6379', 'collection_name': 'idx:test', 'embedding_model_dims': 1024}}
    # Mem0 설정 클래스는 모르는 키를 거부하므로, Redis를 띄우지 않고도 설정 오타를 잡는다.
    RedisDBConfig = pytest.importorskip('mem0.configs.vector_stores.redis').RedisDBConfig
    RedisDBConfig(**cfg['config'])
