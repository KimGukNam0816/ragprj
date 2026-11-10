from contextlib import contextmanager
from functools import lru_cache

from ..config import get_settings


@lru_cache
def langfuse_client():
    s = get_settings()
    if not (s.langfuse_enabled and s.langfuse_base_url and s.langfuse_public_key and s.langfuse_secret_key.get_secret_value()):
        return None
    from langfuse import Langfuse
    return Langfuse(public_key=s.langfuse_public_key, secret_key=s.langfuse_secret_key.get_secret_value(), base_url=s.langfuse_base_url)


@contextmanager
def observation(name: str, *, kind='span', input=None, metadata=None, model=None):
    client = langfuse_client()
    if client is None:
        yield None
    else:
        with client.start_as_current_observation(name=name, as_type=kind, input=input, metadata=metadata, model=model) as span:
            yield span
