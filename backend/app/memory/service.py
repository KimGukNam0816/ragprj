from ..config import get_settings
from ..contracts import MemoryResult
from ..observability import observation
from ..schemas import ServiceError
from .mem0_client import memory_client


def search_memory(question: str, user_id: str) -> MemoryResult:
    s=get_settings()
    with observation('memory_search',input={'question':question,'user_id':user_id}) as span:
        results=memory_client().search(question,user_id=user_id,limit=s.memory_top_k)
        if span:span.update(output=results)
        return results


def add_memory(text: str, user_id: str) -> MemoryResult:
    s=get_settings()
    # 읽기 경로가 고정 응답을 쓰는 동안 쓰기만 Mem0로 새지 않게 한다.
    # dummy는 저장소가 아니라 고정 fixture이므로 저장 요청 자체를 거절한다.
    if s.memory_backend!='mem0':
        raise ServiceError('memory_backend_readonly','Dummy preferences are a fixed fixture; switch MEMORY_BACKEND to mem0 to store preferences',422)
    with observation('memory_add',input={'user_id':user_id,'explicit_preference':text}) as span:
        result=memory_client().add([{'role':'user','content':text}],user_id=user_id,metadata={'scope':'smoke','explicit':True})
        if span:span.update(output=result)
        return result
