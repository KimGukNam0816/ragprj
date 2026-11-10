from ..config import get_settings
from ..contracts import MemoryResult
from ..observability import observation
from .mem0_client import memory_client


def search_memory(question: str, user_id: str) -> MemoryResult:
    s=get_settings()
    with observation('memory_search',input={'question':question,'user_id':user_id}) as span:
        results=memory_client().search(question,user_id=user_id,limit=s.memory_top_k)
        if span:span.update(output=results)
        return results


def add_memory(text: str, user_id: str) -> MemoryResult:
    with observation('memory_add',input={'user_id':user_id,'explicit_preference':text}) as span:
        result=memory_client().add([{'role':'user','content':text}],user_id=user_id,metadata={'scope':'smoke','explicit':True})
        if span:span.update(output=result)
        return result
