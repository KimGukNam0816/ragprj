import os
from functools import lru_cache
from typing import Literal
from pydantic import BaseModel, Field

from ..config import get_settings
from ..observability import observation
from ..storage import build_mem0_vector_store


@lru_cache(maxsize=1)
def memory_client():
    s=get_settings();s.check_inference_policy()
    if s.mem0_llm_provider!='ollama':
        raise ValueError('This smoke service permits local Mem0 inference only')
    os.environ['MEM0_TELEMETRY']='false'
    from mem0 import Memory
    history=s.runtime_dir/'mem0/history.db';history.parent.mkdir(parents=True,exist_ok=True)
    # vector_store: REDIS_ENABLED=true면 Redis, 아니면 기존 Qdrant
    config={'llm':{'provider':'ollama','config':{'model':s.local_mem0_model,'ollama_base_url':s.ollama_base_url,'temperature':0,'max_tokens':1024}},'embedder':{'provider':'ollama','config':{'model':s.mem0_embedding_model,'embedding_dims':s.mem0_embedding_dimensions,'ollama_base_url':s.ollama_base_url}},'vector_store':build_mem0_vector_store(s) if s.redis_enabled else {'provider':'qdrant','config':{'collection_name':'smoke_'+s.qdrant_memory_collection,'embedding_model_dims':s.mem0_embedding_dimensions,'url':s.qdrant_url,'api_key':s.qdrant_api_key.get_secret_value() or None}},'history_db_path':str(history),'custom_fact_extraction_prompt':'사용자가 명시한 답변 언어·길이·표현 선호만 추출하세요. 예시나 업무 금액을 추가하지 마세요. JSON {"facts":["사용자의 선호"]}만 응답하세요. 명시 선호가 없으면 facts를 빈 배열로 반환하세요.'}
    memory=Memory.from_config(config)
    # Use the same guarded local provider as generation/evaluation. The Mem0 Ollama
    # adapter does not forward think=False on this version, so bridge it explicitly.
    from ..providers import Providers
    class Facts(BaseModel):
        facts:list[str]=Field(max_length=5)
    class Action(BaseModel):
        id:str
        text:str
        event:Literal['ADD','UPDATE','DELETE','NONE']
    class Actions(BaseModel):
        memory:list[Action]=Field(max_length=5)
    class LocalMemoryLLM:
        def generate_response(self,messages,response_format=None,tools=None,**kwargs):
            if tools:
                raise ValueError('Tools are not enabled for smoke preference memory')
            provider=Providers(s.model_copy(update={'generation_llm_provider':'ollama','local_llm_model':s.local_mem0_model,'llm_max_output_tokens':1024}))
            with observation('memory_llm',kind='generation',input=messages,model=s.local_mem0_model) as span:
                result=provider.chat(messages,schema=Facts.model_json_schema() if len(messages)>1 else Actions.model_json_schema())
                if span:span.update(output=result['content'],usage_details={k:v for k,v in {'input':result['input_tokens'],'output':result['output_tokens']}.items() if v is not None})
                return result['content']
    class LocalMemoryEmbedding:
        def embed(self,text,memory_action=None):
            provider=Providers(s.model_copy(update={'embedding_model':s.mem0_embedding_model,'embedding_dimensions':s.mem0_embedding_dimensions}))
            return provider.embed([text],'bge')[0]
    memory.llm=LocalMemoryLLM()
    memory.embedding_model=LocalMemoryEmbedding()
    return memory


