import json
from backend.app.config import get_settings
from backend.app.reranker import get_reranker
from backend.app.artifacts import write_json

if __name__=='__main__':
    s=get_settings();model=get_reranker(s)
    scores=model.score('부산–LA 일반 건화물의 대상 컨테이너는 무엇인가요?',['일반 건화물 대상 장비는 20GP와 40GP입니다.','오늘의 점심 메뉴는 김치찌개입니다.'])
    result={'model':s.reranker_model,'device':model.device,'scores':scores,'relevant_above_irrelevant':scores[0]>scores[1],'metric_eligible':False}
    write_json(s.runtime_dir/'checks/reranker.json',result);print(json.dumps(result))
    raise SystemExit(0 if result['relevant_above_irrelevant'] else 1)
