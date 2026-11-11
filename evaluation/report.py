"""기존 비교표 집계 (팀 공통): runs.jsonl + scores.jsonl → summary.json · summary.csv

발제 RAG Comparison Table의 필수 열을 구성(임베딩 × 검색 방식)별 한 줄로 만든다.
더 자세한 집계(해당 없음·평가 불가 건수, 원인 분류, 웹·HTML 보고서)는 evaluation/retrieval/summary.py가 맡는다.
"""
import csv
import json
from pathlib import Path
from statistics import mean

from backend.app.artifacts import read_jsonl, write_json


def report(folder: Path):
    runs=list(read_jsonl(folder/'runs.jsonl'))                                                       # 비교 실행 결과 360건
    scores=list(read_jsonl(folder/'scores.jsonl')) if (folder/'scores.jsonl').exists() else []      # 채점 결과(있을 때만)
    output=[]
    for profile in ['bge','qwen']:
        for mode in ['naive','hybrid','hybrid_rerank']:
            group=[x for x in runs if x['profile']==profile and x['mode']==mode]
            sg=[x for x in scores if x['profile']==profile and x['mode']==mode]

            # ── 운영: 요청 수 · 오류 수 · 오류율 (시스템 오류만, 보류·되묻기는 오류 아님) ──
            row={'profile':profile,'mode':mode,'scope':'smoke','metric_eligible':False,'requests':len(group),'errors':sum(x['status']!='ok' for x in group)}
            row['error_rate']=row['errors']/len(group) if group else None

            # ── 운영: 응답 시간 = 정상 요청의 평균과 건수 ──
            latency=[x['response']['latency_ms'] for x in group if x['status']=='ok']
            row.update(latency_mean_ms=mean(latency) if latency else None,latency_valid_n=len(latency))

            # ── 운영: 입출력 토큰 = 시도별로 확인된 값만 합산 + 확인한 요청 수 (모르면 0으로 채우지 않음) ──
            for key in ['input_tokens','output_tokens']:
                values=[]
                for x in group:
                    if not x.get('result_path'):continue
                    saved=json.loads(Path(x['result_path']).read_text(encoding='utf-8'))
                    attempts=saved.get('generation',{}).get('attempts',[]) or saved.get('error',{}).get('attempts',[])
                    known=[a.get(key) for a in attempts if a.get(key) is not None]
                    if known:values.append(sum(known))
                row[key+'_sum']=sum(values) if values else None
                row[key+'_known_requests']=len(values)

            # ── 검색·답변: Hit@4 · Recall@4 · Faithfulness 평균 (None = 해당 없음·평가 불가는 평균에서 제외) ──
            for metric in ['hit_at_4','recall_at_4','faithfulness']:
                valid=[x[metric] for x in sg if x.get(metric) is not None]
                row[metric]=mean(valid) if valid else None
                row[metric+'_valid_n']=len(valid)
            row['faithfulness_na_n']=sum(x.get('faithfulness_status')=='not_applicable' for x in sg)
            row['faithfulness_unscorable_n']=sum(x.get('faithfulness_status')=='unscorable' for x in sg)
            row['judge_unscorable_n']=sum(x.get('judge_status')=='unscorable' for x in sg)

            # ── 답변: Judge 4항목 = 항목별 충족률 (1점 수 ÷ 평가 대상 수, 합산하지 않음) ──
            for metric in ['facts','conditions','evidence','decision']:
                valid=[x['judge'][metric]['score'] for x in sg if x.get('judge') and x['judge'].get(metric,{}).get('score') is not None]
                row['judge_'+metric]=mean(valid) if valid else None
                row['judge_'+metric+'_valid_n']=len(valid)
            output.append(row)

    # ── 저장: 비용은 로컬 추론이라 계산하지 않음 ──
    write_json(folder/'summary.json',{'scope':'smoke','metric_eligible':False,'rows':output,'cost':None,'cost_reason':'All inference is local; cloud storage/tracing usage billed separately'})
    with (folder/'summary.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(output[0]));w.writeheader();w.writerows(output)
    return folder/'summary.csv'
