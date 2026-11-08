import csv
import json
from pathlib import Path
from statistics import mean

from backend.app.artifacts import read_jsonl, write_json


def report(folder: Path):
    runs=list(read_jsonl(folder/'runs.jsonl'))
    scores=list(read_jsonl(folder/'scores.jsonl')) if (folder/'scores.jsonl').exists() else []
    output=[]
    for profile in ['bge','qwen']:
        for mode in ['naive','hybrid','hybrid_rerank']:
            group=[x for x in runs if x['profile']==profile and x['mode']==mode]
            sg=[x for x in scores if x['profile']==profile and x['mode']==mode]
            row={'profile':profile,'mode':mode,'scope':'smoke','metric_eligible':False,'requests':len(group),'errors':sum(x['status']!='ok' for x in group)}
            row['error_rate']=row['errors']/len(group) if group else None
            latency=[x['response']['latency_ms'] for x in group if x['status']=='ok']
            row.update(latency_mean_ms=mean(latency) if latency else None,latency_valid_n=len(latency))
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
            for metric in ['hit_at_4','recall_at_4','faithfulness']:
                valid=[x[metric] for x in sg if x.get(metric) is not None]
                row[metric]=mean(valid) if valid else None
                row[metric+'_valid_n']=len(valid)
            row['faithfulness_na_n']=sum(x.get('faithfulness_status')=='not_applicable' for x in sg)
            row['faithfulness_unscorable_n']=sum(x.get('faithfulness_status')=='unscorable' for x in sg)
            row['judge_unscorable_n']=sum(x.get('judge_status')=='unscorable' for x in sg)
            for metric in ['facts','conditions','evidence','decision']:
                valid=[x['judge'][metric]['score'] for x in sg if x.get('judge') and x['judge'].get(metric,{}).get('score') is not None]
                row['judge_'+metric]=mean(valid) if valid else None
                row['judge_'+metric+'_valid_n']=len(valid)
            output.append(row)
    write_json(folder/'summary.json',{'scope':'smoke','metric_eligible':False,'rows':output,'cost':None,'cost_reason':'All inference is local; cloud storage/tracing usage billed separately'})
    with (folder/'summary.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(output[0]));w.writeheader();w.writerows(output)
    return folder/'summary.csv'
