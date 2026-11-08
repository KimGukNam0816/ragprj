"""Read-only progress/audit snapshot; never executes inference or deletes data."""
import json
from backend.app.config import get_settings
from backend.app.artifacts import write_json

if __name__=='__main__':
    s=get_settings();report={'scope':'smoke','metric_eligible':False,'comparisons':[]}
    for folder in sorted((s.runtime_dir/'comparisons').glob('*')):
        if not folder.is_dir():continue
        rows=[json.loads(p.read_text(encoding='utf-8')) for p in (folder/'items').glob('*.json')]
        if not rows:continue
        scores=[json.loads(p.read_text(encoding='utf-8')) for p in (folder/'scores').glob('*.json')]
        report['comparisons'].append({'name':folder.name,'requests':len(rows),'errors':sum(r['status']=='error' for r in rows),'scored':len(scores),'ragas_unscorable':sum(r.get('faithfulness_status')=='unscorable' for r in scores),'judge_unscorable':sum(r.get('judge_status')=='unscorable' for r in scores)})
    reports=list((s.runtime_dir/'runs').glob('*/*/result.json'))
    report['external_generation_inference_attempts']=sum(json.loads(p.read_text(encoding='utf-8')).get('external_inference_attempts',0) for p in reports)
    write_json(s.runtime_dir/'checks/smoke_status.json',report)
    print(json.dumps(report,ensure_ascii=False,indent=2))
