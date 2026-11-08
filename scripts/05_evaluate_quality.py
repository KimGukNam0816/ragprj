import argparse
import json
import time
from pathlib import Path
from backend.app.config import get_settings
from evaluation.quality import evaluate_comparison

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--comparison',type=Path,required=True);p.add_argument('--follow',action='store_true',help='Evaluate newly completed comparison items until all expected requests exist');p.add_argument('--report',action='store_true');a=p.parse_args()
    while True:
        result=evaluate_comparison(get_settings(),a.comparison)
        if not a.follow:break
        manifest=json.loads((a.comparison/'manifest.json').read_text(encoding='utf-8'))
        expected=manifest['expected_requests']
        scored=len(list((a.comparison/'scores').glob('*.json')))
        if scored>=expected and (a.comparison/'runs.jsonl').exists():break
        time.sleep(5)
    print(result)
    if a.report:
        from evaluation.report import report
        print(report(a.comparison))
