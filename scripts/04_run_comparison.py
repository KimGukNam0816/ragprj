import argparse
from backend.app.config import get_settings
from evaluation.benchmark import run_comparison

if __name__=='__main__':
    p=argparse.ArgumentParser(description='Run six configurations; current local smoke results are not formal metrics')
    p.add_argument('--set',choices=['core10','final60'],help='Overrides EVAL_SELECTION; an explicit set runs all its questions unless --limit is given')
    p.add_argument('--run-name')
    p.add_argument('--limit',type=int,help='Explicit smoke-only subset')
    a=p.parse_args()
    s=get_settings()
    print('COMPARISON_DIR='+str(run_comparison(s,a.set or s.eval_selection,a.run_name,a.limit if a.set else (a.limit or s.eval_question_limit))))
