import argparse
from backend.app.config import get_settings
from evaluation.security import run_security

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run-name',default='smoke-security');p.add_argument('--embedding',choices=['bge','qwen'],default='bge');p.add_argument('--mode',choices=['naive','hybrid','hybrid_rerank'],default='hybrid_rerank');a=p.parse_args()
    print(run_security(get_settings(),a.run_name,a.embedding,a.mode))
