import argparse
import os


def main():
    parser = argparse.ArgumentParser(description='Synthetic Context diagnostics; no Qdrant or original benchmark access')
    parser.add_argument('--mode',choices=['offline','model'],default='offline')
    parser.add_argument('--limit',type=int)
    parser.add_argument('--trace',action='store_true',help='Explicitly send synthetic diagnostic records to Langfuse')
    parser.add_argument('--run-name')
    args = parser.parse_args()
    # 이 CLI 프로세스에서만 전송을 제어한다. 실제 .env는 변경하지 않는다.
    if not args.trace:
        os.environ['LANGFUSE_ENABLED'] = 'false'
    from ...config import get_settings
    from .runner import run_diagnostics
    settings = get_settings()
    if args.mode == 'model' and args.limit is None:
        parser.error('Model mode requires --limit (1..24); actual model calls may incur cost')
    folder, summary = run_diagnostics(settings, mode=args.mode, limit=args.limit,
                                      trace=args.trace, run_name=args.run_name)
    print(f'Structural checks: {summary["structural_passes"]}/{summary["cases"]}')
    print(folder/'report.html')
    return 0 if summary['structural_passes'] == summary['cases'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
