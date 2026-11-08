import argparse
import json
from pathlib import Path

from backend.app.artifacts import write_json
from backend.app.config import get_settings
from backend.app.pipeline import run_query, verify_trace
from backend.app.schemas import QueryRequest

if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Run the shared PDF query pipeline once')
    q = p.add_mutually_exclusive_group(required=True)
    q.add_argument('--question')
    q.add_argument('--question-id')
    q.add_argument('--verify-existing', type=Path, help='Read back a saved response trace without running inference again')
    p.add_argument('--questions', type=Path, default=Path('evaluation/selection/core10.json'))
    p.add_argument('--embedding', choices=['bge', 'qwen'], default='bge')
    p.add_argument('--mode', choices=['naive', 'hybrid', 'hybrid_rerank'], default='naive')
    p.add_argument('--run-id')
    p.add_argument('--verify-trace', action='store_true')
    args = p.parse_args()
    if args.verify_existing:
        saved = json.loads((args.verify_existing / 'result.json').read_text(encoding='utf-8'))
        response = saved.get('response')
        if not response:
            p.error('Saved result has no successful response')
        manifest=json.loads((args.verify_existing/'manifest.json').read_text(encoding='utf-8'))
        status = verify_trace(get_settings(), response['trace_id'], response['request_id'],manifest.get('started_at'))
        write_json(args.verify_existing / 'trace_verification.json', status)
        print('TRACE=' + json.dumps(status))
        raise SystemExit(0 if status['verified'] else 2)
    question = args.question
    if args.question_id:
        entries = json.loads(args.questions.read_text(encoding='utf-8'))
        matches = [x for x in entries if x['id'] == args.question_id]
        if len(matches) != 1:
            p.error('Question ID must match exactly one entry')
        question = matches[0]['question']
    s = get_settings()
    try:
        response, output = run_query(s, QueryRequest(question=question, mode=args.mode, embedding_profile=args.embedding, memory_enabled=False), run_id=args.run_id, question_id=args.question_id)
        print(json.dumps(response.model_dump(exclude={'contexts'}), ensure_ascii=False, indent=2))
        print('RESULT_DIR=' + str(output))
        if args.verify_trace:
            status = verify_trace(s, response.trace_id, response.request_id)
            write_json(output / 'trace_verification.json', status)
            print('TRACE=' + json.dumps(status))
            if not status['verified']:
                raise SystemExit(2)
    except Exception as exc:
        print('ERROR=' + type(exc).__name__ + ': ' + getattr(exc, 'message', 'Inspect the saved runtime/runs result; credentials are not printed'))
        raise SystemExit(1)
