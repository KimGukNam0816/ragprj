import argparse
import asyncio
import json
from datetime import datetime, timezone

from backend.app.artifacts import write_json
from backend.app.config import get_settings
from backend.app.connections import check_connections
from backend.app.providers import Providers


async def run(args):
    s = get_settings()
    s.check_inference_policy()
    report = {'timestamp': datetime.now(timezone.utc).isoformat(), 'stage': s.execution_stage, 'connections': await check_connections(s)}
    provider = Providers(s)
    if args.embed:
        report['embeddings'] = {}
        for profile in ['bge', 'qwen']:
            try:
                vector = provider.embed(['해운 문서 검색 연결 확인'], profile)[0]
                report['embeddings'][profile] = {'ok': True, 'dimensions': len(vector)}
            except Exception as exc:
                report['embeddings'][profile] = {'ok': False, 'error_type': type(exc).__name__}
    if args.generate:
        try:
            result = provider.chat([{'role': 'user', 'content': '연결 점검입니다. JSON 객체 {"ok":true}만 답하세요.'}])
            report['generation'] = {'ok': True, 'model': result['model'], 'json': isinstance(json.loads(result['content']), dict), 'input_tokens': result['input_tokens'], 'output_tokens': result['output_tokens']}
        except Exception as exc:
            report['generation'] = {'ok': False, 'error_type': type(exc).__name__}
    report['external_inference_attempts'] = provider.external_inference_attempts
    path = s.runtime_dir / 'checks' / 'connections.json'
    write_json(path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print('REPORT=' + str(path))
    required = ['ollama', 'qdrant', 'langfuse'] + (['llm'] if s.generation_llm_provider == 'openai' else [])
    return all(report['connections'][name]['status'] == 'ok' for name in required) and all(v['ok'] for v in report.get('embeddings', {}).values()) and report.get('generation', {'ok': True})['ok']

if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Read-only service checks; optional short inference')
    p.add_argument('--embed', action='store_true')
    p.add_argument('--generate', action='store_true')
    raise SystemExit(0 if asyncio.run(run(p.parse_args())) else 1)
