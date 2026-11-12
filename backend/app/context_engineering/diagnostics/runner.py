"""진단용 합성 근거를 직접 공급한다. Qdrant 검색/기존 평가셋은 사용하지 않는다."""
from contextlib import nullcontext
from datetime import datetime, timezone
import html
import json
from pathlib import Path
import re
import time

from ...artifacts import digest
from ...observability import langfuse_client
from ...providers import Providers
from ...schemas import ModelAnswer
from ..builder import prepare_context
from ..budget import generation_tokenizer, input_budget, TOKEN_MARGIN
from ..prompts import DEFENSE_ADDITION, PROMPT_VERSION
from ..validation import validate_answer


class FixtureTokenizer:
    """오프라인 경계 테스트용 문자 카운터. 실제 모델 토큰 수가 아니다."""
    def encode(self, text):
        return list(text)


def load_questions():
    return json.loads(Path(__file__).with_name('questions.json').read_text(encoding='utf-8'))


def _check_sample(case, contexts):
    try:
        answer = validate_answer(json.dumps(case['sample_output'], ensure_ascii=False), contexts)
        return {'status':'passed', 'answer':answer.model_dump()}
    except ValueError as exc:
        return {'status':'rejected', 'error_type':type(exc).__name__}


def diagnose_case(s, case, *, mode='offline', tokenizer=None):
    if mode not in {'offline', 'model'}:
        raise ValueError('Unknown diagnostic mode')
    if tokenizer is None:
        tokenizer = FixtureTokenizer() if mode == 'offline' else generation_tokenizer(
            s.generation_llm_provider, s.local_llm_model if s.generation_llm_provider == 'ollama' else s.openai_model)
    record = {'id':case['id'], 'title':case['title'], 'question':case['question'],
              'mode':mode, 'synthetic_evidence':True, 'semantic_review_required':True,
              'token_measurement':'fixture_characters_plus_margin' if mode == 'offline' else 'model_token_estimate',
              'candidate_chunks':case['chunks'], 'checks':{}}
    record['expected_structural_rules'] = {k:v for k,v in case.items() if k.startswith('expected_')}
    selected_settings = s
    kwargs = {'memory_text':case.get('memory_text','사용하지 않음'),
              'defense_addition':DEFENSE_ADDITION if case.get('defense') else None, 'tokenizer':tokenizer}
    if case.get('budget_profile') == 'trim_to_first':
        first = prepare_context(s, case['question'], case['chunks'][:1], **kwargs)
        selected_settings = s.model_copy(update={'context_max_tokens':first.estimated_input_tokens})
    elif case.get('budget_profile') == 'too_small':
        selected_settings = s.model_copy(update={'context_max_tokens':1})
    record['budget'] = {'input_limit':input_budget(selected_settings),
                        'output_reserve_tokens':s.llm_max_output_tokens, 'margin_units':TOKEN_MARGIN,
                        'measurement':record['token_measurement']}
    try:
        prepared = prepare_context(selected_settings, case['question'], case['chunks'], **kwargs)
    except Exception as exc:
        code = getattr(exc, 'code', type(exc).__name__)
        record.update(preparation_error=code)
        record['checks']['preparation_error'] = code == case.get('expected_error')
        return record
    record.update(final_context=prepared.contexts, actual_messages=prepared.messages,
                  estimated_input_tokens=prepared.estimated_input_tokens,
                  dropped_chunk_ids=prepared.dropped_chunk_ids, warnings=prepared.warnings,
                  context_hash=prepared.context_hash, messages_hash=digest(prepared.messages))
    record['budget']['estimated_input_units'] = prepared.estimated_input_tokens
    checks = record['checks']
    checks['preparation'] = 'expected_error' not in case
    checks['selected_sources'] = [c['chunk_id'] for c in prepared.contexts] == case['expected_kept_ids']
    checks['dropped_sources'] = prepared.dropped_chunk_ids == case.get('expected_dropped_ids', [])
    sample = _check_sample(case, prepared.contexts)
    record['sample_validation'] = sample
    checks['sample_validation'] = sample['status'] == case['expected_validation']
    if mode == 'model':
        if prepared.contexts:
            provider = Providers(s)
            provider.deadline = time.perf_counter() + s.request_timeout_seconds
            try:
                from ...observability import observation
                model = s.local_llm_model if s.generation_llm_provider == 'ollama' else s.openai_model
                with observation('context_diagnostic_generation', kind='generation', input=prepared.messages,
                                 model=model, metadata={'case_id':case['id'], 'synthetic_evidence':True}) as span:
                    generated = provider.chat(prepared.messages, ModelAnswer.model_json_schema())
                    if span:
                        span.update(output=generated['content'], usage_details={k:v for k,v in {
                            'input':generated.get('input_tokens'), 'output':generated.get('output_tokens')}.items() if v is not None})
                record['model_result'] = {'generated':generated}
                try:
                    answer = validate_answer(generated['content'], prepared.contexts)
                    record['model_result'].update(validation='passed', answer=answer.model_dump())
                except ValueError as exc:
                    record['model_result'].update(validation='rejected', error_type=type(exc).__name__)
            except Exception as exc:
                record['model_result'] = {'status':'error', 'error_type':type(exc).__name__,
                                          'attempts':getattr(exc, 'attempts', [])}
        else:
            record['model_result'] = {'status':'skipped', 'reason':'No evidence; generation not called'}
    return record


def _write_report(folder, records, summary):
    cards = []
    for r in records:
        label = '구조 검사 통과' if all(r['checks'].values()) else '구조 검사 확인 필요'
        sections = []
        for title, key in [('기대하는 구조 규칙','expected_structural_rules'), ('토큰/문자 예산','budget'),
                           ('준비 단계 오류','preparation_error'), ('검색 대체 입력(합성 근거)','candidate_chunks'), ('최종 Context','final_context'),
                           ('실제 모델 입력','actual_messages'), ('제외·경고','warnings'),
                           ('샘플 출력 검증(모델 실행 아님)','sample_validation'), ('실제 모델 결과','model_result')]:
            if key in r:
                sections.append(f'<details><summary>{title}</summary><pre>{html.escape(json.dumps(r[key],ensure_ascii=False,indent=2))}</pre></details>')
        cards.append(f'<article><h2>{html.escape(r["id"]+" · "+r["title"])}</h2>'
                     f'<p>{html.escape(r["question"])}</p><b>{label}</b>'
                     f'<pre>{html.escape(json.dumps(r["checks"],ensure_ascii=False))}</pre>'+''.join(sections)+'</article>')
    page = '''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>Context 단계별 진단</title><style>body{font:15px/1.6 system-ui,sans-serif;max-width:1000px;margin:36px auto;padding:0 20px;background:#f3f6fa;color:#173047}article{background:white;border:1px solid #d8e2ea;border-radius:12px;margin:18px 0;padding:20px}h2{font-size:18px}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#eef3f7;padding:12px}summary{cursor:pointer;padding:8px}aside{background:#fff4d8;padding:16px}</style>
<h1>Context Engineering 단계별 진단</h1><aside>합성 근거를 사용하는 개발용 진단입니다. 기존 PDF 평가·검색 정확도·답변 사실성·보안 성공률을 나타내지 않습니다. offline의 길이는 문자 카운터이며 실제 모델 토큰이 아닙니다.</aside>'''
    page += '<pre>'+html.escape(json.dumps(summary, ensure_ascii=False, indent=2))+'</pre>'+''.join(cards)+'</html>'
    (folder/'report.html').write_text(page, encoding='utf-8')


def run_diagnostics(s, *, mode='offline', limit=None, trace=False, run_name=None):
    if mode == 'model' and (limit is None or not 1 <= limit <= 24):
        raise ValueError('Model mode requires an explicit limit between 1 and 24')
    if limit is not None and limit < 1:
        raise ValueError('Limit must be positive')
    run_name = run_name or 'ce-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    if not re.fullmatch(r'[A-Za-z0-9_-]{1,100}',run_name):
        raise ValueError('Invalid diagnostic run name')
    folder = s.runtime_dir/'context_diagnostics'/run_name
    if folder.exists():
        raise ValueError('Run already exists; choose a new run name')
    client = langfuse_client() if trace else None
    if trace and client is None:
        raise ValueError('Langfuse is not configured/enabled')
    folder.mkdir(parents=True)
    records = []
    for case in load_questions()[:limit]:
        attrs = nullcontext()
        if client:
            from langfuse import propagate_attributes
            attrs = propagate_attributes(session_id=run_name, tags=['evaluation','context_diagnostic'])
        with attrs:
            root = client.start_as_current_observation(name='context_diagnostic', input={'question':case['question']},
                metadata={'case_id':case['id'], 'run_name':run_name, 'execution_type':'evaluation', 'synthetic_evidence':True}) if client else nullcontext(None)
            with root as span:
                result = diagnose_case(s, case, mode=mode)
                if client:
                    result['trace_id'] = client.get_current_trace_id()
                if span:
                    span.update(output=result)
                records.append(result)
                with (folder/'results.jsonl').open('a',encoding='utf-8') as f:
                    f.write(json.dumps(result,ensure_ascii=False)+'\n')
    summary = {'run_name':run_name, 'mode':mode, 'cases':len(records),
               'structural_passes':sum(all(r['checks'].values()) for r in records),
               'model_calls_requested':mode == 'model', 'semantic_correctness':'manual_review_required',
               'trace_requested':trace, 'cloud_verified':False}
    summary.update(question_set_hash=digest(load_questions()), prompt_version=PROMPT_VERSION,
                   generation_provider=s.generation_llm_provider,
                   generation_model=s.local_llm_model if s.generation_llm_provider == 'ollama' else s.openai_model)
    if client:
        try:
            client.flush()
            summary['flush_requested'] = True
        except Exception as exc:
            summary['flush_error'] = type(exc).__name__
    (folder/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    _write_report(folder, records, summary)
    return folder, summary
