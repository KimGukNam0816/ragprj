import json
from backend.app.artifacts import write_json, write_jsonl
from backend.app.config import Settings
from evaluation.report import report


def test_force_local_only_blocks_api_even_when_api_flag_is_enabled():
    import pytest
    s=Settings(_env_file=None,execution_stage='api_core10',generation_llm_provider='openai',allow_external_llm=True,force_local_only=True)
    with pytest.raises(ValueError,match='disabled'):
        s.check_inference_policy()


def test_report_keeps_zero_scores_and_failure_denominators(tmp_path):
    saved=tmp_path/'request.json'
    write_json(saved,{'generation':{'attempts':[{'input_tokens':0,'output_tokens':3}]}})
    write_jsonl(tmp_path/'runs.jsonl',[
        {'profile':'bge','mode':'naive','status':'ok','result_path':str(saved),'response':{'latency_ms':100}},
        {'profile':'bge','mode':'naive','status':'error'},
    ])
    write_jsonl(tmp_path/'scores.jsonl',[
        {'profile':'bge','mode':'naive','hit_at_4':0,'recall_at_4':0,'faithfulness':0,'faithfulness_status':'ok','judge_status':'ok','judge':{k:{'score':0,'reason':'incorrect'} for k in ['facts','conditions','evidence','decision']}},
        {'profile':'bge','mode':'naive','hit_at_4':None,'recall_at_4':None,'faithfulness':None,'faithfulness_status':'unscorable','judge_status':'service_error','judge':{'decision':{'score':0,'reason':'error'}}},
    ])
    report(tmp_path)
    summary=json.loads((tmp_path/'summary.json').read_text(encoding='utf-8'))
    row=summary['rows'][0]
    assert row['error_rate']==.5
    assert row['faithfulness']==0 and row['faithfulness_valid_n']==1
    assert row['judge_decision_valid_n']==2
    assert row['input_tokens_sum']==0 and row['input_tokens_known_requests']==1
    assert row['latency_mean_ms']==100 and row['latency_valid_n']==1
    assert summary['metric_eligible'] is False
