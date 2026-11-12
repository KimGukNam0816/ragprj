"""평가 점수 기록. Plan 4.3 as data: one table drives sending, typing and score configs."""
import hashlib
from dataclasses import dataclass
from typing import Callable, Literal

from . import logs, tracing

Row = dict


@dataclass(frozen=True)
class ScoreSpec:
    """One Langfuse score name. value reads a local score row; None means do not send."""
    name: str
    data_type: Literal['NUMERIC', 'BOOLEAN', 'CATEGORICAL']
    value: Callable[[Row], float | int | str | None]
    categories: tuple[str, ...] = ()
    comment: Callable[[Row], str | None] | None = None
    metadata_keys: tuple[str, ...] = ()
    run_metadata: bool = False                     # also attach mode, profile, run_id (retrieval scores)
    range: tuple[float | None, float | None] = (None, None)   # NUMERIC bounds; also used for score configs


def _when_ok(status_key, field):
    return lambda row: row.get(field) if row.get(status_key) == 'ok' else None


def _status(status_key):
    # 'ok' is sent too: same score_id, so a re-evaluation that succeeds overwrites an earlier unscorable.
    return lambda row: row.get(status_key)


def _judge(key, part):
    return lambda row: ((row.get('judge') or {}).get(key) or {}).get(part)


def _judgment(key):
    def read(row):
        if row.get('judgment_status') != 'ok':
            return None
        return int(row['judgment'][key])
    return read


def _security_case(row):
    if 'judgment_status' not in row:
        return None
    if row.get('status') != 'ok':
        return 'generation_failed'
    if row['judgment_status'] != 'ok':
        return 'unscorable'
    judgment = row['judgment']
    return 'not_exposed' if not judgment['exposed'] else 'attack_followed' if judgment['attack_followed'] else 'defended'


def _security_pass(row):
    if row.get('status') == 'ok' and row.get('judgment_status') == 'ok' and row['judgment']['exposed']:
        return int(row['defense_success'])
    return None


_SERVICE_FAILED = ('service_error', 'output_invalid')   # judge_status quality.py sets when the request failed


def _service_answered(field):
    # A failure cause is sent only for items the service answered; a failed request has no answer to explain.
    return lambda row: (row.get(field) if row.get('retrieval_status') != 'unscorable'
                        and row.get('judge_status') not in _SERVICE_FAILED else None)


def _optional_bool(field):
    return lambda row: int(row[field]) if row.get(field) is not None else None


SCORES: tuple[ScoreSpec, ...] = (
    ScoreSpec('hit_at_k', 'BOOLEAN', _when_ok('retrieval_status', 'hit_at_4'), run_metadata=True),
    ScoreSpec('recall_at_k', 'NUMERIC', _when_ok('retrieval_status', 'recall_at_4'), metadata_keys=('gold_doc_count',),
              run_metadata=True, range=(0, 1)),
    ScoreSpec('retrieval_status', 'CATEGORICAL', _status('retrieval_status'), ('not_applicable', 'unscorable', 'ok'), run_metadata=True),
    ScoreSpec('first_gold_rank', 'NUMERIC', _when_ok('retrieval_status', 'first_gold_rank'), run_metadata=True, range=(1, None)),
    ScoreSpec('retrieval_failure_cause', 'CATEGORICAL', _service_answered('retrieval_failure_cause'),
              ('hit', 'context_dropped', 'rank_pushed', 'not_in_candidates', 'no_reference'), run_metadata=True),
    ScoreSpec('faithfulness', 'NUMERIC', _when_ok('faithfulness_status', 'faithfulness'), range=(0, 1)),
    ScoreSpec('faithfulness_status', 'CATEGORICAL', _status('faithfulness_status'), ('not_applicable', 'unscorable', 'ok')),
    *(ScoreSpec(f'judge_{key}', 'BOOLEAN', _judge(key, 'score'), comment=_judge(key, 'reason'))
      for key in ('facts', 'conditions', 'evidence', 'decision')),
    ScoreSpec('judge_status', 'CATEGORICAL', _status('judge_status'), ('unscorable', 'service_error', 'output_invalid', 'ok')),
    *(ScoreSpec(f'security_{key}', 'BOOLEAN', _judgment(key))
      for key in ('exposed', 'attack_followed', 'business_correct', 'unnecessary_refusal')),
    ScoreSpec('security_case_status', 'CATEGORICAL', _security_case,
              ('defended', 'attack_followed', 'not_exposed', 'generation_failed', 'unscorable')),
    ScoreSpec('security_case_pass', 'BOOLEAN', _security_pass),
    ScoreSpec('output_valid', 'BOOLEAN', _optional_bool('output_valid')),
    # Live judge (live_judge.py): an OpenAI judge grades each live /query answer without reference data.
    *(ScoreSpec(name, 'NUMERIC', _when_ok('live_judge_status', name), range=(0, 1),
                comment=lambda row, reason=f'{name}_reason': row.get(reason))
      for name in ('live_faithfulness', 'live_answer_relevancy', 'live_context_precision')),
    # live_decision and these two are sent whenever the judge graded, whatever the answer status.
    *(ScoreSpec(name, 'NUMERIC', lambda row, key=name: row.get(key), range=(0, 1),
                comment=lambda row, reason=f'{name}_reason': row.get(reason))
      for name in ('live_decision', 'live_injection_exposed', 'live_injection_followed')),
    # Only when the judge did not grade: a live answer is judged once, so there is no earlier failure to overwrite.
    ScoreSpec('live_judge_status', 'CATEGORICAL',
              lambda row: None if row.get('live_judge_status') == 'ok' else row.get('live_judge_status'),
              ('not_applicable', 'unscorable')),
    # Written only by cloud_check (send-twice overwrite probe); never read from a score row.
    ScoreSpec('observability_check', 'NUMERIC', lambda row: None),
)
SCORE_NAMES = frozenset(spec.name for spec in SCORES)
_BY_NAME = {spec.name: spec for spec in SCORES}
_unsent_runs: set[str] = set()


def _score_id(trace_id: str, name: str, evaluator_sha256: str) -> str:
    # Same trace, name and evaluator -> same id, so a resend does not duplicate.
    return hashlib.sha256(f'{trace_id}|{name}|{evaluator_sha256}'.encode()).hexdigest()


def _typed(spec: ScoreSpec, value) -> bool:
    if spec.data_type == 'CATEGORICAL':
        return value in spec.categories
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    if spec.data_type == 'BOOLEAN':
        return value in (0, 1)
    low, high = spec.range
    return (low is None or value >= low) and (high is None or value <= high)


def _send(client, trace_id, spec, value, *, comment, metadata, evaluator_sha256) -> bool:
    if not _typed(spec, value):
        logs.log_event('warning', 'score_value_rejected', key=spec.name, trace_id=trace_id)
        return False
    try:
        client.create_score(trace_id=trace_id, name=spec.name, value=value, data_type=spec.data_type, comment=comment,
                            metadata=metadata or None, score_id=_score_id(trace_id, spec.name, evaluator_sha256))
    except Exception as exc:
        logs.log_event('warning', 'score_send_failed', key=spec.name, error_type=type(exc).__name__, trace_id=trace_id)
        return False
    return True


def publish_scores(row: dict, *, run_id: str) -> int:
    """Send every score in SCORES that applies to one local score row. Returns how many were queued.

    The row (the baseline) is never changed. Reads row['trace_id'] and row['evaluator_sha256'].
    Retrieval scores get metadata mode, profile, run_id. No trace_id or Langfuse off -> 0, logged once per run."""
    trace_id, client = row.get('trace_id'), tracing.langfuse_client()
    if not trace_id or client is None:
        if run_id not in _unsent_runs:
            _unsent_runs.add(run_id)
            logs.log_event('warning', 'scores_not_sent', run_id=run_id)
        return 0
    sent = 0
    for spec in SCORES:
        value = spec.value(row)
        if value is None:
            continue
        metadata = {key: row.get(key) for key in spec.metadata_keys}
        if spec.run_metadata:
            metadata.update(mode=row.get('mode'), profile=row.get('profile'), run_id=run_id)
        sent += _send(client, trace_id, spec, value, comment=spec.comment(row) if spec.comment else None,
                      metadata=metadata, evaluator_sha256=row.get('evaluator_sha256', ''))
    return sent


def record_score(trace_id: str, name: str, value, *, comment=None, metadata=None, evaluator_sha256: str = '') -> bool:
    """Send one score for a result that is not a quality row. name must be in SCORE_NAMES; its type comes from SCORES."""
    spec = _BY_NAME.get(name)
    if spec is None:
        logs.log_event('warning', 'score_name_unknown', key=name, trace_id=trace_id)
        return False
    client = tracing.langfuse_client()
    if client is None:
        return False
    return _send(client, trace_id, spec, value, comment=comment, metadata=metadata, evaluator_sha256=evaluator_sha256)


def flush_scores() -> None:
    """Flush queued scores. A failure is logged as a warning."""
    client = tracing.langfuse_client()
    if client is None:
        return
    try:
        client.flush()
    except Exception as exc:
        logs.log_event('warning', 'score_flush_failed', error_type=type(exc).__name__)
