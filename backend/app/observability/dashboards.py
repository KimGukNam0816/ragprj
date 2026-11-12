"""Create the Langfuse dashboards: python -m backend.app.observability.dashboards [--apply].

Four dashboards (service operations, evaluation metrics, security metrics, context engineering), defined as data below.
Preview by default (reads only). --apply writes to Langfuse Cloud and needs the observability owner's approval per run.
Idempotent by name: existing dashboards and widgets are reused, a widget whose definition changed is updated, and a
widget missing from its dashboard is placed. Dashboards are project-wide; environment is chosen in the Langfuse UI.
Uses the SDK's `api.unstable` dashboard endpoints, which Langfuse may still change. Not exported: a CLI entry point."""
import argparse
from dataclasses import dataclass, field

from . import tracing
from .tracing import ROOT_NAME

CONTEXT_SPANS = ('context_deduplication', 'context_budget', 'context_prompt', 'context_output_validation')
JUDGE_SCORES = ('judge_facts', 'judge_conditions', 'judge_evidence', 'judge_decision')
SECURITY_CHECKS = ('security_exposed', 'security_attack_followed', 'security_business_correct',
                   'security_unnecessary_refusal')
WIDTH, HEIGHT = 6, 6   # two tiles per row on Langfuse's 12-column grid


def _is(column: str, value: str) -> dict:
    return {'column': column, 'type': 'string', 'operator': '=', 'value': value}


def _any(column: str, values, kind: str = 'stringOptions') -> dict:
    return {'column': column, 'type': kind, 'operator': 'any of', 'value': list(values)}


def _tags(*tags: str) -> dict:
    return _any('tags', tags, 'arrayOptions')


def _meta(key: str, value: str) -> dict:
    # The SDK stores metadata values as strings (json.dumps for non-strings), so True is 'true'.
    return {'column': 'metadata', 'type': 'stringObject', 'key': key, 'operator': '=', 'value': value}


@dataclass(frozen=True)
class Widget:
    name: str
    view: str                                   # observations | scores-numeric | scores-boolean | scores-categorical
    metrics: tuple[tuple[str, str], ...]        # (measure, aggregation)
    chart: str
    dimensions: tuple[str, ...] = ()
    filters: tuple[dict, ...] = ()
    description: str = ''
    bins: int | None = None                     # HISTOGRAM only


@dataclass(frozen=True)
class Dashboard:
    name: str
    description: str
    widgets: tuple[Widget, ...] = field(default=())


ROOT = _is('name', ROOT_NAME)
COMPARISON, SECURITY = _tags('comparison'), _tags('security')


def _rank_histogram(mode: str) -> Widget:
    # A histogram takes no breakdown, so each retrieval mode gets its own tile.
    return Widget(f'[평가] 첫 정답 등수 분포 ({mode})', 'scores-numeric', (('value', 'histogram'),), 'HISTOGRAM',
                  filters=(_is('name', 'first_gold_rank'), COMPARISON, _tags(f'mode:{mode}')), bins=20,
                  description='후보 목록에서 정답 문서 청크가 처음 나온 위치(1부터)')


DASHBOARDS: tuple[Dashboard, ...] = (
    Dashboard('[관측] 서비스 운영', '요청 수, 오류, 단계별 지연, 토큰, LLM 시도', (
        Widget('[운영] 요청 수', 'observations', (('count', 'count'),), 'LINE_TIME_SERIES', filters=(ROOT,),
               description='shipping_rag_query trace 수(모든 실행 종류)'),
        Widget('[운영] 오류 건수 (level별)', 'observations', (('count', 'count'),), 'BAR_TIME_SERIES',
               dimensions=('level',), filters=(ROOT,), description='루트 level=ERROR가 실패 요청'),
        Widget('[운영] 단계별 지연 p95', 'observations', (('latency', 'p95'),), 'HORIZONTAL_BAR',
               dimensions=('name',), description='span 이름별 지연(ms)'),
        Widget('[운영] 생성 토큰 합계', 'observations', (('inputTokens', 'sum'), ('outputTokens', 'sum')),
               'LINE_TIME_SERIES', filters=(_is('type', 'GENERATION'),)),
        Widget('[운영] LLM 시도 수', 'observations', (('count', 'count'),), 'BAR_TIME_SERIES',
               filters=(_is('name', 'llm_attempt'),), description='첫 시도 포함. 재시도는 metadata attempt>1. LIVE_JUDGE_ENABLED면 판정 호출(live_judge 아래)도 포함'),
        Widget('[운영] 실서비스 답변 품질 지표', 'scores-numeric', (('value', 'avg'),), 'LINE_TIME_SERIES',
               dimensions=('name',),
               filters=(_any('name', ('live_faithfulness', 'live_answer_relevancy', 'live_context_precision',
                                      'live_decision')),),
               description='LIVE_JUDGE_ENABLED일 때 실서비스 답변마다 계산한 0~1 지표 평균(RAGAS 방식)'),
        Widget('[운영] 실서비스 공격 문장 노출·추종률', 'scores-numeric', (('value', 'avg'),), 'LINE_TIME_SERIES',
               dimensions=('name',), filters=(_any('name', ('live_injection_exposed', 'live_injection_followed')),),
               description='노출 = 질문·문맥에 모델을 향한 지시문이 있던 비율, 추종 = 노출된 질의 중 답변이 따른 비율. '
                           '노출 사례는 Dataset live_security_cases'),
    )),
    Dashboard('[관측] 평가 지표', '검색(Hit@4, Recall@4, 첫 정답 등수, 실패 원인)과 품질(RAGAS faithfulness, LLM judge)', (
        Widget('[평가] Hit@4 추이', 'scores-boolean', (('value', 'avg'),), 'LINE_TIME_SERIES', dimensions=('tags',),
               filters=(_is('name', 'hit_at_k'), COMPARISON), description='평균 = 적중 비율. 태그 조합 = 방식×임베딩'),
        Widget('[평가] Recall@4 추이', 'scores-numeric', (('value', 'avg'),), 'LINE_TIME_SERIES', dimensions=('tags',),
               filters=(_is('name', 'recall_at_k'), COMPARISON)),
        Widget('[평가] 첫 정답 등수 평균', 'scores-numeric', (('value', 'avg'),), 'VERTICAL_BAR', dimensions=('tags',),
               filters=(_is('name', 'first_gold_rank'), COMPARISON)),
        _rank_histogram('naive'), _rank_histogram('hybrid'), _rank_histogram('hybrid_rerank'),
        Widget('[평가] 검색 실패 원인', 'scores-categorical', (('count', 'count'),), 'PIVOT_TABLE',
               dimensions=('stringValue', 'tags'), filters=(_is('name', 'retrieval_failure_cause'), COMPARISON)),
        Widget('[평가] Faithfulness 평균 (RAGAS)', 'scores-numeric', (('value', 'avg'),), 'VERTICAL_BAR',
               dimensions=('tags',), filters=(_is('name', 'faithfulness'), COMPARISON)),
        Widget('[평가] LLM judge 통과율', 'scores-boolean', (('value', 'avg'),), 'PIVOT_TABLE',
               dimensions=('name', 'tags'), filters=(_any('name', JUDGE_SCORES), COMPARISON),
               description='태그에 prompt:<해시 8자>가 있어 프롬프트별로도 나뉜다'),
        Widget('[평가] 채점 상태', 'scores-categorical', (('count', 'count'),), 'PIVOT_TABLE',
               dimensions=('name', 'stringValue'),
               filters=(_any('name', ('retrieval_status', 'faithfulness_status', 'judge_status')), COMPARISON),
               description='ok, not_applicable, unscorable, service_error, output_invalid'),
    )),
    Dashboard('[관측] 보안 지표', '보안 비교(base/defense) 판정 결과와 출력 검증', (
        Widget('[보안] 사례 상태 분포', 'scores-categorical', (('count', 'count'),), 'PIE', dimensions=('stringValue',),
               filters=(_is('name', 'security_case_status'), SECURITY),
               description='defended, attack_followed, not_exposed, generation_failed, unscorable'),
        Widget('[보안] 방어 성공률', 'scores-boolean', (('value', 'avg'),), 'NUMBER',
               filters=(_is('name', 'security_case_pass'), SECURITY), description='노출되고 답변이 생성된 사례만'),
        Widget('[보안] 판정 항목 비율', 'scores-boolean', (('value', 'avg'),), 'HORIZONTAL_BAR', dimensions=('name',),
               filters=(_any('name', SECURITY_CHECKS), SECURITY)),
        Widget('[보안] 출력 검증 통과율', 'scores-boolean', (('value', 'avg'),), 'NUMBER',
               filters=(_is('name', 'output_valid'), SECURITY)),
        Widget('[보안] 보안 실행 요청·오류', 'observations', (('count', 'count'),), 'BAR_TIME_SERIES',
               dimensions=('level',), filters=(ROOT, SECURITY)),
    )),
    Dashboard('[관측] Context engineering', '문맥 조립 단계 지연, 출력 검증 실패, 문맥 축소, 프롬프트별 생성', (
        Widget('[Context] 문맥 단계 지연 p95', 'observations', (('latency', 'p95'),), 'HORIZONTAL_BAR',
               dimensions=('name',), filters=(_any('name', CONTEXT_SPANS),)),
        Widget('[Context] 출력 검증 실패 건수', 'observations', (('count', 'count'),), 'BAR_TIME_SERIES',
               filters=(ROOT, _meta('error_code', 'output_invalid')), description='이유 범주는 trace의 error_reason'),
        Widget('[Context] 문맥이 줄어든 요청', 'observations', (('count', 'count'),), 'LINE_TIME_SERIES',
               filters=(ROOT, _meta('context_reduced', 'true')), description='토큰 예산 때문에 청크를 뺀 요청'),
        Widget('[Context] 프롬프트별 생성 지연 p95', 'observations', (('latency', 'p95'),), 'HORIZONTAL_BAR',
               dimensions=('tags',), filters=(_is('name', 'answer_generation'),),
               description='태그 prompt:<해시 8자>가 프롬프트 식별 값'),
        Widget('[Context] 생성 입력 토큰 평균', 'observations', (('inputTokens', 'avg'),), 'NUMBER',
               filters=(_is('name', 'answer_generation'),)),
    )),
)


def _definition(widget: Widget) -> dict:
    """The fields compared with an existing widget and sent on create/update."""
    return {'view': widget.view, 'chart_type': widget.chart, 'dimensions': [{'field': d} for d in widget.dimensions],
            'metrics': [{'measure': m, 'agg': a} for m, a in widget.metrics], 'filters': [dict(f) for f in widget.filters]}


def _existing_definition(widget) -> dict:
    keys = ('column', 'type', 'operator', 'value', 'key')
    return {'view': str(widget.view), 'chart_type': str(widget.chart_type),
            'dimensions': [{'field': d.field} for d in widget.dimensions],
            'metrics': [{'measure': m.measure, 'agg': str(m.agg)} for m in widget.metrics],
            'filters': [{k: v for k in keys if (v := getattr(f, k, None)) is not None} for f in widget.filters]}


def _all(endpoint) -> list:
    rows, page = [], 1
    while True:
        response = endpoint.list(page=page, limit=100)
        rows += response.data
        if page >= max(response.meta.total_pages, 1):
            return rows
        page += 1


def _state(client) -> tuple[dict, dict]:
    api = client.api.unstable
    return {d.name: d for d in _all(api.dashboards)}, {w.name: w for w in _all(api.dashboard_widgets)}


def plan(client) -> list[tuple[str, str, str]]:
    """Missing or changed items as (action, dashboard name, widget name or '')."""
    dashboards, widgets = _state(client)
    actions = []
    for board in DASHBOARDS:
        existing = dashboards.get(board.name)
        if existing is None:
            actions.append(('create_dashboard', board.name, ''))
        placed = {p.widget_id for p in existing.definition.widgets if getattr(p, 'widget_id', None)} if existing else set()
        for widget in board.widgets:
            found = widgets.get(widget.name)
            if found is None:
                actions.append(('create_widget', board.name, widget.name))
            elif _existing_definition(found) != _definition(widget):
                actions.append(('update_widget', board.name, widget.name))
            if found is None or found.id not in placed:
                actions.append(('place_widget', board.name, widget.name))
    return actions


def _models(widget: Widget) -> dict:
    from langfuse.api.unstable.dashboard_widgets.types import (DashboardWidgetChartConfigInput, DashboardWidgetDimension,
                                                               DashboardWidgetFilter, DashboardWidgetMetric)
    return {'view': widget.view, 'chart_type': widget.chart,
            'dimensions': [DashboardWidgetDimension(field=d) for d in widget.dimensions],
            'metrics': [DashboardWidgetMetric(measure=m, agg=a) for m, a in widget.metrics],
            'filters': [DashboardWidgetFilter(**f) for f in widget.filters],
            'chart_config': DashboardWidgetChartConfigInput(type=widget.chart, bins=widget.bins) if widget.bins else None}


def apply(client, actions: list[tuple[str, str, str]]) -> None:
    """Run the planned actions in order. Placements follow the table order, two tiles per row."""
    from langfuse.api.unstable.dashboards.types import CreateDashboardPlacementRequest_Widget
    api = client.api.unstable
    boards = {b.name: b for b in DASHBOARDS}
    specs = {w.name: w for b in DASHBOARDS for w in b.widgets}
    dashboards, widgets = _state(client)
    for action, board_name, widget_name in actions:
        if action == 'create_dashboard':
            dashboards[board_name] = api.dashboards.create(name=board_name, description=boards[board_name].description)
        elif action in ('create_widget', 'update_widget'):
            spec = specs[widget_name]
            fields = {k: v for k, v in _models(spec).items() if v is not None}
            if action == 'create_widget':
                widgets[widget_name] = api.dashboard_widgets.create(name=widget_name, description=spec.description, **fields)
            else:
                widgets[widget_name] = api.dashboard_widgets.update(widgets[widget_name].id, name=widget_name,
                                                                    description=spec.description, **fields)
        else:
            i = [w.name for w in boards[board_name].widgets].index(widget_name)
            api.dashboards.add_placement(dashboards[board_name].id, request=CreateDashboardPlacementRequest_Widget(
                widget_id=widgets[widget_name].id, x=(i % 2) * WIDTH, y=(i // 2) * HEIGHT, width=WIDTH, height=HEIGHT))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--apply', action='store_true', help="write to Langfuse (needs the observability owner's approval)")
    args = parser.parse_args(argv)
    client = tracing.langfuse_client()
    if client is None:
        print('Langfuse is not configured; nothing to do')
        return 1
    actions = plan(client)
    for action, board, widget in actions:
        print(f'{"apply" if args.apply else "preview"}: {action} {board}' + (f' / {widget}' if widget else ''))
    if not actions:
        print('Langfuse already has every dashboard and widget')
    if args.apply:
        apply(client, actions)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
