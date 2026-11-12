"""dashboards.py against an in-memory stand-in for the SDK's api.unstable dashboard endpoints."""
import itertools
from types import SimpleNamespace

from backend.app.observability import dashboards
from backend.app.observability.dashboards import DASHBOARDS
from backend.app.observability.scores import SCORE_NAMES


class _Store:
    def __init__(self):
        self.items, self.ids, self.calls = {}, itertools.count(1), []

    def list(self, page=1, limit=100):
        rows = list(self.items.values())[(page - 1) * limit: page * limit]
        return SimpleNamespace(data=rows, meta=SimpleNamespace(total_pages=(len(self.items) + limit - 1) // limit))


class _Widgets(_Store):
    def _row(self, id, name, view, dimensions, metrics, filters, chart_type, **_):
        return SimpleNamespace(id=id, name=name, view=view, dimensions=dimensions, metrics=metrics, filters=filters,
                               chart_type=chart_type)

    def create(self, **w):
        self.calls.append(('create', w['name'], w))
        row = self._row(f'w{next(self.ids)}', **w)
        self.items[row.id] = row
        return row

    def update(self, id, **w):
        self.calls.append(('update', w['name'], w))
        self.items[id] = self._row(id, **w)
        return self.items[id]


class _Boards(_Store):
    def create(self, name, description=''):
        row = SimpleNamespace(id=f'd{next(self.ids)}', name=name, definition=SimpleNamespace(widgets=[]))
        self.items[row.id] = row
        return row

    def add_placement(self, dashboard_id, request):
        self.calls.append(('place', dashboard_id, request))
        self.items[dashboard_id].definition.widgets.append(request)


def client():
    return SimpleNamespace(api=SimpleNamespace(unstable=SimpleNamespace(dashboards=_Boards(), dashboard_widgets=_Widgets())))


def test_four_dashboards_with_unique_widget_names():
    names = [w.name for b in DASHBOARDS for w in b.widgets]
    assert [b.name for b in DASHBOARDS] == ['[관측] 서비스 운영', '[관측] 평가 지표', '[관측] 보안 지표', '[관측] Context engineering']
    assert len(names) == len(set(names))


def test_score_widgets_only_reference_known_score_names():
    used = set()
    for widget in (w for b in DASHBOARDS for w in b.widgets if w.view.startswith('scores')):
        for f in widget.filters:
            if f['column'] == 'name':
                used |= set(f['value']) if isinstance(f['value'], list) else {f['value']}
    assert used and used <= SCORE_NAMES


def test_apply_creates_everything_then_plan_is_empty():
    c = client()
    total = sum(len(b.widgets) for b in DASHBOARDS)
    actions = dashboards.plan(c)
    assert sum(a[0] == 'create_dashboard' for a in actions) == 4
    assert sum(a[0] == 'create_widget' for a in actions) == total == sum(a[0] == 'place_widget' for a in actions)
    dashboards.apply(c, actions)
    assert dashboards.plan(c) == []
    placements = [call[2] for call in c.api.unstable.dashboards.calls]
    assert {(p.x, p.y) for p in placements[:2]} == {(0, 0), (6, 0)}


def test_changed_definition_is_updated_not_duplicated(monkeypatch):
    c = client()
    dashboards.apply(c, dashboards.plan(c))
    first = DASHBOARDS[0]
    changed = first.widgets[0].__class__(**{**first.widgets[0].__dict__, 'chart': 'BAR_TIME_SERIES'})
    board = first.__class__(first.name, first.description, (changed, *first.widgets[1:]))
    monkeypatch.setattr(dashboards, 'DASHBOARDS', (board, *DASHBOARDS[1:]))
    assert dashboards.plan(c) == [('update_widget', first.name, changed.name)]
    dashboards.apply(c, dashboards.plan(c))
    assert dashboards.plan(c) == [] and len(c.api.unstable.dashboard_widgets.items) == sum(len(b.widgets) for b in DASHBOARDS)


def test_histogram_sends_bins():
    c = client()
    dashboards.apply(c, dashboards.plan(c))
    sent = {name: w for kind, name, w in c.api.unstable.dashboard_widgets.calls if kind == 'create'}
    assert sent['[평가] 첫 정답 등수 분포 (naive)']['chart_config'].bins == 20
    assert 'chart_config' not in sent['[운영] 요청 수']
