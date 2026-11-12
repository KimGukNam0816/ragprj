import ast
import inspect

from backend.app import observability
from backend.app.config import ROOT

EXEMPT = (ROOT / 'backend/app/observability', ROOT / 'backend/tests/observability')


def test_every_public_name_exists_and_callables_have_docstrings():
    for name in observability.__all__:
        value = getattr(observability, name)
        if inspect.isclass(value) or callable(value):
            assert inspect.getdoc(value), f'{name} has no docstring'


def submodule_imports(path):
    tree = ast.parse(path.read_text(encoding='utf-8'))
    for node in ast.walk(tree):
        modules = [a.name for a in node.names] if isinstance(node, ast.Import) else \
            [node.module or ''] if isinstance(node, ast.ImportFrom) else []
        for module in modules:
            parts = module.split('.')
            # The shared test helper (backend.tests.observability.fake_langfuse) is allowed.
            if 'observability' in parts and parts.index('observability') < len(parts) - 1 and 'tests' not in parts:
                yield '.'.join(parts[parts.index('observability'):])


def test_other_packages_import_only_the_package():
    found = set()
    for folder in ('backend', 'evaluation', 'scripts'):
        for path in (ROOT / folder).rglob('*.py'):
            if '.venv' in path.parts or any(path.is_relative_to(e) for e in EXEMPT):
                continue
            found |= {(path.relative_to(ROOT).as_posix(), m) for m in submodule_imports(path)}
    assert not found, f'import from backend.app.observability instead: {sorted(found)}'
