"""Create the Langfuse prompt and score configs: python -m backend.app.observability.setup [--apply].

Preview by default (reads only). --apply writes to Langfuse Cloud and needs the observability owner's approval per run.
Prompts and score configs are project-wide, not split by environment. Not exported: a CLI entry point, not a library function."""
import argparse

from . import tracing
from .prompting import to_langfuse_messages
from .scores import _BY_NAME, SCORES

PROMPT_NAME = 'rag_answer'
PROMPT_LABELS = ['production', 'comparison-v1']


def _prompt_exists(client) -> bool:
    try:
        return not client.get_prompt(PROMPT_NAME, type='chat', label='production', max_retries=0).is_fallback
    except Exception:
        return False


def _configs(client) -> dict:
    configs, page = {}, 1
    while True:
        response = client.api.score_configs.get(page=page, limit=100)
        configs.update({config.name: config for config in response.data})
        if page >= response.meta.total_pages:
            return configs
        page += 1


def _stale_categories(spec, config) -> bool:
    """A categorical config whose labels differ from the table (e.g. a category added later). Unknown -> not stale."""
    current = getattr(config, 'categories', None)
    return spec.data_type == 'CATEGORICAL' and current is not None and [c.label for c in current] != list(spec.categories)


def _config_fields(spec) -> dict:
    if spec.data_type == 'CATEGORICAL':
        from langfuse.api import ConfigCategory
        return {'categories': [ConfigCategory(value=i, label=c) for i, c in enumerate(spec.categories)]}
    low, high = spec.range
    return {k: v for k, v in {'min_value': low, 'max_value': high}.items() if v is not None}


def plan(client) -> list[tuple[str, str]]:
    """Actions still missing in Langfuse, as (action, name). Existing prompt and config names are skipped."""
    actions = [] if _prompt_exists(client) else [('create_prompt', PROMPT_NAME)]
    existing = _configs(client)
    return actions + [('create_score_config', spec.name) for spec in SCORES if spec.name not in existing] + \
        [('update_score_config', spec.name) for spec in SCORES if spec.name in existing and _stale_categories(spec, existing[spec.name])]


def apply(client, actions: list[tuple[str, str]]) -> None:
    """Run the planned actions. The first prompt version gets both labels."""
    for action, name in actions:
        if action == 'create_prompt':
            from ..context_engineering.prompts import RAG_PROMPT
            client.create_prompt(name=name, prompt=to_langfuse_messages(RAG_PROMPT), labels=PROMPT_LABELS, type='chat')
        elif action == 'update_score_config':   # appended categories keep their old values, so existing scores stay valid
            client.api.score_configs.update(_configs(client)[name].id, **_config_fields(_BY_NAME[name]))
        else:
            spec = _BY_NAME[name]
            client.api.score_configs.create(name=name, data_type=spec.data_type, **_config_fields(spec))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--apply', action='store_true', help="write to Langfuse (needs the observability owner's approval)")
    args = parser.parse_args(argv)
    client = tracing.langfuse_client()
    if client is None:
        print('Langfuse is not configured; nothing to do')
        return 1
    actions = plan(client)
    for action, name in actions:
        print(f'{"apply" if args.apply else "preview"}: {action} {name}')
    if not actions:
        print('Langfuse already has the prompt and every score config')
    if args.apply:
        apply(client, actions)
        client.flush()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
