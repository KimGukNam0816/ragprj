"""The prompt a request really used, and its identity. PROMPT_SOURCE=local|langfuse, default local."""
import json
import re
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Literal

from langchain_core.prompts import ChatPromptTemplate

from ..artifacts import digest, write_json
from ..config import get_settings
from ..schemas import ServiceError
from . import logs, tracing

NEGATIVE_CACHE_SECONDS = 60
_failed_until: dict[tuple, float] = {}
_ROLES = {'SystemMessagePromptTemplate': 'system', 'HumanMessagePromptTemplate': 'user', 'AIMessagePromptTemplate': 'assistant'}


def _local_version() -> str:
    # Lazy: importing context_engineering loads tokenizers and huggingface_hub, which observability callers should not pay for.
    from ..context_engineering.prompts import PROMPT_VERSION
    return PROMPT_VERSION


def _template_sha256(template: ChatPromptTemplate) -> str:
    """Hash of every message (role and text), so a change to the human message also changes the hash."""
    return digest([[m.__class__.__name__, m.prompt.template] for m in template.messages])


@dataclass(frozen=True)
class ResolvedPrompt:
    """The prompt a request really used, with its identity. The only prompt type that crosses the package boundary."""
    template: ChatPromptTemplate
    name: str
    version: str | None
    source: Literal['langfuse', 'fallback', 'local']
    sha256: str
    _link: Any = field(default=None, repr=False, compare=False)

    @classmethod
    def local(cls, name: str, template: ChatPromptTemplate) -> 'ResolvedPrompt':
        """The in-code template, no network. version is prompts.PROMPT_VERSION."""
        return cls(template, name, _local_version(), 'local', _template_sha256(template))

    @property
    def config_fields(self) -> dict:
        """prompt_name, prompt_version, prompt_source, prompt_sha256: same dict for run config and span metadata."""
        return {'prompt_name': self.name, 'prompt_version': self.version,
                'prompt_source': self.source, 'prompt_sha256': self.sha256}


class PromptUnavailable(ServiceError):
    """A strict resolve fell back to the local template. code='prompt_unavailable'."""

    def __init__(self, name: str):
        super().__init__('prompt_unavailable', f'Langfuse prompt {name} could not be fetched for a pinned run')


def _prompt_source() -> str:
    return get_settings().prompt_source


def _fallback(name: str, local: ChatPromptTemplate, strict: bool) -> ResolvedPrompt:
    logs.log_event('warning', 'prompt_fallback')
    if strict:
        raise PromptUnavailable(name)
    return replace(ResolvedPrompt.local(name, local), source='fallback')


def resolve_prompt(name: str, local: ChatPromptTemplate, *, label: str = 'production',
                   version: int | None = None, strict: bool = False) -> ResolvedPrompt:
    """The template to use for `name`, with identity.

    PROMPT_SOURCE=local or Langfuse off: `local`, source 'local', never an error even when strict.
    PROMPT_SOURCE=langfuse: fetch the chat prompt (version wins over label). A failed fetch falls back to `local`
    with source 'fallback' and is not retried for 60 s; with strict=True it raises PromptUnavailable instead."""
    client = tracing.langfuse_client()
    if client is None or _prompt_source() == 'local':
        return ResolvedPrompt.local(name, local)
    key = (name, version if version is not None else label)
    if time.monotonic() < _failed_until.get(key, 0):
        return _fallback(name, local, strict)
    selector = {'version': version} if version is not None else {'label': label}
    try:
        fetched = client.get_prompt(name, type='chat', cache_ttl_seconds=60, fetch_timeout_seconds=2, max_retries=1,
                                    fallback=to_langfuse_messages(local), **selector)
    except Exception:
        fetched = None
    if fetched is None or fetched.is_fallback:
        _failed_until[key] = time.monotonic() + NEGATIVE_CACHE_SECONDS
        return _fallback(name, local, strict)
    template = ChatPromptTemplate.from_messages(fetched.get_langchain_prompt())
    return ResolvedPrompt(template, name, f'lf-{fetched.version}', 'langfuse', _template_sha256(template), fetched)


def pinned_prompt_version(run_folder: Path, name: str = 'rag_answer', label: str = 'comparison-v1', *,
                          fetch: bool = True) -> int | None:
    """The Langfuse prompt version frozen for a comparison or security run, stored in prompt_version.json.

    Idempotent across resumes. null means the run started with the local template (config_hash pins it).
    fetch=False records the local template without asking Langfuse, for callers whose queries still use it."""
    path = Path(run_folder) / 'prompt_version.json'
    if path.exists():
        return json.loads(path.read_text(encoding='utf-8'))['version']
    from ..context_engineering.prompts import RAG_PROMPT
    prompt = resolve_prompt(name, RAG_PROMPT, label=label, strict=True) if fetch else ResolvedPrompt.local(name, RAG_PROMPT)
    pinned = prompt._link.version if prompt.source == 'langfuse' else None
    write_json(path, {'name': name, 'label': label, 'version': pinned, 'source': prompt.source, 'sha256': prompt.sha256})
    return pinned


def _to_langfuse_text(text: str) -> str:
    # LangChain `{x}` -> Langfuse `{{x}}`; LangChain escapes `{{`/`}}` -> literal braces.
    return re.sub(r'\{\{|\}\}|\{(\w+)\}', lambda m: '{{' + m.group(1) + '}}' if m.group(1) else m.group(0)[0], text)


def to_langfuse_messages(template: ChatPromptTemplate) -> list[dict]:
    """LangChain chat template -> Langfuse chat messages. get_langchain_prompt() reverses it."""
    return [{'role': _ROLES[type(m).__name__], 'content': _to_langfuse_text(m.prompt.template)} for m in template.messages]
