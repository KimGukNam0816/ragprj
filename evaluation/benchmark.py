"""기존 CLI 호환 경로."""
import sys
from .retrieval import benchmark as _implementation
sys.modules[__name__] = _implementation
