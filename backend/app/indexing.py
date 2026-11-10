"""기존 import 경로 호환용. 실제 구현은 담당 패키지에서 수정하세요."""
import sys
from backend.app.retrieval import service as _implementation
sys.modules[__name__] = _implementation
