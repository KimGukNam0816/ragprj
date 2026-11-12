# FastAPI 백엔드 이미지.
# 기본은 CPU 서버용이며 리랭커(Torch)는 포함하지 않는다.
# hybrid_rerank까지 컨테이너에서 쓰려면: --build-arg WITH_RERANKER=1
FROM python:3.12-slim

ARG WITH_RERANKER=0
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_PROJECT_ENVIRONMENT=/usr/local \
    UV_COMPILE_BYTECODE=1

RUN apt-get update \
 && apt-get install -y --no-install-recommends curl ca-certificates \
 && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.11 /uv /usr/local/bin/uv

WORKDIR /app
COPY pyproject.toml uv.lock ./

# torch는 기본 lock이 CUDA wheel을 가리키므로 컨테이너에서는 제외하고,
# 리랭커가 필요할 때만 CPU wheel을 따로 설치한다.
RUN uv sync --locked --extra memory --no-install-package torch --no-dev \
 && if [ "$WITH_RERANKER" = "1" ]; then \
      uv pip install --index-url https://download.pytorch.org/whl/cpu torch ; \
    fi

COPY backend ./backend
COPY evaluation ./evaluation
COPY scripts ./scripts

# 컨테이너 안에서는 네트워크 인터페이스 전체에 바인딩한다(노출 범위는 compose/리버스 프록시가 정한다).
ENV BACKEND_HOST=0.0.0.0 \
    BACKEND_PORT=8000 \
    RUNTIME_DIR=/app/runtime \
    DATA_DIR=/app/data

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=5 \
  CMD curl -fsS http://127.0.0.1:8000/health || exit 1

CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
