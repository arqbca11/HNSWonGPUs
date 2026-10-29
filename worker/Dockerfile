# Build context is the REPO ROOT (the worker depends on builder/ by path):
#   docker build -f worker/Dockerfile -t kgpu-worker .
FROM python:3.12-slim AS build
COPY --from=ghcr.io/astral-sh/uv:0.9 /uv /usr/local/bin/uv
# hnswlib compiles from source (no wheel): a compiler for this stage only
RUN apt-get update && apt-get install -y --no-install-recommends g++ && rm -rf /var/lib/apt/lists/*
WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
# the builder package (a path dependency of the worker), then the worker
COPY builder/pyproject.toml builder/README.md /app/builder/
COPY builder/kgpu_builder /app/builder/kgpu_builder
COPY worker/pyproject.toml worker/uv.lock /app/worker/
WORKDIR /app/worker
RUN uv sync --frozen --no-dev --no-install-project
COPY worker/kgpu_worker /app/worker/kgpu_worker
RUN uv sync --frozen --no-dev

FROM python:3.12-slim
WORKDIR /app/worker
COPY --from=build /app /app
ENV PATH="/app/worker/.venv/bin:$PATH"
# tini as PID 1: forwards SIGTERM to python (so the release-on-shutdown path
# runs) and reaps zombies. It also means python is NOT PID 1, so a SIGKILL
# sent to it from inside the container is a real crash: the kernel ignores
# SIGKILL aimed at a PID namespace's init from within. Compose had init: true
# for this; Kubernetes has no such flag, so the image carries its own init.
RUN apt-get update && apt-get install -y --no-install-recommends tini && rm -rf /var/lib/apt/lists/*
RUN useradd --no-create-home --uid 10001 worker
USER worker
ENTRYPOINT ["/usr/bin/tini", "--", "python", "-m", "kgpu_worker"]
