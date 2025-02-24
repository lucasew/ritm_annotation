ARG PYTHON_VERSION=3.12

# Use a Python image with uv pre-installed
FROM ghcr.io/astral-sh/uv:python$PYTHON_VERSION-bookworm-slim

# Copy from the cache instead of linking since it's a mounted volume
ENV UV_LINK_MODE=copy

RUN apt update && apt install -y libgl1-mesa-glx libglib2.0-0
WORKDIR /app
COPY . /app

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --directory /app

ENTRYPOINT ["uv", "run", "--directory", "/app", "ritm_annotation"]
