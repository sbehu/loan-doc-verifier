FROM python:3.12-slim

WORKDIR /app

# Install uv (same tool used for local dev, keeps dependency resolution
# identical between local and deployed environments).
RUN pip install --no-cache-dir uv

# Copy dependency files first so Docker can cache this layer -- rebuilds
# are much faster when only application code changes, not dependencies.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

# Application code and the data the checks/UI need at runtime. Excludes
# dataset_creation/, experiments/, interview_prep/ -- see .dockerignore.
COPY checks/ ./checks/
COPY main.py app.py ./
COPY data/ ./data/

EXPOSE 8501

# --server.address=0.0.0.0 is required -- Streamlit defaults to
# localhost, which isn't reachable from outside the container.
CMD ["uv", "run", "streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]
