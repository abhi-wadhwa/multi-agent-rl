FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml .
COPY src/ src/
COPY tests/ tests/
COPY examples/ examples/
COPY README.md .
COPY LICENSE .

RUN pip install --no-cache-dir -e ".[dev]"

RUN pytest tests/ -v --tb=short

EXPOSE 8501

CMD ["streamlit", "run", "src/viz/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
