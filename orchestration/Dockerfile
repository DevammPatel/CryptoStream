FROM python:3.11-slim

WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Build context is the repo root (see docker-compose.yml): the pipeline runs dbt,
# the trainer and the LLM agent as well as the Dagster code.
COPY orchestration/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install DuckDB's S3 extension now so the container needs no download at runtime.
RUN python -c "import duckdb; duckdb.connect().execute('INSTALL httpfs')"

COPY ml /app/ml
COPY llm /app/llm
COPY warehouse /app/warehouse
COPY orchestration/cryptostream_dagster /app/orchestration/cryptostream_dagster
ENV PYTHONPATH=/app/orchestration
ENV DAGSTER_HOME=/opt/dagster
RUN mkdir -p /opt/dagster

EXPOSE 3000
CMD ["dagster", "dev", "-h", "0.0.0.0", "-p", "3000", "-m", "cryptostream_dagster.definitions"]
