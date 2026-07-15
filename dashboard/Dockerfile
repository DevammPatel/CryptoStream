FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Build context is the repo root (see docker-compose.yml) so we can pull in the
# dashboard plus the llm agent it imports.
COPY dashboard/app.py /app/dashboard/app.py
COPY llm /app/llm
ENV PYTHONPATH=/app

EXPOSE 8501
CMD ["streamlit", "run", "/app/dashboard/app.py", "--server.address=0.0.0.0", "--server.port=8501"]
