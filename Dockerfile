FROM python:3.12-slim

WORKDIR /app

COPY requirements-base.txt .
RUN pip install --no-cache-dir -r requirements-base.txt

COPY src/ src/
COPY memory_mcp.py .
COPY .env .

ENV PYTHONPATH=/app/src

EXPOSE 8000

CMD ["uvicorn", "src.server:app", "--host", "0.0.0.0", "--port", "8000"]
