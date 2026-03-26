FROM python:3.12-slim

RUN useradd --create-home appuser

WORKDIR /app

# Only the deps the server actually needs (not torch/diffusers/etc.)
COPY requirements-server.txt ./
RUN pip install --no-cache-dir -r requirements-server.txt

COPY server.py ./
COPY web/ ./web/

USER appuser

EXPOSE 3002

CMD ["uvicorn", "server:app_for_reload", "--host", "0.0.0.0", "--port", "3002", "--no-access-log"]
