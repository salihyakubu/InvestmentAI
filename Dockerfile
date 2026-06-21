FROM python:3.11-slim

WORKDIR /app

# build-essential/libpq-dev build native wheels; libgomp1 is the OpenMP runtime
# that xgboost / lightgbm load at import time (the worker imports them).
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libpq-dev libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Drop root: run as an unprivileged user.
RUN useradd --create-home --uid 10001 appuser && chown -R appuser:appuser /app
USER appuser

ENV PORT=8080
EXPOSE 8080

CMD uvicorn api.main:app --host 0.0.0.0 --port $PORT
