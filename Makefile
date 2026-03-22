.PHONY: setup dev test lint db-up db-down docker-up docker-down migrate frontend

setup:
	python -m venv .venv
	.venv/bin/pip install -e ".[dev]"
	cp -n .env.example .env || true

dev:
	.venv/bin/uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

test:
	.venv/bin/pytest tests/ -v --cov=. --cov-report=term-missing

lint:
	.venv/bin/ruff check .
	.venv/bin/ruff format --check .

format:
	.venv/bin/ruff check --fix .
	.venv/bin/ruff format .

typecheck:
	.venv/bin/mypy config core services api

db-up:
	docker compose up -d postgres redis

db-down:
	docker compose down

docker-up:
	docker compose up -d

docker-down:
	docker compose down -v

migrate:
	.venv/bin/alembic upgrade head

migrate-create:
	.venv/bin/alembic revision --autogenerate -m "$(msg)"

worker:
	.venv/bin/python -m services.worker

frontend:
	cd frontend && npm run dev
