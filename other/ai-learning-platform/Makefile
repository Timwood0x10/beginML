# AIScope — one-command dev launcher & static checks.
#
#   make start      launch backend (FastAPI :8000) + frontend (Vite :5173)
#   make stop       stop both services
#   make format     format TS (prettier) + Python (ruff format)
#   make check      static checks: frontend (prettier + tsc) & backend (ruff + compile)
#   make lint       alias of `check`
#   make lint-fix   auto-fix backend ruff findings
#
# NOTE: recipes must be indented with real TAB characters.

.PHONY: start stop format check lint lint-fix \
        frontend-format backend-format frontend-check backend-check

start:
	./start.sh

stop:
	./start.sh --stop

format: frontend-format backend-format

frontend-format:
	cd frontend && npm run format

backend-format:
	cd backend && uv run ruff format app.py lab

check: frontend-check backend-check

lint: check

frontend-check:
	cd frontend && npm run format:check && npx tsc --noEmit

backend-check:
	cd backend && uv run ruff check app.py lab && uv run python -m compileall -q app.py lab

lint-fix:
	cd backend && uv run ruff check --fix app.py lab
