.PHONY: install sync run dev clean test lint format

# Install dependencies
install:
	uv sync

# Sync dependencies (same as install)
sync:
	uv sync --all-groups

# Run the application
run:
	./scripts/run_local.sh

# Run in development mode with auto-reload
dev:
	uv run uvicorn goldenverba.server.api:app --host 0.0.0.0 --port 8000 --reload

# Clean up build artifacts
clean:
	rm -rf build/ dist/ *.egg-info/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

# Run tests
test:
	uv run pytest

# Run linting
lint:
	uv run ruff check .
	uv run mypy goldenverba/

# Format code
format:
	uv run black goldenverba/
	uv run ruff format .

# Update dependencies
update:
	uv lock --upgrade
	uv sync

# Show help
help:
	@echo "Available commands:"
	@echo "  install/sync  - Install dependencies"
	@echo "  run          - Run the application"
	@echo "  dev          - Run in development mode"
	@echo "  test         - Run tests"
	@echo "  lint         - Run linting"
	@echo "  format       - Format code"
	@echo "  clean        - Clean build artifacts"
	@echo "  update       - Update dependencies"