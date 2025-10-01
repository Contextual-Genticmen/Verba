.PHONY: install sync run dev clean test lint format docker-build docker-up docker-down docker-restart docker-logs docker-ps debug-setup debug debug-stop validate-sources

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

# Debug Setup - Stop full app and start only Weaviate
debug-setup:
	@echo "Setting up local debug environment..."
	docker-compose down
	docker-compose up -d weaviate
	@echo "✓ Weaviate started on http://localhost:8080"
	@echo ""
	@echo "Next steps:"
	@echo "  1. Run 'make debug' to start Verba locally"
	@echo "  2. Or run 'make validate-sources' to test configuration"

# Debug - Run Verba locally with hot reload (Weaviate in container)
debug:
	@echo "Starting Verba in debug mode..."
	@./scripts/debug_local.sh

# Stop local debug (keep Weaviate running)
debug-stop:
	@echo "Stopping local Verba (Weaviate remains running)"
	@pkill -f "uvicorn goldenverba.server.api" || true
	@echo "✓ Local Verba stopped"

# Validate default sources configuration
validate-sources:
	@echo "Validating default_sources.yaml..."
	@uv run python scripts/validate_default_sources.py

# Docker Compose commands
docker-build:
	docker-compose build

docker-up:
	docker-compose up -d

docker-down:
	docker-compose down

docker-restart:
	docker-compose restart

docker-logs:
	docker-compose logs -f

docker-ps:
	docker-compose ps

# Start the full application with Docker Compose
start:
	docker-compose up -d
	@echo "Application started! Access it at:"
	@echo "  Verba: http://localhost:8000"
	@echo "  Weaviate: http://localhost:8080"
	@echo ""
	@echo "To view logs: make docker-logs"
	@echo "To stop: make docker-down"

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
	@echo ""
	@echo "Development:"
	@echo "  install/sync     - Install dependencies"
	@echo "  run              - Run the application"
	@echo "  dev              - Run in development mode"
	@echo ""
	@echo "Local Debugging:"
	@echo "  debug-setup      - Stop Docker app, start only Weaviate"
	@echo "  debug            - Run Verba locally with hot reload"
	@echo "  debug-stop       - Stop local Verba (keep Weaviate)"
	@echo "  validate-sources - Validate default_sources.yaml"
	@echo ""
	@echo "Docker Compose:"
	@echo "  start            - Start full application with Docker"
	@echo "  docker-build     - Build Docker images"
	@echo "  docker-up        - Start Docker Compose services"
	@echo "  docker-down      - Stop Docker Compose services"
	@echo "  docker-restart   - Restart Docker Compose services"
	@echo "  docker-logs      - View Docker Compose logs"
	@echo "  docker-ps        - Show Docker Compose status"
	@echo ""
	@echo "Code Quality:"
	@echo "  test             - Run tests"
	@echo "  lint             - Run linting"
	@echo "  format           - Format code"
	@echo "  clean            - Clean build artifacts"
	@echo "  update           - Update dependencies"