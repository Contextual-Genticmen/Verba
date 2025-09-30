#!/bin/bash
set -e

echo "🔧 Syncing dependencies with uv..."
uv sync

echo "🚀 Starting Verba server..."
uv run verba start --host 0.0.0.0 --port 8000 --no-prod