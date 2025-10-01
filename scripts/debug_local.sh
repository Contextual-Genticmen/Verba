#!/bin/bash
# Local debugging script for Verba
# Runs Verba locally with Python while Weaviate runs in Docker

set -e

echo "======================================"
echo "Verba Local Debug Environment"
echo "======================================"
echo ""

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if we're in the right directory
if [ ! -f "pyproject.toml" ]; then
    echo "Error: Must be run from Verba root directory"
    exit 1
fi

# Set environment variables for local development
export WEAVIATE_URL_VERBA="http://localhost:8080"
export WEAVIATE_API_KEY_VERBA=""
export DEFAULT_DEPLOYMENT="Local"
export VERBA_PRODUCTION="Local"

echo -e "${BLUE}Step 1: Checking Weaviate container...${NC}"
if docker ps | grep -q verba-weaviate-1; then
    echo -e "${GREEN}✓ Weaviate container is running${NC}"
else
    echo -e "${YELLOW}⚠ Weaviate container not found, starting it...${NC}"
    docker-compose up -d weaviate
    echo "Waiting for Weaviate to be ready..."
    sleep 5
fi

echo ""
echo -e "${BLUE}Step 2: Installing/updating Python dependencies with uv...${NC}"
uv pip install -e .

echo ""
echo -e "${BLUE}Step 3: Starting Verba locally...${NC}"
echo -e "${GREEN}Environment:${NC}"
echo "  WEAVIATE_URL_VERBA: $WEAVIATE_URL_VERBA"
echo "  DEFAULT_DEPLOYMENT: $DEFAULT_DEPLOYMENT"
echo "  VERBA_PRODUCTION: $VERBA_PRODUCTION"
echo ""
echo -e "${YELLOW}Starting FastAPI server with hot reload...${NC}"
echo -e "${YELLOW}Access the app at: http://localhost:8000${NC}"
echo -e "${YELLOW}Press Ctrl+C to stop${NC}"
echo ""

# Run with uvicorn directly for hot reload
uv run uvicorn goldenverba.server.api:app --reload --host 0.0.0.0 --port 8000
