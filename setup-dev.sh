#!/bin/bash

# Verba Development Setup Script
echo "🚀 Setting up Verba for development..."

# Check if we're in the right directory
if [ ! -f "docker-compose.yml" ]; then
    echo "❌ Please run this script from the Verba root directory"
    exit 1
fi

echo "📦 Installing frontend dependencies..."
cd frontend
npm install

echo "🏗️ Building frontend for production..."
npm run build:production

echo "🐳 Starting backend services..."
cd ..
docker compose up -d

echo "✅ Setup complete!"
echo ""
echo "🌐 Access Verba at: http://localhost:8000"
echo ""
echo "For development:"
echo "  Frontend dev: cd frontend && npm run dev (http://localhost:3000)"
echo "  Backend logs: docker compose logs -f"