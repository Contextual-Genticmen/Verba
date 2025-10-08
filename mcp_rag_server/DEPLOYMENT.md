# MCP RAG Server Deployment Guide

This guide covers different deployment scenarios for the MCP RAG Server.

## Prerequisites

### System Requirements

- Python 3.10+ (< 3.13)
- Weaviate instance (for document storage)
- Qdrant instance (for mem0 memory storage, optional)
- OpenAI API key or compatible LLM API

### Dependencies

Install the base Verba package with MCP support:

```bash
pip install goldenverba[mcp]
```

Or install manually:

```bash
pip install mcp>=1.16.0 mem0ai>=0.1.118
```

## Configuration

### 1. Environment Variables

Set required environment variables:

```bash
export OPENAI_API_KEY="your-openai-api-key"
export WEAVIATE_URL="http://localhost:8080"
export WEAVIATE_API_KEY="your-weaviate-key"  # Optional
export QDRANT_URL="http://localhost:6333"    # Optional
export QDRANT_API_KEY="your-qdrant-key"      # Optional
```

### 2. Configuration File

Create a configuration file:

```bash
# Generate default config
python -m mcp_rag_server.cli --create-config config.json

# Edit the configuration as needed
nano config.json
```

Example configuration:

```json
{
  "server": {
    "name": "production-rag-server",
    "log_level": "INFO"
  },
  "weaviate": {
    "url": "https://your-weaviate-cluster.weaviate.network",
    "key": "your-api-key"
  },
  "mem0": {
    "vector_store": {
      "provider": "qdrant",
      "config": {
        "url": "https://your-qdrant-cluster.qdrant.tech:6333",
        "api_key": "your-qdrant-key",
        "collection_name": "production_memories"
      }
    },
    "llm": {
      "provider": "openai",
      "config": {
        "model": "gpt-4o-mini",
        "api_key": "your-openai-key"
      }
    }
  }
}
```

## Deployment Options

### 1. Local Development

For local development and testing:

```bash
# Run with default configuration
python -m mcp_rag_server.cli

# Run with custom configuration
python -m mcp_rag_server.cli --config config.json
```

### 2. Docker Deployment

Create a Dockerfile:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Install the package
RUN pip install -e .[mcp]

# Expose port (if needed for HTTP transport)
EXPOSE 8000

# Run the MCP server
CMD ["python", "-m", "mcp_rag_server.cli", "--config", "config.json"]
```

Build and run:

```bash
docker build -t verba-mcp-server .
docker run -d \
  --name verba-mcp \
  -e OPENAI_API_KEY="your-key" \
  -e WEAVIATE_URL="http://weaviate:8080" \
  -v $(pwd)/config.json:/app/config.json \
  verba-mcp-server
```

### 3. Docker Compose with Dependencies

Complete stack with Weaviate and Qdrant:

```yaml
# docker-compose.yml
version: '3.8'

services:
  weaviate:
    image: semitechnologies/weaviate:latest
    environment:
      QUERY_DEFAULTS_LIMIT: 25
      AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED: 'true'
    ports:
      - "8080:8080"
    volumes:
      - weaviate_data:/var/lib/weaviate

  qdrant:
    image: qdrant/qdrant:latest
    ports:
      - "6333:6333"
    volumes:
      - qdrant_data:/qdrant/storage

  verba-mcp:
    build: .
    depends_on:
      - weaviate
      - qdrant
    environment:
      OPENAI_API_KEY: "${OPENAI_API_KEY}"
      WEAVIATE_URL: "http://weaviate:8080"
      QDRANT_URL: "http://qdrant:6333"
    volumes:
      - ./config.json:/app/config.json
    stdin_open: true
    tty: true

volumes:
  weaviate_data:
  qdrant_data:
```

Run the stack:

```bash
docker-compose up -d
```

### 4. Kubernetes Deployment

Example Kubernetes manifests:

```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: verba-mcp-server
spec:
  replicas: 2
  selector:
    matchLabels:
      app: verba-mcp-server
  template:
    metadata:
      labels:
        app: verba-mcp-server
    spec:
      containers:
      - name: verba-mcp
        image: verba-mcp-server:latest
        env:
        - name: OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: api-keys
              key: openai-key
        - name: WEAVIATE_URL
          value: "http://weaviate-service:8080"
        - name: QDRANT_URL  
          value: "http://qdrant-service:6333"
        volumeMounts:
        - name: config
          mountPath: /app/config.json
          subPath: config.json
      volumes:
      - name: config
        configMap:
          name: mcp-config

---
apiVersion: v1
kind: ConfigMap
metadata:
  name: mcp-config
data:
  config.json: |
    {
      "server": {"name": "k8s-rag-server"},
      "weaviate": {"url": "http://weaviate-service:8080"},
      "mem0": {
        "vector_store": {"provider": "qdrant"},
        "llm": {"provider": "openai"}
      }
    }

---
apiVersion: v1
kind: Secret
metadata:
  name: api-keys
type: Opaque
data:
  openai-key: <base64-encoded-key>
```

Deploy:

```bash
kubectl apply -f deployment.yaml
```

## Production Considerations

### 1. Security

- Use secrets management for API keys
- Enable authentication on Weaviate/Qdrant
- Run containers as non-root user
- Use network policies to restrict access

### 2. Monitoring

Add health checks and monitoring:

```python
# Add to server.py
@app.get("/health")
async def health_check():
    return {"status": "healthy", "server": self.name}
```

### 3. Scaling

- Use horizontal pod autoscaler for Kubernetes
- Consider read replicas for Weaviate
- Implement connection pooling for databases

### 4. Logging

Configure structured logging:

```json
{
  "server": {
    "log_level": "INFO",
    "log_format": "json"
  }
}
```

### 5. Backup & Recovery

- Regular backups of Weaviate data
- Backup Qdrant collections
- Version control configuration files

## Integration Examples

### 1. Claude Desktop Integration

Add to Claude Desktop configuration:

```json
{
  "mcpServers": {
    "verba-rag": {
      "command": "python",
      "args": ["-m", "mcp_rag_server.cli", "--config", "/path/to/config.json"],
      "env": {
        "OPENAI_API_KEY": "your-key"
      }
    }
  }
}
```

### 2. Custom Client Integration

```python
import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def use_verba_rag():
    server_params = StdioServerParameters(
        command="python",
        args=["-m", "mcp_rag_server.cli"]
    )
    
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # Initialize
            await session.initialize()
            
            # List available tools
            tools = await session.list_tools()
            print(f"Available tools: {[t.name for t in tools.tools]}")
            
            # Use a tool
            result = await session.call_tool(
                "chunk_documents",
                arguments={
                    "documents": [{"name": "test.txt", "content": "..."}]
                }
            )
            print(result.content[0].text)
```

## Troubleshooting

### Common Issues

1. **Connection refused to Weaviate/Qdrant**
   - Check service URLs and ports
   - Verify network connectivity
   - Check firewall rules

2. **API key errors**
   - Verify environment variables are set
   - Check API key validity
   - Ensure proper permissions

3. **Memory issues**
   - Monitor memory usage
   - Adjust batch sizes in configuration
   - Consider scaling horizontally

4. **Import errors**
   - Ensure all dependencies are installed
   - Check Python version compatibility
   - Verify package installation

### Debugging

Enable debug logging:

```bash
export MCP_RAG_LOG_LEVEL=DEBUG
python -m mcp_rag_server.cli --config config.json
```

### Support

For issues and questions:

1. Check the main README.md
2. Review example configurations
3. Check container logs
4. File issues in the repository