# MCP RAG Server

A Model Context Protocol (MCP) server that exposes Verba's RAG capabilities as standardized tools, with integrated mem0 memory management for multi-modal queries.

## Overview

The MCP RAG Server provides a unified interface to Verba's RAG pipeline through the Model Context Protocol, enabling:

- **Document Processing**: Chunking documents using various strategies
- **Vector Indexing**: Embedding and storing documents in vector databases  
- **Intelligent Retrieval**: Finding relevant context with memory enhancement
- **Response Generation**: Creating responses with conversation memory
- **Multi-modal Support**: Processing text, images, audio, and other media
- **Memory Management**: Persistent memory across sessions using mem0

## Features

### RAG Tools Available

1. **chunk_documents** - Split documents into optimized chunks
2. **index_documents** - Embed and store documents in vector database
3. **retrieve_context** - Find relevant context with memory enhancement
4. **generate_response** - Generate responses with conversation history
5. **multimodal_query** - Process text + media queries  
6. **get_memory_context** - Retrieve user memory context
7. **search_memories** - Semantic search through memories

### Memory Management

- Conversation history tracking
- User preferences and facts storage
- Context-aware memory retrieval
- Semantic memory search
- Persistent storage via mem0

## Installation

1. Install the required dependencies:

```bash
pip install mcp mem0ai
```

2. Ensure Verba is properly installed and configured.

3. Add the MCP RAG server to your Verba installation.

## Configuration

Create a configuration file (`config.json`):

```json
{
  "server": {
    "name": "verba-rag-server",
    "version": "1.0.0",
    "log_level": "INFO"
  },
  "weaviate": {
    "url": "http://localhost:8080",
    "key": null
  },
  "mem0": {
    "vector_store": {
      "provider": "qdrant",
      "config": {
        "collection_name": "verba_memories",
        "host": "localhost", 
        "port": 6333
      }
    },
    "llm": {
      "provider": "openai",
      "config": {
        "model": "gpt-4o-mini",
        "temperature": 0.1
      }
    },
    "embedder": {
      "provider": "openai",
      "config": {
        "model": "text-embedding-ada-002"
      }
    }
  },
  "rag": {
    "default_chunker": "TokenChunker",
    "default_embedder": "OpenAIEmbedder",
    "default_retriever": "BasicRetriever", 
    "default_generator": "OpenAIGenerator",
    "chunk_size": 500,
    "chunk_overlap": 50,
    "max_results": 10
  }
}
```

## Usage

### Command Line Interface

Generate a default configuration:

```bash
python -m mcp_rag_server.cli --create-config config.json
```

Run the server:

```bash
python -m mcp_rag_server.cli --config config.json
```

Or with environment variables:

```bash
export WEAVIATE_URL="http://localhost:8080"
export OPENAI_API_KEY="your-openai-key"
export QDRANT_URL="http://localhost:6333"

python -m mcp_rag_server.cli
```

### Programmatic Usage

```python
import asyncio
from mcp_rag_server.server import MCPRAGServer

async def main():
    # Initialize server
    server = MCPRAGServer(
        name="my-rag-server",
        weaviate_url="http://localhost:8080",
        mem0_config={
            "vector_store": {"provider": "qdrant"},
            "llm": {"provider": "openai"}
        }
    )
    
    # Run server
    await server.run(transport_type="stdio")

if __name__ == "__main__":
    asyncio.run(main())
```

### Tool Usage Examples

#### Chunk Documents

```json
{
  "name": "chunk_documents",
  "arguments": {
    "documents": [
      {
        "name": "example.txt",
        "content": "This is a long document that needs to be chunked...",
        "type": "text"
      }
    ],
    "chunker": "TokenChunker",
    "config": {
      "chunk_size": 500,
      "chunk_overlap": 50
    }
  }
}
```

#### Retrieve with Memory Context

```json
{
  "name": "retrieve_context", 
  "arguments": {
    "query": "What did we discuss about machine learning?",
    "user_id": "user123",
    "config": {
      "max_results": 5,
      "similarity_threshold": 0.7
    }
  }
}
```

#### Multi-modal Query

```json
{
  "name": "multimodal_query",
  "arguments": {
    "text_query": "Describe this image and relate it to our conversation",
    "media": [
      {
        "type": "image",
        "content": "base64_encoded_image_data"
      }
    ],
    "user_id": "user123"
  }
}
```

## Environment Variables

- `WEAVIATE_URL` - Weaviate instance URL
- `WEAVIATE_API_KEY` - Weaviate API key
- `OPENAI_API_KEY` - OpenAI API key for LLM and embeddings
- `QDRANT_URL` - Qdrant instance URL for mem0
- `QDRANT_API_KEY` - Qdrant API key
- `MCP_RAG_SERVER_NAME` - Server name override
- `MCP_RAG_LOG_LEVEL` - Logging level

## Architecture

The MCP RAG Server is built on several key components:

### Server Components

- **MCPRAGServer**: Main server implementation with MCP protocol handlers
- **Mem0MemoryManager**: Memory management using mem0 with fallback storage
- **RAG Tools**: Tool definitions and execution logic
- **Configuration**: Flexible configuration from files and environment

### Integration Points

- **Verba Manager**: Access to chunkers, embedders, retrievers, generators
- **Weaviate**: Vector database for document storage
- **mem0**: Memory management for user context and conversation history
- **MCP Protocol**: Standardized tool interface

## Development

### Running Tests

```bash
# Install test dependencies
pip install pytest pytest-asyncio

# Run tests
pytest mcp_rag_server/tests/
```

### Adding New Tools

1. Define the tool schema in `tools.py`
2. Implement the tool handler in `server.py`
3. Add tests in `tests/`
4. Update documentation

### Memory Storage

The system uses mem0 for advanced memory management with automatic fallback to in-memory storage when mem0 is not available. This ensures the server works even without full mem0 setup.

## Troubleshooting

### Common Issues

1. **mem0 connection errors**: Check Qdrant/vector store configuration
2. **Weaviate connection issues**: Verify URL and API key
3. **OpenAI API errors**: Check API key and model availability
4. **Missing dependencies**: Ensure all required packages are installed

### Logs

Set `MCP_RAG_LOG_LEVEL=DEBUG` for detailed logging.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## License

This project follows the same license as the parent Verba project.