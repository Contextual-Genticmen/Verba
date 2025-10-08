# End-to-End Tests for MCP RAG Server

This directory contains end-to-end tests for the MCP RAG server, demonstrating real-world integration scenarios with file operations and RAG workflows.

## Overview

The test suite provides comprehensive end-to-end testing of the MCP RAG server including:

- **File Operations**: Reading and processing real code files
- **MCP Client Integration**: Using the MCP Python SDK to connect to the server
- **RAG Operations**: Complete workflow testing (chunking, indexing, retrieval, generation)
- **CLI Query Experience**: Simulating user interactions via command-line interface
- **Memory Management**: Testing conversation history and context maintenance
- **Multimodal Processing**: Handling multiple content types
- **LangChain Integration**: Examples of using LangChain with MCP tools

## Test Files

- **`test_e2e_file_operations.py`**: Main end-to-end test suite with file operations
- **`test_langchain_integration.py`**: Examples of LangChain integration with MCP RAG server
- **`test_resources/sample_code.py`**: Sample Python code file for testing
- **`test_requirements.txt`**: Python dependencies for running tests
- **`README.md`**: This file

## Prerequisites

### Required Dependencies

Install the test dependencies:

```bash
pip install -r test_requirements.txt
```

Or install the full Verba package with MCP support:

```bash
pip install goldenverba[mcp]
```

### System Requirements

- Python 3.10+ (< 3.13)
- MCP SDK >= 1.16.0
- pytest >= 7.4.0
- pytest-asyncio >= 0.21.0

## Running the Tests

### Run All Tests with pytest

```bash
# From the repository root
pytest mcp_rag_server/tests/test_e2e_file_operations.py -v

# Or from the tests directory
cd mcp_rag_server/tests
pytest test_e2e_file_operations.py -v
```

### Run Standalone Test (Development Mode)

The test files can be run directly for development and debugging:

```bash
# Run main E2E test
python mcp_rag_server/tests/test_e2e_file_operations.py

# Run LangChain integration examples
python mcp_rag_server/tests/test_langchain_integration.py
```

These will execute simplified versions of the tests without requiring pytest, showing
demonstrations of the MCP RAG server capabilities even without full dependencies.

### Run Specific Test Cases

```bash
# Run only file read and chunk test
pytest mcp_rag_server/tests/test_e2e_file_operations.py::TestMCPRAGServerEndToEnd::test_file_read_and_chunk_operations -v

# Run CLI query experience test
pytest mcp_rag_server/tests/test_e2e_file_operations.py::TestMCPRAGServerEndToEnd::test_cli_query_experience -v

# Run memory search test
pytest mcp_rag_server/tests/test_e2e_file_operations.py::TestMCPRAGServerEndToEnd::test_memory_search_across_sessions -v
```

## Test Structure

### Test Resource Files

- **`test_resources/sample_code.py`**: A real Python code file used for testing file operations and RAG processing

### Test Cases

1. **`test_file_read_and_chunk_operations`**
   - Reads a Python source file
   - Chunks the content using Verba's TokenChunker
   - Verifies chunks are created correctly

2. **`test_file_index_and_retrieve_operations`**
   - Indexes file content into the vector store
   - Performs retrieval queries
   - Validates relevant context is returned

3. **`test_cli_query_experience`**
   - Simulates a complete CLI interaction
   - Tests multi-turn conversations
   - Verifies memory persistence across queries

4. **`test_multimodal_with_file`**
   - Processes file content as multimodal input
   - Combines text and document media types
   - Tests multimodal query handling

5. **`test_memory_search_across_sessions`**
   - Tests memory storage across multiple queries
   - Validates semantic memory search
   - Verifies context maintenance

### LangChain Integration Examples (test_langchain_integration.py)

1. **`demo_cli_with_mcp_and_files`**
   - Demonstrates CLI workflow with file operations
   - Shows complete RAG pipeline from file to answer
   - Example of production-ready integration

2. **`demo_with_community_filesystem_tools`**
   - Shows how to combine multiple MCP servers
   - Integrates filesystem server with RAG server
   - Example code for using @modelcontextprotocol/server-filesystem

3. **`create_langchain_agent_with_mcp`**
   - Creates LangChain agent with MCP RAG tools
   - Wraps MCP tools as LangChain tools
   - Enables LangChain agent capabilities with RAG

## Integration with MCP Filesystem Tools

While this test focuses on the MCP RAG server's built-in capabilities, it can be extended to use community MCP filesystem tools:

### Using MCP Filesystem Server

The test can be extended to integrate with the official MCP filesystem server:

```python
# Example integration (requires @modelcontextprotocol/server-filesystem)
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def test_with_filesystem_server():
    # Connect to MCP filesystem server
    fs_server_params = StdioServerParameters(
        command="npx",
        args=["-y", "@modelcontextprotocol/server-filesystem", "/path/to/files"]
    )
    
    async with stdio_client(fs_server_params) as (fs_read, fs_write):
        async with ClientSession(fs_read, fs_write) as fs_session:
            await fs_session.initialize()
            
            # Use filesystem tools to read files
            result = await fs_session.call_tool(
                "read_file",
                arguments={"path": "/path/to/code.py"}
            )
            
            file_content = result.content[0].text
            
            # Now process with RAG server
            # ... (rest of test)
```

### Community Tools Integration

The test architecture supports integration with popular community MCP tools:

- **@modelcontextprotocol/server-filesystem**: Official filesystem server (Node.js)
- **langchain-mcp-tools**: LangChain integration for MCP tools
- **mcp-text-editor**: Advanced text file editing capabilities

## Expected Output

When running the tests, you should see output similar to:

```
=== Starting CLI Query Experience Test ===

[Step 1] Indexing document...
  Result: Successfully indexed 1 documents using OpenAIEmbedder

[Step 2] User asks: 'What is the DocumentProcessor class?'
  Retrieved: Retrieved 1 relevant documents for query: What is the DocumentProcessor class?

[Step 3] Generating response...
  Response: Generated response for 'What is the DocumentProcessor class?' with context length 150 characters and 0 previous messages

[Step 4] User asks follow-up: 'How does it chunk documents?'
  Retrieved: Retrieved 1 relevant documents for query: How does it chunk documents?

[Step 5] Checking conversation history...
  Found 4 conversation turns in memory

✓ CLI query experience test completed successfully
```

## Troubleshooting

### MCP SDK Not Available

If you see "MCP SDK not available", install it:

```bash
pip install mcp>=1.16.0
```

### Server Components Not Available

If server components fail to import:

```bash
# Install Verba with MCP support
pip install goldenverba[mcp]

# Or install dependencies manually
pip install mcp>=1.16.0 mem0ai>=0.1.118
```

### Import Errors

Make sure you're running from the repository root:

```bash
cd /path/to/Verba
python -m pytest mcp_rag_server/tests/test_e2e_file_operations.py
```

### Async Test Issues

If pytest doesn't recognize async tests, ensure pytest-asyncio is installed:

```bash
pip install pytest-asyncio>=0.21.0
```

## Contributing

When adding new test cases:

1. Follow the existing test structure
2. Use descriptive test names that explain what is being tested
3. Add docstrings explaining the test scenario
4. Include print statements for debugging
5. Ensure tests can run standalone (without pytest) for development

## References

- [MCP Python SDK Documentation](https://github.com/modelcontextprotocol/python-sdk)
- [MCP Filesystem Server](https://github.com/modelcontextprotocol/servers/tree/main/src/filesystem)
- [LangChain MCP Integration](https://python.langchain.com/docs/integrations/tools/toolbox/)
- [Verba Documentation](https://github.com/weaviate/Verba)
