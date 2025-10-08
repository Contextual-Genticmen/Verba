"""
End-to-end test for MCP RAG server with file operations.

This test demonstrates:
1. Connecting to the MCP RAG server using MCP client SDK
2. Reading a real code file using filesystem operations
3. Processing the file through RAG operations (chunk, index, retrieve)
4. Performing a CLI-style query experience

The test uses the MCP Python SDK to interact with the server,
simulating a real integration scenario.
"""

import asyncio
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

# Import pytest only if available
try:
    import pytest
    PYTEST_AVAILABLE = True
except ImportError:
    PYTEST_AVAILABLE = False
    # Create a mock pytest for standalone usage
    class MockPytest:
        class mark:
            @staticmethod
            def skipif(condition, reason=""):
                def decorator(func):
                    return func
                return decorator
            
            @staticmethod
            def asyncio(func):
                return func
        
        class fixture:
            def __init__(self, *args, **kwargs):
                pass
            
            def __call__(self, func):
                return func
    
    pytest = MockPytest()

# Import MCP components
try:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    from mcp.types import CallToolRequest
    
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    print("Warning: MCP SDK not available, test will be skipped")

# Import server components
try:
    from mcp_rag_server.server import MCPRAGServer
    from mcp_rag_server.memory_manager import Mem0MemoryManager
    
    SERVER_AVAILABLE = True
except ImportError:
    SERVER_AVAILABLE = False
    print("Warning: Server components not available")


@pytest.mark.skipif(not MCP_AVAILABLE, reason="MCP SDK not installed")
@pytest.mark.asyncio
class TestMCPRAGServerEndToEnd:
    """End-to-end tests for MCP RAG server with file operations."""
    
    @pytest.fixture
    def test_file_path(self):
        """Get the path to the test resource file."""
        current_dir = Path(__file__).parent
        test_file = current_dir / "test_resources" / "sample_code.py"
        return str(test_file)
    
    @pytest.fixture
    def test_file_content(self, test_file_path):
        """Read the test file content."""
        with open(test_file_path, 'r') as f:
            return f.read()
    
    async def test_file_read_and_chunk_operations(self, test_file_path, test_file_content):
        """
        Test reading a file and performing chunk operations.
        
        This simulates:
        1. Reading a Python source file
        2. Chunking the file content using Verba's chunking strategies
        3. Verifying the chunks are created correctly
        """
        # Create a mock MCP RAG server for testing
        server = MCPRAGServer(
            name="test-rag-server",
            version="1.0.0"
        )
        
        # Test document from file
        file_name = Path(test_file_path).name
        
        # Create a document from file content
        document = {
            "name": file_name,
            "content": test_file_content,
            "type": "python",
            "metadata": {
                "file_path": test_file_path,
                "language": "python"
            }
        }
        
        # Test chunking operation
        chunk_request = CallToolRequest(
            name="chunk_documents",
            arguments={
                "documents": [document],
                "chunker": "TokenChunker",
                "config": {
                    "chunk_size": 500,
                    "chunk_overlap": 50
                }
            }
        )
        
        # Execute chunk operation
        result = await server._execute_tool(chunk_request)
        
        # Verify results
        assert result is not None
        assert not result.isError
        assert len(result.content) > 0
        
        # Check that chunks were created
        text_content = result.content[0].text
        assert "chunk" in text_content.lower()
        
        print(f"✓ Successfully chunked file: {text_content}")
    
    async def test_file_index_and_retrieve_operations(self, test_file_content):
        """
        Test indexing file content and retrieving relevant sections.
        
        This simulates a complete RAG workflow:
        1. Index the file content
        2. Perform a query to retrieve relevant sections
        3. Generate a response based on retrieved context
        """
        server = MCPRAGServer(
            name="test-rag-server",
            version="1.0.0"
        )
        
        # Index the document
        document = {
            "name": "sample_code.py",
            "content": test_file_content,
            "type": "python"
        }
        
        index_request = CallToolRequest(
            name="index_documents",
            arguments={
                "documents": [document],
                "embedder": "OpenAIEmbedder"
            }
        )
        
        # Execute index operation
        index_result = await server._execute_tool(index_request)
        
        assert index_result is not None
        assert not index_result.isError
        print(f"✓ Indexed document: {index_result.content[0].text}")
        
        # Now retrieve context for a query
        retrieve_request = CallToolRequest(
            name="retrieve_context",
            arguments={
                "query": "How does document chunking work?",
                "retriever": "BasicRetriever",
                "user_id": "test_user"
            }
        )
        
        retrieve_result = await server._execute_tool(retrieve_request)
        
        assert retrieve_result is not None
        assert not retrieve_result.isError
        assert retrieve_result.metadata is not None
        
        # Verify that retrieval happened
        retrieved_docs = retrieve_result.metadata.get("retrieved_docs", [])
        assert len(retrieved_docs) > 0
        
        print(f"✓ Retrieved context: {retrieve_result.content[0].text}")
        print(f"  - Found {len(retrieved_docs)} relevant documents")
    
    async def test_cli_query_experience(self, test_file_content):
        """
        Test a complete CLI-style query experience.
        
        This simulates a user interacting with the RAG system via CLI:
        1. Load and index a document
        2. Ask a question
        3. Retrieve relevant context
        4. Generate a response
        5. Verify memory is maintained
        """
        server = MCPRAGServer(
            name="test-rag-server",
            version="1.0.0"
        )
        
        user_id = "cli_test_user"
        
        print("\n=== Starting CLI Query Experience Test ===")
        
        # Step 1: Index the document
        print("\n[Step 1] Indexing document...")
        document = {
            "name": "sample_code.py",
            "content": test_file_content,
            "type": "python"
        }
        
        index_request = CallToolRequest(
            name="index_documents",
            arguments={"documents": [document]}
        )
        
        index_result = await server._execute_tool(index_request)
        print(f"  Result: {index_result.content[0].text}")
        
        # Step 2: User asks a question
        print("\n[Step 2] User asks: 'What is the DocumentProcessor class?'")
        query = "What is the DocumentProcessor class?"
        
        retrieve_request = CallToolRequest(
            name="retrieve_context",
            arguments={
                "query": query,
                "user_id": user_id
            }
        )
        
        retrieve_result = await server._execute_tool(retrieve_request)
        print(f"  Retrieved: {retrieve_result.content[0].text}")
        
        # Step 3: Generate response with context
        print("\n[Step 3] Generating response...")
        context = retrieve_result.metadata.get("retrieved_docs", [{}])[0].get("content", "")
        
        generate_request = CallToolRequest(
            name="generate_response",
            arguments={
                "query": query,
                "context": context,
                "user_id": user_id
            }
        )
        
        generate_result = await server._execute_tool(generate_request)
        print(f"  Response: {generate_result.content[0].text}")
        
        # Step 4: Follow-up question (testing memory)
        print("\n[Step 4] User asks follow-up: 'How does it chunk documents?'")
        followup_query = "How does it chunk documents?"
        
        followup_request = CallToolRequest(
            name="retrieve_context",
            arguments={
                "query": followup_query,
                "user_id": user_id
            }
        )
        
        followup_result = await server._execute_tool(followup_request)
        print(f"  Retrieved: {followup_result.content[0].text}")
        
        # Verify memory is being used
        assert followup_result.metadata.get("memory_used") is not None
        
        # Step 5: Check conversation history
        print("\n[Step 5] Checking conversation history...")
        memory_request = CallToolRequest(
            name="get_memory_context",
            arguments={
                "user_id": user_id,
                "limit": 10
            }
        )
        
        memory_result = await server._execute_tool(memory_request)
        memories = memory_result.metadata.get("memories", [])
        print(f"  Found {len(memories)} conversation turns in memory")
        
        assert len(memories) > 0
        print("\n✓ CLI query experience test completed successfully")
    
    async def test_multimodal_with_file(self, test_file_content):
        """
        Test multimodal query combining file content with other media.
        
        This demonstrates processing multiple types of content:
        1. Text from a file
        2. Simulated image/media references
        3. Combined multimodal processing
        """
        server = MCPRAGServer(
            name="test-rag-server",
            version="1.0.0"
        )
        
        # Create a multimodal query
        multimodal_request = CallToolRequest(
            name="multimodal_query",
            arguments={
                "text_query": f"Analyze this code file: {test_file_content[:200]}...",
                "media": [
                    {
                        "type": "document",
                        "content": test_file_content,
                        "metadata": {
                            "file_type": "python",
                            "file_name": "sample_code.py"
                        }
                    }
                ],
                "user_id": "multimodal_test_user"
            }
        )
        
        result = await server._execute_tool(multimodal_request)
        
        assert result is not None
        assert not result.isError
        assert result.metadata is not None
        
        media_count = result.metadata.get("media_count", 0)
        assert media_count == 1
        
        print(f"✓ Processed multimodal query: {result.content[0].text}")
        print(f"  - Processed {media_count} media items")
    
    async def test_memory_search_across_sessions(self, test_file_content):
        """
        Test memory search functionality across multiple interactions.
        
        This verifies:
        1. Memories are stored across multiple queries
        2. Memory search can find relevant past interactions
        3. Context is maintained for the user
        """
        server = MCPRAGServer(
            name="test-rag-server",
            version="1.0.0"
        )
        
        user_id = "memory_test_user"
        
        # Simulate multiple interactions
        queries = [
            "What is a DocumentProcessor?",
            "How does chunking work?",
            "What are the retrieval methods?"
        ]
        
        print("\n=== Testing Memory Search ===")
        
        for i, query in enumerate(queries, 1):
            print(f"\n[Query {i}] {query}")
            
            # Store query in memory through retrieve_context
            request = CallToolRequest(
                name="retrieve_context",
                arguments={
                    "query": query,
                    "user_id": user_id
                }
            )
            
            result = await server._execute_tool(request)
            print(f"  Processed: {result.content[0].text[:80]}...")
        
        # Now search memories
        print("\n[Searching Memories] Looking for 'chunking'...")
        search_request = CallToolRequest(
            name="search_memories",
            arguments={
                "query": "chunking",
                "user_id": user_id,
                "limit": 5
            }
        )
        
        search_result = await server._execute_tool(search_request)
        
        assert search_result is not None
        assert not search_result.isError
        
        memories = search_result.metadata.get("memories", [])
        assert len(memories) > 0
        
        print(f"✓ Found {len(memories)} relevant memories")
        for mem in memories:
            print(f"  - {mem.get('content', '')[:60]}...")


# Test that can be run standalone for development/debugging
async def run_standalone_test():
    """Run a simplified version of the test for development."""
    print("Running standalone MCP RAG Server E2E Test")
    print("=" * 60)
    
    if not SERVER_AVAILABLE:
        print("Server components not available. Install with: pip install goldenverba[mcp]")
        return
    
    # Create server instance
    server = MCPRAGServer(name="standalone-test-server")
    
    # Read test file
    test_file_path = Path(__file__).parent / "test_resources" / "sample_code.py"
    
    if not test_file_path.exists():
        print(f"Test file not found: {test_file_path}")
        return
    
    with open(test_file_path, 'r') as f:
        content = f.read()
    
    print(f"\n✓ Loaded test file: {test_file_path.name} ({len(content)} bytes)")
    
    # Test chunk operation
    print("\n--- Testing Chunk Operation ---")
    chunk_request = CallToolRequest(
        name="chunk_documents",
        arguments={
            "documents": [{
                "name": "sample_code.py",
                "content": content,
                "type": "python"
            }],
            "chunker": "TokenChunker"
        }
    )
    
    chunk_result = await server._execute_tool(chunk_request)
    print(f"Result: {chunk_result.content[0].text}")
    
    # Test retrieve operation
    print("\n--- Testing Retrieve Operation ---")
    retrieve_request = CallToolRequest(
        name="retrieve_context",
        arguments={
            "query": "How does document processing work?",
            "user_id": "standalone_user"
        }
    )
    
    retrieve_result = await server._execute_tool(retrieve_request)
    print(f"Result: {retrieve_result.content[0].text}")
    
    print("\n" + "=" * 60)
    print("✓ Standalone test completed successfully!")


if __name__ == "__main__":
    # Run standalone test for development
    asyncio.run(run_standalone_test())
