#!/usr/bin/env python3
"""
Basic usage example for MCP RAG Server.

This example demonstrates how to use the MCP RAG server programmatically
for document processing and memory-enhanced queries.
"""

import asyncio
import json
from mcp.types import CallToolRequest

# Import components (conditional due to dependencies)
try:
    from mcp_rag_server.server import MCPRAGServer
    from mcp_rag_server.memory_manager import Mem0MemoryManager
    from mcp_rag_server.config import MCPRAGConfig
    server_available = True
except ImportError as e:
    print(f"Warning: Could not import server components: {e}")
    server_available = False
    
    # Import just the memory manager for testing
    from mcp_rag_server.memory_manager import Mem0MemoryManager


async def test_memory_manager():
    """Test the memory manager functionality."""
    print("=== Testing Memory Manager ===")
    
    # Create memory manager
    memory_manager = Mem0MemoryManager({
        "llm_provider": "openai",
        "vector_store_provider": "qdrant"
    })
    
    user_id = "example_user"
    
    # Add some conversation
    await memory_manager.add_user_message(user_id, "I'm working on a Python project about machine learning")
    await memory_manager.add_assistant_message(user_id, "That sounds interesting! What specific area of ML are you focusing on?")
    await memory_manager.add_user_message(user_id, "I'm particularly interested in natural language processing and transformers")
    
    # Get conversation history
    history = await memory_manager.get_conversation_history(user_id, limit=5)
    print(f"Conversation history: {len(history)} messages")
    
    # Search for relevant memories
    relevant = await memory_manager.get_relevant_memories(user_id, "programming")
    print(f"Relevant memories for 'programming': {len(relevant)} characters")
    
    # Get user stats
    stats = await memory_manager.get_user_stats(user_id)
    print(f"User stats: {stats}")
    
    print("✓ Memory manager test completed\n")


async def test_configuration():
    """Test configuration loading."""
    print("=== Testing Configuration ===")
    
    # Test default configuration
    config = MCPRAGConfig()
    
    print(f"Server name: {config.get('server.name')}")
    print(f"Default chunker: {config.get('rag.default_chunker')}")
    print(f"Mem0 LLM provider: {config.get('mem0.llm.provider')}")
    
    # Test configuration validation
    is_valid = config.validate()
    print(f"Configuration is valid: {is_valid}")
    
    print("✓ Configuration test completed\n")


async def main():
    """Run all examples."""
    print("MCP RAG Server - Basic Usage Examples")
    print("=" * 50)
    
    # Test memory manager
    await test_memory_manager()
    
    # Test configuration
    await test_configuration()
    
    print("All examples completed!")


if __name__ == "__main__":
    asyncio.run(main())