#!/usr/bin/env python3
"""
Simple test script to verify the MCP RAG server setup and tools.
"""

import asyncio
import json
import logging
from mcp_rag_server.server import MCPRAGServer
from mcp_rag_server.config import load_config

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


async def test_server_initialization():
    """Test server initialization with config."""
    logger.info("Testing MCP RAG Server initialization...")
    
    try:
        # Load configuration
        config_path = "mcp_rag_server/config.json"
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        logger.info(f"Loaded config: {json.dumps(config, indent=2)}")
        
        # Initialize server
        server = MCPRAGServer(
            name=config["server"]["name"],
            version=config["server"]["version"],
            weaviate_url=config["weaviate"]["url"],
            weaviate_key=config["weaviate"].get("key"),
            mem0_config=config["mem0"]
        )
        
        logger.info("✓ Server initialized successfully!")
        logger.info(f"  Server name: {server.name}")
        logger.info(f"  Memory manager initialized: {server.memory_manager is not None}")
        logger.info(f"  Verba manager initialized: {server.verba_manager is not None}")
        
        return server
        
    except Exception as e:
        logger.error(f"✗ Failed to initialize server: {e}")
        raise


async def test_memory_manager(server):
    """Test memory manager functionality."""
    logger.info("\nTesting Memory Manager...")
    
    try:
        memory_mgr = server.memory_manager
        user_id = "test_user_123"
        
        # Test adding user message
        logger.info("  Adding user message...")
        msg_id = await memory_mgr.add_user_message(
            user_id=user_id,
            message="Hello! I'm testing the RAG system.",
            metadata={"type": "conversation"}
        )
        logger.info(f"  ✓ Added user message with ID: {msg_id}")
        
        # Test adding assistant message
        logger.info("  Adding assistant message...")
        assistant_msg_id = await memory_mgr.add_assistant_message(
            user_id=user_id,
            message="Hello! I'm here to help with your RAG queries.",
            metadata={"type": "conversation"}
        )
        logger.info(f"  ✓ Added assistant message with ID: {assistant_msg_id}")
        
        # Test retrieving conversation history
        logger.info("  Retrieving conversation history...")
        history = await memory_mgr.get_conversation_history(user_id=user_id, limit=5)
        logger.info(f"  ✓ Retrieved {len(history)} messages")
        for i, msg in enumerate(history):
            content = msg.get("content", "")[:50] + "..." if len(msg.get("content", "")) > 50 else msg.get("content", "")
            logger.info(f"    [{i+1}] {msg.get('role', 'unknown')}: {content}")
        
        # Test memory search
        logger.info("  Searching memories...")
        relevant = await memory_mgr.get_relevant_memories(
            user_id=user_id,
            query="RAG system",
            limit=3
        )
        logger.info(f"  ✓ Found relevant memories (length: {len(relevant)})")
        
        # Get user stats
        stats = await memory_mgr.get_user_stats(user_id)
        logger.info(f"  User stats: {stats}")
        
        logger.info("✓ Memory manager tests passed!")
        
    except Exception as e:
        logger.error(f"✗ Memory manager test failed: {e}")
        raise


async def test_verba_components(server):
    """Test Verba component access."""
    logger.info("\nTesting Verba Components...")
    
    try:
        verba_mgr = server.verba_manager
        
        # Test chunkers
        logger.info("  Available chunkers:")
        chunkers = verba_mgr.get_chunkers()
        for chunker in chunkers:
            logger.info(f"    - {chunker.name}")
        logger.info(f"  ✓ Found {len(chunkers)} chunkers")
        
        # Test embedders
        logger.info("  Available embedders:")
        embedders = verba_mgr.get_embedders()
        for embedder in embedders:
            logger.info(f"    - {embedder.name}")
        logger.info(f"  ✓ Found {len(embedders)} embedders")
        
        # Test retrievers
        logger.info("  Available retrievers:")
        retrievers = verba_mgr.get_retrievers()
        for retriever in retrievers:
            logger.info(f"    - {retriever.name}")
        logger.info(f"  ✓ Found {len(retrievers)} retrievers")
        
        # Test generators
        logger.info("  Available generators:")
        generators = verba_mgr.get_generators()
        for generator in generators:
            logger.info(f"    - {generator.name}")
        logger.info(f"  ✓ Found {len(generators)} generators")
        
        logger.info("✓ Verba components test passed!")
        
    except Exception as e:
        logger.error(f"✗ Verba components test failed: {e}")
        raise


async def test_tool_listing(server):
    """Test MCP tool listing."""
    logger.info("\nTesting MCP Tools...")
    
    try:
        from mcp_rag_server.tools import rag_tools
        
        logger.info(f"  Available tools: {len(rag_tools)}")
        for tool in rag_tools:
            logger.info(f"    - {tool.name}: {tool.description[:60]}...")
        
        logger.info("✓ Tool listing test passed!")
        
    except Exception as e:
        logger.error(f"✗ Tool listing test failed: {e}")
        raise


async def main():
    """Run all tests."""
    logger.info("=" * 70)
    logger.info("MCP RAG Server Test Suite")
    logger.info("=" * 70)
    
    try:
        # Initialize server
        server = await test_server_initialization()
        
        # Test components
        await test_memory_manager(server)
        await test_verba_components(server)
        await test_tool_listing(server)
        
        logger.info("\n" + "=" * 70)
        logger.info("✓ All tests passed successfully!")
        logger.info("=" * 70)
        logger.info("\nThe MCP RAG server is ready to use!")
        logger.info("\nTo run the server:")
        logger.info("  python -m mcp_rag_server.cli --config mcp_rag_server/config.json")
        
    except Exception as e:
        logger.error("\n" + "=" * 70)
        logger.error("✗ Tests failed!")
        logger.error("=" * 70)
        logger.error(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)
