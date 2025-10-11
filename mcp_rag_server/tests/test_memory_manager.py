"""
Tests for Mem0MemoryManager.
"""

import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock, patch

from mcp_rag_server.memory_manager import Mem0MemoryManager


class TestMem0MemoryManager:
    """Test cases for Mem0MemoryManager."""
    
    @pytest.fixture
    def memory_manager(self):
        """Create a memory manager instance for testing."""
        return Mem0MemoryManager(config={})
    
    @pytest.mark.asyncio
    async def test_add_user_message_fallback(self, memory_manager):
        """Test adding user message with fallback storage."""
        # Since mem0 might not be fully configured, test fallback
        user_id = "test_user"
        message = "Hello, this is a test message"
        
        memory_id = await memory_manager.add_user_message(user_id, message)
        
        assert memory_id is not None
        assert user_id in memory_manager._memory_store
        assert len(memory_manager._memory_store[user_id]) == 1
        assert memory_manager._memory_store[user_id][0]["content"] == message
        assert memory_manager._memory_store[user_id][0]["role"] == "user"
    
    @pytest.mark.asyncio
    async def test_add_assistant_message_fallback(self, memory_manager):
        """Test adding assistant message with fallback storage."""
        user_id = "test_user"
        message = "This is an assistant response"
        
        memory_id = await memory_manager.add_assistant_message(user_id, message)
        
        assert memory_id is not None
        assert user_id in memory_manager._memory_store
        assert len(memory_manager._memory_store[user_id]) == 1
        assert memory_manager._memory_store[user_id][0]["content"] == message
        assert memory_manager._memory_store[user_id][0]["role"] == "assistant"
    
    @pytest.mark.asyncio
    async def test_get_conversation_history_fallback(self, memory_manager):
        """Test getting conversation history with fallback storage."""
        user_id = "test_user"
        
        # Add some messages
        await memory_manager.add_user_message(user_id, "First message")
        await memory_manager.add_assistant_message(user_id, "First response")
        await memory_manager.add_user_message(user_id, "Second message")
        
        # Get conversation history
        history = await memory_manager.get_conversation_history(user_id, limit=2)
        
        assert len(history) == 2
        assert history[0]["content"] == "First response"
        assert history[1]["content"] == "Second message"
    
    @pytest.mark.asyncio
    async def test_get_relevant_memories_fallback(self, memory_manager):
        """Test getting relevant memories with fallback search."""
        user_id = "test_user"
        
        # Add some messages
        await memory_manager.add_user_message(user_id, "I love pizza and pasta")
        await memory_manager.add_user_message(user_id, "My favorite color is blue")
        await memory_manager.add_user_message(user_id, "I work as a software engineer")
        
        # Search for relevant memories
        relevant = await memory_manager.get_relevant_memories(user_id, "food")
        
        # Should find the pizza message in fallback text search
        assert "pizza" in relevant.lower()
    
    @pytest.mark.asyncio
    async def test_search_memories_fallback(self, memory_manager):
        """Test searching memories with fallback search."""
        user_id = "test_user"
        
        # Add some messages
        await memory_manager.add_user_message(user_id, "I love programming in Python")
        await memory_manager.add_user_message(user_id, "JavaScript is also interesting")
        await memory_manager.add_user_message(user_id, "I enjoy hiking on weekends")
        
        # Search for programming-related memories
        results = await memory_manager.search_memories(user_id, "programming")
        
        assert len(results) >= 1
        assert any("Python" in result["content"] for result in results)
    
    @pytest.mark.asyncio
    async def test_get_user_stats(self, memory_manager):
        """Test getting user statistics."""
        user_id = "test_user"
        
        # Add some messages
        await memory_manager.add_user_message(user_id, "Test message 1")
        await memory_manager.add_assistant_message(user_id, "Test response 1")
        
        # Get stats
        stats = await memory_manager.get_user_stats(user_id)
        
        assert stats["user_id"] == user_id
        assert stats["total_memories"] == 2
        assert stats["has_mem0"] == (memory_manager.memory_client is not None)


if __name__ == "__main__":
    pytest.main([__file__])