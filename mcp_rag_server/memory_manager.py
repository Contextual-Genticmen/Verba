"""
Memory manager using mem0 for persistent memory across RAG operations.

Provides unified memory management capabilities that can be accessed through
the MCP application for storing and retrieving user context, conversation history,
preferences, and other persistent information.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional, Union
from datetime import datetime

try:
    from mem0 import AsyncMemory, MemoryClient
except ImportError:
    # Fallback if mem0 is not available
    AsyncMemory = None
    MemoryClient = None


class Mem0MemoryManager:
    """
    Memory manager that provides unified access to mem0 capabilities.
    
    This class handles:
    - User conversation history
    - User preferences and facts  
    - Context-aware memory retrieval
    - Memory search and filtering
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the memory manager.
        
        Args:
            config: Configuration for mem0 connection and settings
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Initialize mem0 client
        self._initialize_mem0()
        
    def _initialize_mem0(self):
        """Initialize mem0 memory client."""
        # Always initialize fallback storage
        self._memory_store = {}
        
        if AsyncMemory is None:
            self.logger.warning("mem0 not available, using mock memory manager")
            self.memory_client = None
            return
            
        try:
            # Configure mem0 based on config
            mem0_config = {
                "vector_store": {
                    "provider": self.config.get("vector_store_provider", "qdrant"),
                    "config": self.config.get("vector_store_config", {
                        "collection_name": "verba_memories",
                        "host": "localhost",
                        "port": 6333
                    })
                },
                "llm": {
                    "provider": self.config.get("llm_provider", "openai"),
                    "config": self.config.get("llm_config", {
                        "model": "gpt-4o-mini",
                        "temperature": 0.1
                    })
                },
                "embedder": {
                    "provider": self.config.get("embedder_provider", "openai"), 
                    "config": self.config.get("embedder_config", {
                        "model": "text-embedding-ada-002"
                    })
                }
            }
            
            # Try to create AsyncMemory synchronously for now - in production this would be async
            try:
                # Note: This is a simplified approach. In production, proper async initialization would be needed
                self.memory_client = None  # Disable mem0 for now due to async initialization complexity
                self.logger.info("mem0 disabled - using fallback storage (async initialization needed)")
            except Exception as e:
                self.logger.error(f"Failed to initialize mem0: {e}")
                self.memory_client = None
                
        except Exception as e:
            self.logger.error(f"Failed to configure mem0: {e}")
            self.memory_client = None

    async def add_user_message(self, user_id: str, message: str, metadata: Optional[Dict] = None) -> str:
        """
        Add a user message to memory.
        
        Args:
            user_id: User identifier
            message: User message content
            metadata: Additional metadata
            
        Returns:
            Memory ID
        """
        if self.memory_client:
            try:
                result = await self.memory_client.add(
                    messages=[{"role": "user", "content": message}],
                    user_id=user_id,
                    metadata=metadata or {}
                )
                return result.get("id", "unknown")
            except Exception as e:
                self.logger.error(f"Error adding user message to mem0: {e}")
                
        # Fallback storage
        memory_id = f"{user_id}_{datetime.now().isoformat()}"
        if user_id not in self._memory_store:
            self._memory_store[user_id] = []
        self._memory_store[user_id].append({
            "id": memory_id,
            "role": "user", 
            "content": message,
            "metadata": metadata or {},
            "timestamp": datetime.now().isoformat()
        })
        return memory_id

    async def add_assistant_message(self, user_id: str, message: str, metadata: Optional[Dict] = None) -> str:
        """
        Add an assistant message to memory.
        
        Args:
            user_id: User identifier
            message: Assistant message content
            metadata: Additional metadata
            
        Returns:
            Memory ID
        """
        if self.memory_client:
            try:
                result = await self.memory_client.add(
                    messages=[{"role": "assistant", "content": message}],
                    user_id=user_id,
                    metadata=metadata or {}
                )
                return result.get("id", "unknown")
            except Exception as e:
                self.logger.error(f"Error adding assistant message to mem0: {e}")
                
        # Fallback storage
        memory_id = f"{user_id}_{datetime.now().isoformat()}"
        if user_id not in self._memory_store:
            self._memory_store[user_id] = []
        self._memory_store[user_id].append({
            "id": memory_id,
            "role": "assistant",
            "content": message, 
            "metadata": metadata or {},
            "timestamp": datetime.now().isoformat()
        })
        return memory_id

    async def get_conversation_history(self, user_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get recent conversation history for a user.
        
        Args:
            user_id: User identifier
            limit: Maximum number of messages to return
            
        Returns:
            List of conversation messages
        """
        if self.memory_client:
            try:
                # Get memories and filter for conversation messages
                memories = await self.memory_client.get_all(user_id=user_id)
                conversation = []
                for memory in memories:
                    if memory.get("metadata", {}).get("type") == "conversation":
                        conversation.append(memory)
                return conversation[-limit:] if conversation else []
            except Exception as e:
                self.logger.error(f"Error getting conversation history from mem0: {e}")
                
        # Fallback storage
        if user_id in self._memory_store:
            messages = self._memory_store[user_id]
            return messages[-limit:] if messages else []
        return []

    async def get_relevant_memories(self, user_id: str, query: str, limit: int = 5) -> str:
        """
        Get relevant memories for a query using semantic search.
        
        Args:
            user_id: User identifier
            query: Query to find relevant memories for
            limit: Maximum number of memories to return
            
        Returns:
            Formatted string of relevant memories
        """
        if self.memory_client:
            try:
                memories = await self.memory_client.search(
                    query=query,
                    user_id=user_id,
                    limit=limit
                )
                
                if memories:
                    # Format memories into context string
                    context_parts = []
                    for memory in memories:
                        content = memory.get("memory", memory.get("content", ""))
                        if content:
                            context_parts.append(content)
                    return "\n".join(context_parts)
                return ""
            except Exception as e:
                self.logger.error(f"Error searching memories in mem0: {e}")
                
        # Fallback: simple text search
        if user_id in self._memory_store:
            relevant = []
            query_lower = query.lower()
            for memory in self._memory_store[user_id]:
                if query_lower in memory["content"].lower():
                    relevant.append(memory["content"])
                    if len(relevant) >= limit:
                        break
            return "\n".join(relevant)
        return ""

    async def get_memories(self, user_id: str, memory_type: str = "all", limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get memories of a specific type for a user.
        
        Args:
            user_id: User identifier
            memory_type: Type of memories to retrieve (all, conversation, facts, preferences)
            limit: Maximum number of memories to return
            
        Returns:
            List of memory objects
        """
        if self.memory_client:
            try:
                if memory_type == "all":
                    memories = await self.memory_client.get_all(user_id=user_id)
                else:
                    # Filter by memory type
                    all_memories = await self.memory_client.get_all(user_id=user_id)
                    memories = [
                        m for m in all_memories 
                        if m.get("metadata", {}).get("type") == memory_type
                    ]
                return memories[:limit] if memories else []
            except Exception as e:
                self.logger.error(f"Error getting memories from mem0: {e}")
                
        # Fallback storage
        if user_id in self._memory_store:
            memories = self._memory_store[user_id]
            if memory_type != "all":
                memories = [
                    m for m in memories
                    if m.get("metadata", {}).get("type") == memory_type
                ]
            return memories[:limit] if memories else []
        return []

    async def search_memories(self, user_id: str, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Search through memories using semantic search.
        
        Args:
            user_id: User identifier
            query: Search query
            limit: Maximum number of results
            
        Returns:
            List of matching memory objects
        """
        if self.memory_client:
            try:
                memories = await self.memory_client.search(
                    query=query,
                    user_id=user_id,
                    limit=limit
                )
                return memories if memories else []
            except Exception as e:
                self.logger.error(f"Error searching memories in mem0: {e}")
                
        # Fallback: simple text search
        if user_id in self._memory_store:
            results = []
            query_lower = query.lower()
            for memory in self._memory_store[user_id]:
                if query_lower in memory["content"].lower():
                    results.append(memory)
                    if len(results) >= limit:
                        break
            return results
        return []

    async def update_memory(self, memory_id: str, content: str, metadata: Optional[Dict] = None) -> bool:
        """
        Update an existing memory.
        
        Args:
            memory_id: Memory identifier
            content: New content
            metadata: Updated metadata
            
        Returns:
            Success status
        """
        if self.memory_client:
            try:
                await self.memory_client.update(
                    memory_id=memory_id,
                    data=content
                )
                return True
            except Exception as e:
                self.logger.error(f"Error updating memory in mem0: {e}")
                return False
                
        # Fallback: update in local storage
        for user_memories in self._memory_store.values():
            for memory in user_memories:
                if memory["id"] == memory_id:
                    memory["content"] = content
                    if metadata:
                        memory["metadata"].update(metadata)
                    return True
        return False

    async def delete_memory(self, memory_id: str) -> bool:
        """
        Delete a memory by ID.
        
        Args:
            memory_id: Memory identifier
            
        Returns:
            Success status
        """
        if self.memory_client:
            try:
                await self.memory_client.delete(memory_id=memory_id)
                return True
            except Exception as e:
                self.logger.error(f"Error deleting memory from mem0: {e}")
                return False
                
        # Fallback: delete from local storage
        for user_memories in self._memory_store.values():
            for i, memory in enumerate(user_memories):
                if memory["id"] == memory_id:
                    del user_memories[i]
                    return True
        return False

    async def get_user_stats(self, user_id: str) -> Dict[str, Any]:
        """
        Get statistics about a user's memories.
        
        Args:
            user_id: User identifier
            
        Returns:
            Statistics dictionary
        """
        memories = await self.get_memories(user_id, "all", limit=1000)  # Get all memories
        
        total_memories = len(memories)
        memory_types = {}
        
        for memory in memories:
            mem_type = memory.get("metadata", {}).get("type", "unknown")
            memory_types[mem_type] = memory_types.get(mem_type, 0) + 1
            
        return {
            "user_id": user_id,
            "total_memories": total_memories,
            "memory_types": memory_types,
            "has_mem0": self.memory_client is not None
        }