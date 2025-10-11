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
    from mem0 import Memory
    HAS_MEM0 = True
except ImportError:
    # Fallback if mem0 is not available
    Memory = None
    HAS_MEM0 = False


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
        
        if not HAS_MEM0:
            self.logger.warning("mem0 not available, using mock memory manager")
            self.memory_client = None
            return
            
        try:
            # Configure mem0 based on config with proper structure
            mem0_config = {
                "vector_store": {
                    "provider": self.config.get("vector_store", {}).get("provider", "milvus"),
                    "config": self.config.get("vector_store", {}).get("config", {
                        "collection_name": "verba_memories",
                        "embedding_model_dims": "768",
                        "url": "./milvus.db"
                    })
                },
                "llm": {
                    "provider": self.config.get("llm", {}).get("provider", "ollama"),
                    "config": self.config.get("llm", {}).get("config", {
                        "model": "qwen2.5:0.5b",
                        "temperature": 0.1,
                        "base_url": "http://localhost:11434"
                    })
                },
                "embedder": {
                    "provider": self.config.get("embedder", {}).get("provider", "ollama"), 
                    "config": self.config.get("embedder", {}).get("config", {
                        "model": "nomic-embed-text:latest",
                        "base_url": "http://localhost:11434"
                    })
                },
                "version": self.config.get("version", "v1.1")
            }
            
            # Create Memory client (synchronous)
            try:
                self.memory_client = Memory.from_config(mem0_config)
                self.logger.info("mem0 Memory client initialized successfully with Milvus")
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
                # Use synchronous mem0 API in async context
                result = await asyncio.to_thread(
                    self.memory_client.add,
                    messages=message,
                    user_id=user_id,
                    metadata=metadata or {}
                )
                return result.get("id", "unknown") if isinstance(result, dict) else str(result)
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
                result = await asyncio.to_thread(
                    self.memory_client.add,
                    messages=message,
                    user_id=user_id,
                    metadata=metadata or {}
                )
                return result.get("id", "unknown") if isinstance(result, dict) else str(result)
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
                # Get all memories for user
                memories = await asyncio.to_thread(
                    self.memory_client.get_all,
                    user_id=user_id
                )
                # Convert to list if needed
                if isinstance(memories, dict) and "results" in memories:
                    memories = memories["results"]
                elif not isinstance(memories, list):
                    memories = []
                return memories[-limit:] if memories else []
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
                memories = await asyncio.to_thread(
                    self.memory_client.search,
                    query=query,
                    user_id=user_id,
                    limit=limit
                )
                
                # Handle different response formats
                if isinstance(memories, dict) and "results" in memories:
                    memories = memories["results"]
                elif not isinstance(memories, list):
                    memories = []
                
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
                all_memories = await asyncio.to_thread(
                    self.memory_client.get_all,
                    user_id=user_id
                )
                # Handle different response formats
                if isinstance(all_memories, dict) and "results" in all_memories:
                    all_memories = all_memories["results"]
                elif not isinstance(all_memories, list):
                    all_memories = []
                    
                if memory_type != "all":
                    # Filter by memory type
                    memories = [
                        m for m in all_memories 
                        if m.get("metadata", {}).get("type") == memory_type
                    ]
                else:
                    memories = all_memories
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
                memories = await asyncio.to_thread(
                    self.memory_client.search,
                    query=query,
                    user_id=user_id,
                    limit=limit
                )
                # Handle different response formats
                if isinstance(memories, dict) and "results" in memories:
                    memories = memories["results"]
                elif not isinstance(memories, list):
                    memories = []
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
                await asyncio.to_thread(
                    self.memory_client.update,
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
                await asyncio.to_thread(
                    self.memory_client.delete,
                    memory_id=memory_id
                )
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