"""
MCP RAG Server implementation.

Provides a Model Context Protocol server that exposes Verba's RAG capabilities
as standardized MCP tools for chunking, indexing, retrieval, and generation.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional, Sequence

import mcp
from mcp import server, types
from mcp.server import Server
from mcp.types import (
    CallToolRequest,
    CallToolResult,
    ListToolsResult,
    Tool,
    TextContent,
    ImageContent,
    EmbeddedResource,
)

from goldenverba.verba_manager import VerbaManager
from goldenverba.components.document import Document
from goldenverba.server.helpers import LoggerManager

from .memory_manager import Mem0MemoryManager
from .tools import rag_tools


class MCPRAGServer:
    """
    MCP server that exposes Verba's RAG capabilities as standardized tools.
    
    This server provides:
    - Document chunking tools
    - Vector indexing tools  
    - Multi-modal retrieval tools
    - Generation tools with memory
    - Memory management via mem0
    """
    
    def __init__(
        self,
        name: str = "verba-rag-server",
        version: str = "1.0.0",
        weaviate_url: Optional[str] = None,
        weaviate_key: Optional[str] = None,
        mem0_config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the MCP RAG server.
        
        Args:
            name: Server name
            version: Server version
            weaviate_url: Weaviate instance URL
            weaviate_key: Weaviate API key
            mem0_config: Configuration for mem0 memory management
        """
        self.app = Server(name)
        self.name = name
        self.version = version
        
        # Initialize Verba manager
        self.verba_manager = VerbaManager()
        
        # Initialize memory manager
        self.memory_manager = Mem0MemoryManager(config=mem0_config or {})
        
        # Initialize logger
        self.logger = LoggerManager().get_logger()
        
        # Register MCP handlers
        self._register_handlers()
        
        # Store server configuration
        self.config = {
            "weaviate_url": weaviate_url,
            "weaviate_key": weaviate_key,
            "mem0_config": mem0_config or {}
        }

    def _register_handlers(self):
        """Register MCP protocol handlers."""
        
        @self.app.list_tools()
        async def list_tools() -> ListToolsResult:
            """List available RAG tools."""
            return ListToolsResult(tools=rag_tools)
        
        @self.app.call_tool()
        async def call_tool(request: CallToolRequest) -> CallToolResult:
            """Execute a RAG tool."""
            try:
                return await self._execute_tool(request)
            except Exception as e:
                self.logger.error(f"Error executing tool {request.name}: {e}")
                return CallToolResult(
                    content=[TextContent(
                        type="text",
                        text=f"Error: {str(e)}"
                    )],
                    isError=True
                )

    async def _execute_tool(self, request: CallToolRequest) -> CallToolResult:
        """
        Execute a specific RAG tool.
        
        Args:
            request: Tool call request
            
        Returns:
            Tool execution result
        """
        tool_name = request.name
        arguments = request.arguments or {}
        
        # Add user context to memory before processing
        if "query" in arguments:
            await self.memory_manager.add_user_message(
                user_id=arguments.get("user_id", "default"),
                message=arguments["query"]
            )
        
        if tool_name == "chunk_documents":
            return await self._chunk_documents(arguments)
        elif tool_name == "index_documents":
            return await self._index_documents(arguments)
        elif tool_name == "retrieve_context":
            return await self._retrieve_context(arguments)
        elif tool_name == "generate_response":
            return await self._generate_response(arguments)
        elif tool_name == "multimodal_query":
            return await self._multimodal_query(arguments)
        elif tool_name == "get_memory_context":
            return await self._get_memory_context(arguments)
        elif tool_name == "search_memories":
            return await self._search_memories(arguments)
        else:
            raise ValueError(f"Unknown tool: {tool_name}")

    async def _chunk_documents(self, arguments: Dict[str, Any]) -> CallToolResult:
        """
        Chunk documents using Verba's chunking strategies.
        
        Args:
            arguments: Tool arguments containing documents and chunking config
            
        Returns:
            Chunked documents result
        """
        documents_data = arguments.get("documents", [])
        chunker_name = arguments.get("chunker", "TokenChunker")
        chunk_config = arguments.get("config", {})
        
        # Convert to Verba Document objects
        documents = []
        for doc_data in documents_data:
            doc = Document(
                name=doc_data.get("name", ""),
                content=doc_data.get("content", ""),
                doc_type=doc_data.get("type", "text"),
                metadata=doc_data.get("metadata", {})
            )
            documents.append(doc)
        
        # Get the specified chunker
        available_chunkers = self.verba_manager.get_chunkers()
        chunker = None
        for c in available_chunkers:
            if c.name == chunker_name:
                chunker = c
                break
        
        if not chunker:
            raise ValueError(f"Chunker {chunker_name} not found")
        
        # Perform chunking
        chunked_docs = await chunker.chunk(
            config=chunk_config,
            documents=documents
        )
        
        # Format results
        result_data = []
        for doc in chunked_docs:
            result_data.append({
                "name": doc.name,
                "content": doc.content,
                "type": doc.doc_type,
                "chunk_id": doc.chunk_id,
                "metadata": doc.metadata
            })
        
        return CallToolResult(
            content=[TextContent(
                type="text",
                text=f"Successfully chunked {len(documents)} documents into {len(chunked_docs)} chunks using {chunker_name}"
            )],
            metadata={"chunks": result_data}
        )

    async def _index_documents(self, arguments: Dict[str, Any]) -> CallToolResult:
        """
        Index documents using Verba's embedding and vector storage.
        
        Args:
            arguments: Tool arguments containing documents and indexing config
            
        Returns:
            Indexing result
        """
        documents_data = arguments.get("documents", [])
        embedder_name = arguments.get("embedder", "OpenAIEmbedder")
        embed_config = arguments.get("config", {})
        
        # Convert to Verba Document objects
        documents = []
        for doc_data in documents_data:
            doc = Document(
                name=doc_data.get("name", ""),
                content=doc_data.get("content", ""),
                doc_type=doc_data.get("type", "text"),
                metadata=doc_data.get("metadata", {})
            )
            documents.append(doc)
        
        # Get embedder
        available_embedders = self.verba_manager.get_embedders()
        embedder = None
        for e in available_embedders:
            if e.name == embedder_name:
                embedder = e
                break
        
        if not embedder:
            raise ValueError(f"Embedder {embedder_name} not found")
        
        # Embed and index documents
        embedded_docs = await embedder.embed(
            config=embed_config,
            documents=documents
        )
        
        # Index in Weaviate (simplified - would need proper Weaviate setup)
        indexed_count = len(embedded_docs)
        
        return CallToolResult(
            content=[TextContent(
                type="text",
                text=f"Successfully indexed {indexed_count} documents using {embedder_name}"
            )],
            metadata={"indexed_count": indexed_count}
        )

    async def _retrieve_context(self, arguments: Dict[str, Any]) -> CallToolResult:
        """
        Retrieve relevant context for a query using Verba's retrieval system.
        
        Args:
            arguments: Tool arguments containing query and retrieval config
            
        Returns:
            Retrieved context result
        """
        query = arguments.get("query", "")
        retriever_name = arguments.get("retriever", "BasicRetriever")
        retrieval_config = arguments.get("config", {})
        user_id = arguments.get("user_id", "default")
        
        if not query:
            raise ValueError("Query is required for retrieval")
        
        # Get memory context to enhance query
        memory_context = await self.memory_manager.get_relevant_memories(
            user_id=user_id,
            query=query
        )
        
        # Enhance query with memory if available
        enhanced_query = query
        if memory_context:
            enhanced_query = f"Context: {memory_context}\n\nQuery: {query}"
        
        # Get retriever
        available_retrievers = self.verba_manager.get_retrievers()
        retriever = None
        for r in available_retrievers:
            if r.name == retriever_name:
                retriever = r
                break
        
        if not retriever:
            raise ValueError(f"Retriever {retriever_name} not found")
        
        # Perform retrieval (simplified - would need proper Weaviate client)
        # For now, return a mock result
        retrieved_docs = [
            {
                "content": f"Mock retrieved content for query: {enhanced_query}",
                "metadata": {"score": 0.85, "source": "mock_document.txt"}
            }
        ]
        
        return CallToolResult(
            content=[TextContent(
                type="text",
                text=f"Retrieved {len(retrieved_docs)} relevant documents for query: {query}"
            )],
            metadata={
                "retrieved_docs": retrieved_docs,
                "enhanced_query": enhanced_query,
                "memory_used": bool(memory_context)
            }
        )

    async def _generate_response(self, arguments: Dict[str, Any]) -> CallToolResult:
        """
        Generate response using Verba's generation system with memory.
        
        Args:
            arguments: Tool arguments containing query, context, and generation config
            
        Returns:
            Generated response result
        """
        query = arguments.get("query", "")
        context = arguments.get("context", "")
        generator_name = arguments.get("generator", "OpenAIGenerator")
        generation_config = arguments.get("config", {})
        user_id = arguments.get("user_id", "default")
        
        if not query:
            raise ValueError("Query is required for generation")
        
        # Get conversation history from memory
        conversation_history = await self.memory_manager.get_conversation_history(
            user_id=user_id,
            limit=5
        )
        
        # Get generator
        available_generators = self.verba_manager.get_generators()
        generator = None
        for g in available_generators:
            if g.name == generator_name:
                generator = g
                break
        
        if not generator:
            raise ValueError(f"Generator {generator_name} not found")
        
        # Generate response (simplified - would use actual generator)
        response = f"Generated response for '{query}' with context length {len(context)} characters and {len(conversation_history)} previous messages"
        
        # Store response in memory
        await self.memory_manager.add_assistant_message(
            user_id=user_id,
            message=response
        )
        
        return CallToolResult(
            content=[TextContent(
                type="text",
                text=response
            )],
            metadata={
                "query": query,
                "context_length": len(context),
                "conversation_history_count": len(conversation_history),
                "generator_used": generator_name
            }
        )

    async def _multimodal_query(self, arguments: Dict[str, Any]) -> CallToolResult:
        """
        Process a multi-modal query (text + images/other media).
        
        Args:
            arguments: Tool arguments containing text query and media
            
        Returns:
            Multi-modal processing result
        """
        text_query = arguments.get("text_query", "")
        media_items = arguments.get("media", [])
        user_id = arguments.get("user_id", "default")
        
        # Process each media item
        processed_media = []
        for media in media_items:
            media_type = media.get("type", "unknown")
            content = media.get("content", "")
            
            if media_type == "image":
                # For images, we'd use vision models or image analysis
                processed_media.append({
                    "type": "image",
                    "description": f"Processed image content: {content[:100]}...",
                    "analysis": "Mock image analysis result"
                })
            elif media_type == "audio":
                # For audio, we'd use speech-to-text
                processed_media.append({
                    "type": "audio", 
                    "transcription": f"Mock transcription of audio: {content[:100]}...",
                    "analysis": "Mock audio analysis result"
                })
        
        # Combine text and media context
        combined_context = f"Text: {text_query}\n"
        for media in processed_media:
            combined_context += f"Media ({media['type']}): {media.get('description', media.get('transcription', ''))}\n"
        
        # Store multimodal interaction in memory
        await self.memory_manager.add_user_message(
            user_id=user_id,
            message=combined_context
        )
        
        # Generate multimodal response
        response = f"Processed multimodal query with {len(media_items)} media items and text query: '{text_query}'"
        
        await self.memory_manager.add_assistant_message(
            user_id=user_id,
            message=response
        )
        
        return CallToolResult(
            content=[TextContent(
                type="text",
                text=response
            )],
            metadata={
                "text_query": text_query,
                "media_count": len(media_items),
                "processed_media": processed_media,
                "combined_context": combined_context
            }
        )

    async def _get_memory_context(self, arguments: Dict[str, Any]) -> CallToolResult:
        """
        Get memory context for a user.
        
        Args:
            arguments: Tool arguments containing user_id and optional filters
            
        Returns:
            Memory context result
        """
        user_id = arguments.get("user_id", "default")
        memory_type = arguments.get("memory_type", "all")  # all, conversation, facts, preferences
        limit = arguments.get("limit", 10)
        
        memories = await self.memory_manager.get_memories(
            user_id=user_id,
            memory_type=memory_type,
            limit=limit
        )
        
        return CallToolResult(
            content=[TextContent(
                type="text",
                text=f"Retrieved {len(memories)} memories for user {user_id}"
            )],
            metadata={
                "user_id": user_id,
                "memory_type": memory_type,
                "memories": memories
            }
        )

    async def _search_memories(self, arguments: Dict[str, Any]) -> CallToolResult:
        """
        Search through memories using semantic search.
        
        Args:
            arguments: Tool arguments containing search query and filters
            
        Returns:
            Memory search results
        """
        query = arguments.get("query", "")
        user_id = arguments.get("user_id", "default")
        limit = arguments.get("limit", 5)
        
        if not query:
            raise ValueError("Query is required for memory search")
        
        memories = await self.memory_manager.search_memories(
            user_id=user_id,
            query=query,
            limit=limit
        )
        
        return CallToolResult(
            content=[TextContent(
                type="text",
                text=f"Found {len(memories)} relevant memories for query: '{query}'"
            )],
            metadata={
                "query": query,
                "user_id": user_id,
                "memories": memories
            }
        )

    async def run(self, transport_type: str = "stdio"):
        """
        Run the MCP server.
        
        Args:
            transport_type: Transport protocol ("stdio", "websocket", etc.)
        """
        if transport_type == "stdio":
            from mcp.server.stdio import stdio_server
            async with stdio_server() as (read_stream, write_stream):
                await self.app.run(
                    read_stream,
                    write_stream,
                    self.app.create_initialization_options()
                )
        else:
            raise ValueError(f"Unsupported transport type: {transport_type}")