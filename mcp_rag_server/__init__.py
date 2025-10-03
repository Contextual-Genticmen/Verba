"""
MCP RAG Server - Model Context Protocol server for Verba RAG operations.

This module provides a Model Context Protocol server that exposes Verba's RAG capabilities
as MCP tools, including chunking, indexing, retrieval, and generation with integrated
mem0 memory management for multi-modal queries.
"""

__version__ = "1.0.0"

# Import components conditionally to avoid dependency issues
try:
    from .server import MCPRAGServer
    from .tools import rag_tools
    _server_available = True
except ImportError:
    MCPRAGServer = None
    rag_tools = []
    _server_available = False

from .memory_manager import Mem0MemoryManager

if _server_available:
    __all__ = ["MCPRAGServer", "rag_tools", "Mem0MemoryManager"]
else:
    __all__ = ["Mem0MemoryManager"]