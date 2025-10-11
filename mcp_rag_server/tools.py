"""
MCP tool definitions for Verba RAG operations.

Defines the available tools that can be called through the Model Context Protocol,
covering chunking, indexing, retrieval, generation, and memory management.
"""

from mcp.types import Tool

# Define all available RAG tools
rag_tools = [
    Tool(
        name="chunk_documents",
        description="Chunk documents using Verba's chunking strategies for optimal retrieval",
        inputSchema={
            "type": "object",
            "properties": {
                "documents": {
                    "type": "array",
                    "description": "List of documents to chunk",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string", "description": "Document name"},
                            "content": {"type": "string", "description": "Document content"},
                            "type": {"type": "string", "description": "Document type (text, pdf, etc.)"},
                            "metadata": {"type": "object", "description": "Additional metadata"}
                        },
                        "required": ["name", "content"]
                    }
                },
                "chunker": {
                    "type": "string", 
                    "description": "Chunking strategy to use",
                    "enum": ["TokenChunker", "SentenceChunker", "RecursiveChunker", "SemanticChunker"],
                    "default": "TokenChunker"
                },
                "config": {
                    "type": "object",
                    "description": "Chunker-specific configuration",
                    "properties": {
                        "chunk_size": {"type": "integer", "description": "Target chunk size"},
                        "chunk_overlap": {"type": "integer", "description": "Overlap between chunks"},
                        "separators": {"type": "array", "items": {"type": "string"}, "description": "Custom separators"}
                    }
                }
            },
            "required": ["documents"]
        }
    ),
    
    Tool(
        name="index_documents", 
        description="Index documents using Verba's embedding and vector storage system",
        inputSchema={
            "type": "object",
            "properties": {
                "documents": {
                    "type": "array",
                    "description": "List of documents to index",
                    "items": {
                        "type": "object", 
                        "properties": {
                            "name": {"type": "string", "description": "Document name"},
                            "content": {"type": "string", "description": "Document content"},
                            "type": {"type": "string", "description": "Document type"},
                            "metadata": {"type": "object", "description": "Additional metadata"}
                        },
                        "required": ["name", "content"]
                    }
                },
                "embedder": {
                    "type": "string",
                    "description": "Embedding model to use", 
                    "enum": ["OpenAIEmbedder", "CohereEmbedder", "HuggingFaceEmbedder"],
                    "default": "OpenAIEmbedder"
                },
                "config": {
                    "type": "object",
                    "description": "Embedder configuration",
                    "properties": {
                        "model": {"type": "string", "description": "Specific model name"},
                        "batch_size": {"type": "integer", "description": "Batch size for embedding"}
                    }
                }
            },
            "required": ["documents"]
        }
    ),
    
    Tool(
        name="retrieve_context",
        description="Retrieve relevant context for a query using Verba's retrieval system with memory enhancement",
        inputSchema={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Query to search for relevant context"
                },
                "retriever": {
                    "type": "string",
                    "description": "Retrieval strategy to use",
                    "enum": ["BasicRetriever", "AdvancedRetriever", "HybridRetriever"],
                    "default": "BasicRetriever"
                },
                "config": {
                    "type": "object",
                    "description": "Retrieval configuration",
                    "properties": {
                        "max_results": {"type": "integer", "description": "Maximum number of results to return"},
                        "similarity_threshold": {"type": "number", "description": "Minimum similarity score"},
                        "filters": {"type": "object", "description": "Metadata filters"}
                    }
                },
                "user_id": {
                    "type": "string",
                    "description": "User ID for memory context",
                    "default": "default"
                }
            },
            "required": ["query"]
        }
    ),
    
    Tool(
        name="generate_response",
        description="Generate response using Verba's generation system with conversation memory",
        inputSchema={
            "type": "object", 
            "properties": {
                "query": {
                    "type": "string",
                    "description": "User query to respond to"
                },
                "context": {
                    "type": "string",
                    "description": "Retrieved context for the query"
                },
                "generator": {
                    "type": "string",
                    "description": "Generation model to use",
                    "enum": ["OpenAIGenerator", "AnthropicGenerator", "CohereGenerator"],
                    "default": "OpenAIGenerator"
                },
                "config": {
                    "type": "object",
                    "description": "Generation configuration",
                    "properties": {
                        "temperature": {"type": "number", "description": "Response creativity (0.0-1.0)"},
                        "max_tokens": {"type": "integer", "description": "Maximum response length"},
                        "system_message": {"type": "string", "description": "System instruction"}
                    }
                },
                "user_id": {
                    "type": "string", 
                    "description": "User ID for conversation memory",
                    "default": "default"
                }
            },
            "required": ["query"]
        }
    ),
    
    Tool(
        name="multimodal_query",
        description="Process a multi-modal query combining text, images, audio, and other media types",
        inputSchema={
            "type": "object",
            "properties": {
                "text_query": {
                    "type": "string",
                    "description": "Text component of the query"
                },
                "media": {
                    "type": "array",
                    "description": "Media items to process",
                    "items": {
                        "type": "object",
                        "properties": {
                            "type": {
                                "type": "string",
                                "enum": ["image", "audio", "video", "document"],
                                "description": "Type of media"
                            },
                            "content": {
                                "type": "string", 
                                "description": "Media content (base64 encoded or URL)"
                            },
                            "metadata": {
                                "type": "object",
                                "description": "Additional media metadata"
                            }
                        },
                        "required": ["type", "content"]
                    }
                },
                "user_id": {
                    "type": "string",
                    "description": "User ID for memory context",
                    "default": "default"
                }
            },
            "required": ["text_query"]
        }
    ),
    
    Tool(
        name="get_memory_context",
        description="Retrieve memory context for a user from mem0 storage",
        inputSchema={
            "type": "object",
            "properties": {
                "user_id": {
                    "type": "string",
                    "description": "User ID to get memory for",
                    "default": "default"
                },
                "memory_type": {
                    "type": "string",
                    "enum": ["all", "conversation", "facts", "preferences"],
                    "description": "Type of memories to retrieve",
                    "default": "all"
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of memories to return",
                    "default": 10
                }
            }
        }
    ),
    
    Tool(
        name="search_memories",
        description="Search through user memories using semantic search",
        inputSchema={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query for finding relevant memories"
                },
                "user_id": {
                    "type": "string", 
                    "description": "User ID to search memories for",
                    "default": "default"
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of results to return",
                    "default": 5
                }
            },
            "required": ["query"]
        }
    )
]