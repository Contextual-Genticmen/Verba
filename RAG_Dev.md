# RAG Development Guide for Verba

## Table of Contents
1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Chunking and Indexing Customization](#chunking-and-indexing-customization)
4. [Retrieval System Customization](#retrieval-system-customization)
5. [Continuous Chat Memory and Related Components](#continuous-chat-memory-and-related-components)
6. [Vector Database Integration](#vector-database-integration)
7. [Component Development Guidelines](#component-development-guidelines)
8. [Testing and Best Practices](#testing-and-best-practices)
9. [Troubleshooting](#troubleshooting)

## Overview

Verba's RAG (Retrieval-Augmented Generation) system is built on a modular architecture that allows developers to customize and extend various components. This guide provides comprehensive instructions for contributing to and customizing the RAG system, focusing on three main areas:

1. **Chunking and Indexing**: How documents are split and prepared for vector storage
2. **Retrieval System**: How relevant chunks are found and retrieved
3. **Chat Memory**: How conversations are maintained and context is preserved

## Architecture

### Core Components

Verba's RAG system consists of five primary component types, each managed by dedicated managers:

```
VerbaManager
├── ReaderManager       # Document ingestion and parsing
├── ChunkerManager      # Text chunking strategies
├── EmbeddingManager    # Vector embedding generation
├── RetrieverManager    # Document retrieval logic
├── GeneratorManager    # Response generation
└── WeaviateManager     # Vector database operations
```

### Component Interface Pattern

All components inherit from the base `VerbaComponent` class and implement specific interfaces:

```python
class VerbaComponent:
    def __init__(self):
        self.name = ""
        self.requires_env = []      # Required environment variables
        self.requires_library = []   # Required Python libraries
        self.description = ""
        self.config = {}            # Component configuration schema
        self.type = ""
```

## Chunking and Indexing Customization

### Understanding the Chunking Pipeline

The chunking process transforms raw documents into smaller, semantically meaningful pieces that can be effectively embedded and retrieved.

#### Flow:
1. **Document Reading**: Raw documents are parsed by Reader components
2. **Chunking**: Documents are split into chunks by Chunker components
3. **Embedding**: Chunks are converted to vectors by Embedding components
4. **Indexing**: Vectors are stored in Weaviate with metadata

### Available Chunking Strategies

#### 1. Token-Based Chunking
```python
# Location: goldenverba/components/chunking/TokenChunker.py
class TokenChunker(Chunker):
    """Splits text based on token counts using spaCy"""
```

#### 2. Sentence-Based Chunking
```python
# Location: goldenverba/components/chunking/SentenceChunker.py
class SentenceChunker(Chunker):
    """Splits text based on sentence boundaries"""
```

#### 3. Recursive Chunking
```python
# Location: goldenverba/components/chunking/RecursiveChunker.py
class RecursiveChunker(Chunker):
    """Recursively splits text using predefined separators"""
```

#### 4. Semantic Chunking
```python
# Location: goldenverba/components/chunking/SemanticChunker.py
class SemanticChunker(Chunker):
    """Groups sentences by semantic similarity"""
```

#### 5. Code-Specific Chunking
```python
# Location: goldenverba/components/chunking/CodeChunker.py
class CodeChunker(Chunker):
    """Language-aware code splitting using LangChain"""
```

#### 6. Markdown Chunking
```python
# Location: goldenverba/components/chunking/MarkdownChunker.py
class MarkdownChunker(Chunker):
    """Markdown-aware text splitting preserving structure"""
```

### Creating a Custom Chunker

To create a custom chunking strategy:

#### Step 1: Implement the Chunker Interface

```python
# goldenverba/components/chunking/YourCustomChunker.py
from goldenverba.components.interfaces import Chunker, Embedding
from goldenverba.components.document import Document
from goldenverba.components.chunk import Chunk
from goldenverba.components.types import InputConfig

class YourCustomChunker(Chunker):
    """Your custom chunking implementation"""
    
    def __init__(self):
        super().__init__()
        self.name = "YourCustom"
        self.description = "Description of your chunking strategy"
        self.requires_library = ["any_required_libraries"]
        
        # Define configuration options
        self.config = {
            "Chunk Size": InputConfig(
                type="number",
                value=500,
                description="Maximum chunk size in characters",
                values=[],
            ),
            "Strategy": InputConfig(
                type="dropdown",
                value="semantic",
                description="Chunking strategy to use",
                values=["semantic", "syntactic", "hybrid"],
            ),
        }
    
    async def chunk(
        self,
        config: dict,
        documents: list[Document],
        embedder: Embedding | None = None,
        embedder_config: dict | None = None,
    ) -> list[Document]:
        """
        Split documents into chunks
        
        Args:
            config: Chunker configuration from self.config
            documents: List of Verba documents to chunk
            embedder: Optional embedder if chunking requires vectorization
            embedder_config: Optional embedder configuration
        
        Returns:
            List of documents with populated chunks
        """
        chunk_size = config["Chunk Size"].value
        strategy = config["Strategy"].value
        
        for document in documents:
            # Skip if document already contains chunks
            if len(document.chunks) > 0:
                continue
            
            # Implement your chunking logic here
            chunks_text = self.your_custom_chunking_logic(
                document.content, 
                chunk_size, 
                strategy
            )
            
            # Create Chunk objects
            for i, chunk_text in enumerate(chunks_text):
                chunk = Chunk(
                    content=chunk_text,
                    chunk_id=i,
                    start_i=None,  # Optional: character start index
                    end_i=None,    # Optional: character end index
                    content_without_overlap=chunk_text,
                )
                document.chunks.append(chunk)
        
        return documents
    
    def your_custom_chunking_logic(self, text: str, chunk_size: int, strategy: str) -> list[str]:
        """Implement your chunking algorithm here"""
        # Your custom implementation
        pass
```

#### Step 2: Register Your Chunker

Add your chunker to the imports in `goldenverba/components/managers.py`:

```python
# Import your chunker
from goldenverba.components.chunking.YourCustomChunker import YourCustomChunker

# Add to the chunkers list
chunkers = [
    TokenChunker(),
    SentenceChunker(),
    RecursiveChunker(),
    # ... other chunkers
    YourCustomChunker(),  # Add your chunker here
]
```

### Advanced Chunking Techniques

#### Semantic Chunking with Embeddings

For semantic chunking that requires embeddings:

```python
async def chunk(self, config: dict, documents: list[Document], 
                embedder: Embedding | None = None, 
                embedder_config: dict | None = None) -> list[Document]:
    
    if embedder is None:
        raise ValueError("Semantic chunking requires an embedder")
    
    for document in documents:
        sentences = self.split_into_sentences(document.content)
        
        # Generate embeddings for sentences
        embeddings = await embedder.vectorize(
            embedder_config, 
            [s["text"] for s in sentences]
        )
        
        # Group by semantic similarity
        chunks = self.group_by_similarity(sentences, embeddings)
        
        # Convert to Chunk objects
        for i, chunk_text in enumerate(chunks):
            document.chunks.append(Chunk(
                content=chunk_text,
                chunk_id=i,
                content_without_overlap=chunk_text,
            ))
```

#### Overlap Strategy Implementation

```python
def create_overlapping_chunks(self, text: str, chunk_size: int, overlap: int) -> list[str]:
    """Create chunks with specified overlap"""
    chunks = []
    start = 0
    
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end]
        chunks.append(chunk)
        
        if end >= len(text):
            break
            
        start = end - overlap
    
    return chunks
```

### Indexing Process

After chunking, documents go through the indexing pipeline:

1. **Embedding Generation**: Chunks are converted to vectors
2. **Metadata Extraction**: Document and chunk metadata is prepared
3. **Vector Storage**: Embeddings and metadata are stored in Weaviate

The indexing process is handled by the `EmbeddingManager`:

```python
# goldenverba/components/managers.py
class EmbeddingManager:
    async def vectorize(self, embedder: str, fileConfig: FileConfig, 
                       documents: list[Document], logger: LoggerManager):
        """Vectorize document chunks and prepare for storage"""
```

## Retrieval System Customization

### Understanding the Retrieval Pipeline

The retrieval system finds relevant chunks based on user queries through:

1. **Query Processing**: User query is embedded using the same embedder
2. **Vector Search**: Similar chunks are found using vector similarity
3. **Filtering**: Results are filtered by metadata, labels, or documents
4. **Ranking**: Results are scored and ranked by relevance
5. **Context Assembly**: Retrieved chunks are formatted for the generator

### Available Retrieval Strategies

#### Window Retriever
The primary retriever implementation provides advanced search capabilities:

```python
# goldenverba/components/retriever/WindowRetriever.py
class WindowRetriever(Retriever):
    """
    Retrieves chunks and surrounding context based on window size
    """
    
    # Configuration options:
    # - Search Mode: Hybrid Search (semantic + keyword)
    # - Limit Mode: Autocut (automatic) or Fixed (manual limit)
    # - Chunk Window: Number of surrounding chunks to include
    # - Threshold: Minimum score for window expansion
```

### Creating a Custom Retriever

#### Step 1: Implement the Retriever Interface

```python
# goldenverba/components/retriever/YourCustomRetriever.py
from goldenverba.components.interfaces import Retriever
from goldenverba.components.types import InputConfig
from goldenverba.components.managers import WeaviateManager

class YourCustomRetriever(Retriever):
    """Your custom retrieval implementation"""
    
    def __init__(self):
        super().__init__()
        self.name = "YourCustom"
        self.description = "Description of your retrieval strategy"
        
        self.config = {
            "Search Type": InputConfig(
                type="dropdown",
                value="semantic",
                description="Type of search to perform",
                values=["semantic", "keyword", "hybrid"],
            ),
            "Max Results": InputConfig(
                type="number",
                value=10,
                description="Maximum number of results to return",
                values=[],
            ),
            "Min Score": InputConfig(
                type="number", 
                value=0.7,
                description="Minimum similarity score (0-1)",
                values=[],
            ),
        }
    
    async def retrieve(
        self,
        client,
        query: str,
        vector: list[float],
        config: dict,
        weaviate_manager: WeaviateManager,
        labels: list[str] = [],
        document_uuids: list[str] = [],
    ):
        """
        Retrieve relevant chunks
        
        Args:
            client: Weaviate client
            query: User query string
            vector: Query embedding vector
            config: Retriever configuration
            weaviate_manager: Weaviate operations manager
            labels: Filter by document labels
            document_uuids: Filter by specific documents
        
        Returns:
            Tuple of (documents, context_string)
        """
        search_type = config["Search Type"].value
        max_results = config["Max Results"].value
        min_score = config["Min Score"].value
        
        # Implement your retrieval logic
        if search_type == "semantic":
            results = await self.semantic_search(
                client, vector, max_results, min_score, weaviate_manager
            )
        elif search_type == "keyword":
            results = await self.keyword_search(
                client, query, max_results, weaviate_manager
            )
        elif search_type == "hybrid":
            results = await self.hybrid_search(
                client, query, vector, max_results, min_score, weaviate_manager
            )
        
        # Apply filters
        filtered_results = self.apply_filters(results, labels, document_uuids)
        
        # Format results
        documents, context = self.format_results(filtered_results)
        
        return documents, context
    
    async def semantic_search(self, client, vector, max_results, min_score, weaviate_manager):
        """Implement semantic vector search"""
        # Your implementation here
        pass
    
    async def keyword_search(self, client, query, max_results, weaviate_manager):
        """Implement keyword-based search"""
        # Your implementation here
        pass
    
    async def hybrid_search(self, client, query, vector, max_results, min_score, weaviate_manager):
        """Implement hybrid search combining semantic and keyword"""
        # Your implementation here
        pass
```

#### Step 2: Advanced Retrieval Techniques

##### Reranking Implementation
```python
async def rerank_results(self, query: str, results: list, reranker_model: str = None):
    """Rerank results based on query relevance"""
    if not reranker_model:
        return results
    
    # Implement reranking logic
    # Example: using cross-encoder models for more accurate ranking
    scores = []
    for result in results:
        score = self.calculate_rerank_score(query, result.content, reranker_model)
        scores.append((result, score))
    
    # Sort by rerank scores
    reranked = sorted(scores, key=lambda x: x[1], reverse=True)
    return [result for result, score in reranked]
```

##### Multi-Vector Search
```python
async def multi_vector_search(self, client, vectors: list[list[float]], weights: list[float]):
    """Search using multiple vector representations"""
    all_results = []
    
    for vector, weight in zip(vectors, weights):
        results = await self.semantic_search(client, vector, max_results=50, min_score=0.5)
        # Weight and combine results
        weighted_results = [(r, r.score * weight) for r in results]
        all_results.extend(weighted_results)
    
    # Deduplicate and merge scores
    merged_results = self.merge_and_deduplicate(all_results)
    return merged_results
```

#### Step 3: Register Your Retriever

Add your retriever to the managers:

```python
# goldenverba/components/managers.py
from goldenverba.components.retriever.YourCustomRetriever import YourCustomRetriever

retrievers = [
    WindowRetriever(),
    YourCustomRetriever(),  # Add your retriever here
]
```

### Weaviate Integration

Verba uses Weaviate as the primary vector database. Key operations include:

#### Vector Search Operations
```python
# Example Weaviate search
collection = client.collections.get("Verba")
response = collection.query.near_vector(
    near_vector=query_vector,
    limit=limit,
    where=filters,
    return_metadata=MetadataQuery(score=True)
)
```

#### Metadata Filtering
```python
# Filter by document labels
label_filter = Filter.by_property("doc_labels").contains_any(labels)

# Filter by document UUIDs  
uuid_filter = Filter.by_property("doc_uuid").contains_any(document_uuids)

# Combine filters
combined_filter = label_filter & uuid_filter
```

## Continuous Chat Memory and Related Components

### Understanding Chat Memory

Verba maintains conversation context through several mechanisms:

1. **Conversation History**: Previous messages in the current session
2. **Context Window Management**: Truncating long conversations to fit model limits  
3. **Memory Integration**: Using conversation history to improve retrieval and generation

### Conversation Flow

```python
# goldenverba/verba_manager.py
async def generate_stream_answer(
    self,
    rag_config: dict,
    query: str, 
    context: str,
    conversation: list[dict],
):
    """Generate streaming response with conversation memory"""
    
    # conversation format:
    # [
    #   {"type": "user", "content": "Previous user message"},
    #   {"type": "assistant", "content": "Previous assistant response"},
    #   {"type": "user", "content": "Current user message"}
    # ]
```

### Generator Implementation

Generators handle conversation memory through the `prepare_messages` method:

```python
# Example from OpenAIGenerator
class OpenAIGenerator(Generator):
    def prepare_messages(
        self, 
        query: str, 
        context: str, 
        conversation: list[dict] = None
    ) -> list[dict]:
        """Format messages for the LLM including conversation history"""
        
        messages = []
        
        # System message with context
        system_content = f"{self.system_message}\n\nContext:\n{context}"
        messages.append({"role": "system", "content": system_content})
        
        # Add conversation history
        if conversation:
            for msg in conversation[:-1]:  # Exclude current query
                if msg["type"] == "user":
                    messages.append({"role": "user", "content": msg["content"]})
                elif msg["type"] == "assistant":
                    messages.append({"role": "assistant", "content": msg["content"]})
        
        # Add current query
        messages.append({"role": "user", "content": query})
        
        return messages
```

### Creating a Custom Generator with Advanced Memory

```python
# goldenverba/components/generation/YourCustomGenerator.py
from goldenverba.components.interfaces import Generator
from goldenverba.components.types import InputConfig

class YourCustomGenerator(Generator):
    """Custom generator with advanced memory management"""
    
    def __init__(self):
        super().__init__()
        self.name = "YourCustom"
        self.description = "Generator with advanced conversation memory"
        self.context_window = 8000  # Model context limit
        
        self.config = {
            "Memory Type": InputConfig(
                type="dropdown",
                value="sliding_window",
                description="Type of memory management",
                values=["sliding_window", "summarization", "entity_tracking"],
            ),
            "Memory Length": InputConfig(
                type="number",
                value=10,
                description="Number of previous exchanges to remember",
                values=[],
            ),
        }
    
    def prepare_messages(
        self, 
        query: str, 
        context: str, 
        conversation: list[dict] = None
    ) -> list[dict]:
        """Prepare messages with advanced memory management"""
        
        memory_type = self.config.get("Memory Type", {}).get("value", "sliding_window")
        memory_length = self.config.get("Memory Length", {}).get("value", 10)
        
        if memory_type == "sliding_window":
            return self.sliding_window_memory(query, context, conversation, memory_length)
        elif memory_type == "summarization":
            return self.summarization_memory(query, context, conversation)
        elif memory_type == "entity_tracking":
            return self.entity_tracking_memory(query, context, conversation)
    
    def sliding_window_memory(self, query: str, context: str, 
                             conversation: list[dict], window_size: int) -> list[dict]:
        """Keep only the last N conversation exchanges"""
        messages = []
        
        # System message
        system_content = f"{self.system_message}\n\nContext:\n{context}"
        messages.append({"role": "system", "content": system_content})
        
        if conversation:
            # Take only recent exchanges (user + assistant pairs)
            recent_conversation = conversation[-(window_size * 2):]
            
            for msg in recent_conversation[:-1]:  # Exclude current query
                role = "user" if msg["type"] == "user" else "assistant"
                messages.append({"role": role, "content": msg["content"]})
        
        # Current query
        messages.append({"role": "user", "content": query})
        return messages
    
    def summarization_memory(self, query: str, context: str, 
                           conversation: list[dict]) -> list[dict]:
        """Summarize old conversation and keep recent exchanges"""
        messages = []
        
        if not conversation or len(conversation) <= 6:
            return self.sliding_window_memory(query, context, conversation, 3)
        
        # Split conversation into old and recent parts
        old_conversation = conversation[:-6]  # Older exchanges to summarize
        recent_conversation = conversation[-6:]  # Recent exchanges to keep
        
        # Generate summary of old conversation
        summary = self.generate_conversation_summary(old_conversation)
        
        # System message with summary
        system_content = f"{self.system_message}\n\nContext:\n{context}\n\nConversation Summary:\n{summary}"
        messages.append({"role": "system", "content": system_content})
        
        # Add recent conversation
        for msg in recent_conversation[:-1]:
            role = "user" if msg["type"] == "user" else "assistant"
            messages.append({"role": role, "content": msg["content"]})
        
        # Current query
        messages.append({"role": "user", "content": query})
        return messages
    
    def entity_tracking_memory(self, query: str, context: str, 
                              conversation: list[dict]) -> list[dict]:
        """Track important entities and topics across conversation"""
        # Extract key entities and topics from conversation
        entities = self.extract_entities_from_conversation(conversation)
        topics = self.extract_topics_from_conversation(conversation)
        
        # Enhanced system message with entity context
        entity_context = f"Key entities discussed: {', '.join(entities)}\n"
        topic_context = f"Main topics: {', '.join(topics)}\n"
        
        system_content = f"{self.system_message}\n\nContext:\n{context}\n\n{entity_context}{topic_context}"
        
        messages = [{"role": "system", "content": system_content}]
        
        # Add recent conversation (last 3 exchanges)
        if conversation:
            recent = conversation[-6:]
            for msg in recent[:-1]:
                role = "user" if msg["type"] == "user" else "assistant"
                messages.append({"role": role, "content": msg["content"]})
        
        messages.append({"role": "user", "content": query})
        return messages
    
    def generate_conversation_summary(self, conversation: list[dict]) -> str:
        """Generate a concise summary of conversation history"""
        # Implement summarization logic (could use a separate LLM call)
        # For now, return a simple summary
        user_questions = [msg["content"] for msg in conversation if msg["type"] == "user"]
        return f"Previous discussion covered: {', '.join(user_questions[:3])}..."
    
    def extract_entities_from_conversation(self, conversation: list[dict]) -> list[str]:
        """Extract key entities mentioned in conversation"""
        # Implement named entity recognition
        # This could use spaCy, NLTK, or send to an NER service
        entities = set()
        for msg in conversation:
            # Simple implementation - extract capitalized words
            words = msg["content"].split()
            entities.update([w for w in words if w[0].isupper() and len(w) > 2])
        return list(entities)[:5]  # Return top 5 entities
    
    def extract_topics_from_conversation(self, conversation: list[dict]) -> list[str]:
        """Extract main topics from conversation"""
        # Implement topic extraction
        # Could use keyword extraction, topic modeling, or LLM analysis
        topics = ["general discussion"]  # Placeholder
        return topics
```

### Context Window Management

Managing context windows effectively is crucial for maintaining performance:

```python
def estimate_token_count(self, text: str) -> int:
    """Estimate token count for context window management"""
    # Simple estimation: ~4 characters per token
    return len(text) // 4

def truncate_context_to_fit(
    self, 
    system_msg: str, 
    conversation: list[dict], 
    context: str, 
    query: str,
    max_tokens: int
) -> tuple[list[dict], str]:
    """Truncate conversation and context to fit within token limit"""
    
    # Reserve tokens for system message and query
    system_tokens = self.estimate_token_count(system_msg)
    query_tokens = self.estimate_token_count(query)
    reserved_tokens = system_tokens + query_tokens + 100  # Buffer
    
    available_tokens = max_tokens - reserved_tokens
    
    # Truncate context if too long
    context_tokens = self.estimate_token_count(context)
    if context_tokens > available_tokens // 2:
        # Keep only the most relevant parts of context
        context = self.truncate_context(context, available_tokens // 2)
        context_tokens = self.estimate_token_count(context)
    
    # Truncate conversation to fit remaining tokens
    remaining_tokens = available_tokens - context_tokens
    truncated_conversation = self.truncate_conversation(conversation, remaining_tokens)
    
    return truncated_conversation, context
```

### Memory Persistence

For persistent memory across sessions, you can implement:

```python
class PersistentMemoryManager:
    """Manage persistent conversation memory"""
    
    def __init__(self, weaviate_client):
        self.client = weaviate_client
    
    async def save_conversation(self, session_id: str, conversation: list[dict]):
        """Save conversation to persistent storage"""
        # Implementation depends on your storage choice
        pass
    
    async def load_conversation(self, session_id: str) -> list[dict]:
        """Load conversation from persistent storage"""
        # Implementation depends on your storage choice
        pass
    
    async def save_entity_memory(self, session_id: str, entities: dict):
        """Save extracted entities and topics"""
        pass
    
    async def load_entity_memory(self, session_id: str) -> dict:
        """Load saved entities and topics"""
        pass
```

## Vector Database Integration

### Weaviate as the Primary Vector Database

Verba is designed around Weaviate, but the architecture allows for other vector databases:

#### Weaviate Manager
```python
# goldenverba/components/managers.py
class WeaviateManager:
    """Handles all Weaviate operations"""
    
    async def create_verba_collection(self, client):
        """Create the main Verba collection schema"""
    
    async def add_document(self, client, document: Document):
        """Add a document with its chunks to Weaviate"""
    
    async def delete_document(self, client, document_uuid: str):
        """Delete a document and all its chunks"""
    
    async def search_chunks(self, client, vector: list[float], limit: int):
        """Search for similar chunks"""
```

### Supporting Multiple Vector Databases

To add support for additional vector databases:

#### Step 1: Create a Database Manager Interface

```python
# goldenverba/components/interfaces.py
class VectorDatabase(VerbaComponent):
    """Interface for vector database implementations"""
    
    async def create_collection(self, collection_name: str, schema: dict):
        """Create a collection with the given schema"""
        raise NotImplementedError
    
    async def add_vectors(self, collection_name: str, vectors: list, metadata: list):
        """Add vectors with metadata to collection"""
        raise NotImplementedError
    
    async def search_vectors(self, collection_name: str, query_vector: list, 
                           limit: int, filters: dict = None):
        """Search for similar vectors"""
        raise NotImplementedError
    
    async def delete_by_filter(self, collection_name: str, filters: dict):
        """Delete vectors matching filters"""
        raise NotImplementedError
```

#### Step 2: Implement Alternative Database Managers

```python
# goldenverba/components/vector_db/PineconeManager.py
class PineconeManager(VectorDatabase):
    """Pinecone vector database implementation"""
    
    def __init__(self):
        super().__init__()
        self.name = "Pinecone"
        self.requires_env = ["PINECONE_API_KEY", "PINECONE_ENVIRONMENT"]
        self.requires_library = ["pinecone-client"]
    
    async def create_collection(self, collection_name: str, schema: dict):
        # Implement Pinecone index creation
        pass
    
    async def add_vectors(self, collection_name: str, vectors: list, metadata: list):
        # Implement Pinecone vector insertion
        pass
    
    async def search_vectors(self, collection_name: str, query_vector: list,
                           limit: int, filters: dict = None):
        # Implement Pinecone search
        pass

# goldenverba/components/vector_db/ChromaManager.py  
class ChromaManager(VectorDatabase):
    """ChromaDB vector database implementation"""
    
    def __init__(self):
        super().__init__()
        self.name = "ChromaDB"
        self.requires_library = ["chromadb"]
    
    # Implement ChromaDB methods...
```

#### Step 3: Abstract Vector Operations

```python
# goldenverba/components/managers.py
class VectorDatabaseManager:
    """Manages different vector database implementations"""
    
    def __init__(self):
        self.databases = {
            "weaviate": WeaviateManager(),
            "pinecone": PineconeManager(),
            "chroma": ChromaManager(),
        }
        self.active_db = "weaviate"  # Default
    
    def get_database(self, db_name: str = None):
        """Get the specified database manager"""
        if db_name is None:
            db_name = self.active_db
        return self.databases.get(db_name)
    
    async def create_collection(self, collection_name: str, schema: dict, db_name: str = None):
        """Create collection in the specified database"""
        db = self.get_database(db_name)
        return await db.create_collection(collection_name, schema)
    
    # Implement other abstracted methods...
```

## Component Development Guidelines

### Design Principles

1. **Modularity**: Each component should be self-contained and interchangeable
2. **Configuration**: All parameters should be configurable through the `config` attribute
3. **Error Handling**: Components should handle errors gracefully and provide meaningful messages
4. **Async Support**: All I/O operations should be asynchronous
5. **Type Hints**: Use proper type annotations for better code maintainability

### Configuration Schema

Use the `InputConfig` class to define configurable parameters:

```python
from goldenverba.components.types import InputConfig

# Different input types
self.config = {
    "Text Input": InputConfig(
        type="text",
        value="default_value",
        description="Description for users",
        values=[],  # Not used for text inputs
    ),
    "Number Input": InputConfig(
        type="number", 
        value=100,
        description="Numeric parameter",
        values=[],
    ),
    "Dropdown": InputConfig(
        type="dropdown",
        value="option1",
        description="Select an option",
        values=["option1", "option2", "option3"],
    ),
    "Multi-Select": InputConfig(
        type="multi",
        value="",
        description="Multi-select parameter", 
        values=["item1", "item2", "item3"],
    ),
    "Password": InputConfig(
        type="password",
        value="",
        description="Sensitive information",
        values=[],
    ),
    "Textarea": InputConfig(
        type="textarea",
        value="Default text content",
        description="Large text input",
        values=[],
    ),
}
```

### Error Handling Patterns

```python
from wasabi import msg

class YourComponent(VerbaComponent):
    async def your_method(self, *args, **kwargs):
        try:
            # Your implementation
            result = await self.do_something()
            return result
            
        except ImportError as e:
            msg.fail(f"Required library not installed: {e}")
            raise e
        except ValueError as e:
            msg.warn(f"Invalid configuration: {e}")
            raise e
        except Exception as e:
            msg.fail(f"Unexpected error in {self.name}: {e}")
            raise e
```

### Logging and Progress Reporting

```python
from goldenverba.server.helpers import LoggerManager
from goldenverba.server.types import FileStatus

async def process_documents(self, documents: list[Document], logger: LoggerManager, file_id: str):
    """Example of proper logging"""
    
    await logger.send_report(
        file_id,
        FileStatus.PROCESSING,
        f"Starting to process {len(documents)} documents",
        took=0
    )
    
    start_time = time.time()
    
    try:
        # Process documents
        for i, doc in enumerate(documents):
            await self.process_single_document(doc)
            
            # Progress updates
            if i % 10 == 0:
                await logger.send_report(
                    file_id,
                    FileStatus.PROCESSING,
                    f"Processed {i+1}/{len(documents)} documents",
                    took=time.time() - start_time
                )
        
        elapsed = time.time() - start_time
        await logger.send_report(
            file_id,
            FileStatus.COMPLETED,
            f"Successfully processed all documents",
            took=elapsed
        )
        
    except Exception as e:
        await logger.send_report(
            file_id,
            FileStatus.ERROR,
            f"Error processing documents: {str(e)}",
            took=time.time() - start_time
        )
        raise
```

## Testing and Best Practices

### Unit Testing Components

```python
# goldenverba/tests/components/test_your_component.py
import pytest
from goldenverba.components.your_component import YourComponent
from goldenverba.components.document import Document

class TestYourComponent:
    @pytest.fixture
    def component(self):
        return YourComponent()
    
    @pytest.fixture
    def sample_document(self):
        return Document(
            title="Test Document",
            content="This is a test document with some content.",
            extension="txt"
        )
    
    def test_component_initialization(self, component):
        """Test component initializes correctly"""
        assert component.name != ""
        assert isinstance(component.config, dict)
        assert component.description != ""
    
    def test_configuration_validation(self, component):
        """Test configuration schema is valid"""
        for key, config in component.config.items():
            assert hasattr(config, 'type')
            assert hasattr(config, 'value')
            assert hasattr(config, 'description')
    
    @pytest.mark.asyncio
    async def test_main_functionality(self, component, sample_document):
        """Test the main component functionality"""
        # Implement specific tests for your component
        result = await component.your_main_method([sample_document])
        assert result is not None
        assert len(result) > 0
```

### Integration Testing

```python
# goldenverba/tests/integration/test_rag_pipeline.py
import pytest
from goldenverba.verba_manager import VerbaManager

class TestRAGPipeline:
    @pytest.fixture
    def verba_manager(self):
        return VerbaManager()
    
    @pytest.mark.asyncio
    async def test_full_rag_pipeline(self, verba_manager):
        """Test complete RAG pipeline"""
        # This would require a test Weaviate instance
        # and proper test data setup
        pass
```

### Performance Testing

```python
import time
import asyncio
from goldenverba.components.document import Document

async def benchmark_component(component, test_data, iterations=100):
    """Benchmark component performance"""
    
    start_time = time.time()
    
    for _ in range(iterations):
        await component.process(test_data)
    
    elapsed = time.time() - start_time
    avg_time = elapsed / iterations
    
    print(f"Average time per iteration: {avg_time:.4f}s")
    print(f"Total time for {iterations} iterations: {elapsed:.2f}s")
    
    return avg_time
```

### Best Practices

1. **Input Validation**:
   ```python
   def validate_config(self, config: dict):
       """Validate configuration parameters"""
       required_keys = ["param1", "param2"]
       for key in required_keys:
           if key not in config:
               raise ValueError(f"Missing required parameter: {key}")
   ```

2. **Resource Management**:
   ```python
   async def __aenter__(self):
       """Async context manager entry"""
       self.client = await self.create_client()
       return self
   
   async def __aexit__(self, exc_type, exc_val, exc_tb):
       """Async context manager exit"""
       if self.client:
           await self.client.close()
   ```

3. **Batch Processing**:
   ```python
   async def process_in_batches(self, items: list, batch_size: int = 50):
       """Process items in batches to avoid memory issues"""
       for i in range(0, len(items), batch_size):
           batch = items[i:i + batch_size]
           await self.process_batch(batch)
   ```

4. **Configuration Validation**:
   ```python
   def validate_config(self, config: dict) -> bool:
       """Validate component configuration"""
       for key, input_config in self.config.items():
           if key in config:
               value = config[key].value
               # Type checking
               if input_config.type == "number" and not isinstance(value, (int, float)):
                   raise ValueError(f"{key} must be a number")
               # Range checking for dropdowns
               if input_config.type == "dropdown" and value not in input_config.values:
                   raise ValueError(f"{key} must be one of {input_config.values}")
       return True
   ```

## Troubleshooting

### Common Issues and Solutions

#### 1. Component Not Loading
**Problem**: Custom component not appearing in the interface

**Solutions**:
- Verify the component is imported in `managers.py`
- Check that the component inherits from the correct interface
- Ensure `__init__.py` files are present in directories
- Validate that required libraries are installed

#### 2. Configuration Errors
**Problem**: Component configuration not working properly

**Solutions**:
- Check `InputConfig` definitions match expected types
- Validate default values are appropriate
- Ensure dropdown values list is not empty for dropdown types
- Verify configuration keys match exactly in code

#### 3. Embedding/Vector Issues
**Problem**: Embeddings not generating or searching properly

**Solutions**:
- Verify embedding model is properly loaded
- Check vector dimensions match between embedding and search
- Ensure Weaviate schema is correctly configured
- Validate that chunks contain actual content

#### 4. Memory/Performance Issues
**Problem**: System running out of memory or performing slowly

**Solutions**:
- Implement batch processing for large datasets
- Use streaming for large file processing
- Configure appropriate chunk sizes
- Monitor and limit context window sizes

#### 5. Weaviate Connection Issues
**Problem**: Cannot connect to Weaviate instance

**Solutions**:
- Verify Weaviate URL and API key are correct
- Check network connectivity
- Ensure Weaviate instance is running and accessible
- Validate authentication credentials

### Debug Mode

Enable debug logging for development:

```python
import logging
from wasabi import msg

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)
msg.info("Debug mode enabled")

# Add debug prints in your components
class YourComponent(VerbaComponent):
    async def your_method(self, *args, **kwargs):
        msg.info(f"Processing with args: {args}")
        # Your implementation
        msg.good("Processing completed successfully")
```

### Performance Monitoring

```python
import time
import asyncio
from functools import wraps

def timing_decorator(func):
    """Decorator to measure function execution time"""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start = time.time()
        result = await func(*args, **kwargs)
        elapsed = time.time() - start
        msg.info(f"{func.__name__} took {elapsed:.2f}s")
        return result
    return wrapper

class YourComponent(VerbaComponent):
    @timing_decorator
    async def your_method(self, *args, **kwargs):
        # Your implementation
        pass
```

## Conclusion

This guide provides a comprehensive foundation for contributing to Verba's RAG system. The modular architecture allows for extensive customization while maintaining consistency and reliability. When developing new components:

1. Follow the established patterns and interfaces
2. Implement proper configuration and error handling
3. Write tests for your components
4. Document your implementations
5. Consider performance and scalability implications

For additional support, refer to the main repository documentation and the active community discussions.

---

*This documentation is maintained by the Verba community. Please contribute improvements and report issues.*