# RAG Development Quick Reference

## Core Component Interfaces

### Chunker Interface
```python
class YourChunker(Chunker):
    async def chunk(self, config: dict, documents: list[Document], 
                   embedder: Embedding | None = None, 
                   embedder_config: dict | None = None) -> list[Document]:
        # Implementation here
        pass
```

### Retriever Interface  
```python
class YourRetriever(Retriever):
    async def retrieve(self, client, query: str, vector: list[float], 
                      config: dict, weaviate_manager: WeaviateManager,
                      labels: list[str] = [], document_uuids: list[str] = []):
        # Implementation here
        pass
```

### Generator Interface
```python
class YourGenerator(Generator):
    def prepare_messages(self, query: str, context: str, 
                        conversation: list[dict] = None) -> list[dict]:
        # Implementation here
        pass
        
    async def generate_stream(self, config: dict, query: str, 
                             context: str, conversation: list[dict]):
        # Implementation here
        pass
```

## Key File Locations

- **Component Interfaces**: `goldenverba/components/interfaces.py`
- **Component Managers**: `goldenverba/components/managers.py`
- **Main RAG Logic**: `goldenverba/verba_manager.py`
- **Chunkers**: `goldenverba/components/chunking/`
- **Retrievers**: `goldenverba/components/retriever/`
- **Generators**: `goldenverba/components/generation/`
- **Embedders**: `goldenverba/components/embedding/`

## Registration Pattern

1. Create your component in the appropriate directory
2. Import it in `goldenverba/components/managers.py`
3. Add to the respective list (chunkers, retrievers, generators, etc.)

```python
# In managers.py
from goldenverba.components.chunking.YourChunker import YourChunker

chunkers = [
    TokenChunker(),
    SentenceChunker(),
    YourChunker(),  # Add here
]
```

## Configuration Schema

```python
self.config = {
    "Parameter Name": InputConfig(
        type="dropdown|number|text|textarea|password|multi",
        value="default_value",
        description="User-friendly description",
        values=["list", "of", "options"],  # For dropdown/multi
    ),
}
```

## Memory Management Patterns

### Sliding Window
```python
def sliding_window_memory(self, conversation: list[dict], window_size: int):
    return conversation[-(window_size * 2):]  # Keep last N exchanges
```

### Context Truncation
```python
def truncate_to_token_limit(self, text: str, max_tokens: int):
    estimated_tokens = len(text) // 4  # ~4 chars per token
    if estimated_tokens > max_tokens:
        char_limit = max_tokens * 4
        return text[:char_limit]
    return text
```

## Vector Database Operations

### Weaviate Search
```python
collection = client.collections.get("Verba")
response = collection.query.near_vector(
    near_vector=query_vector,
    limit=limit,
    where=filters,
    return_metadata=MetadataQuery(score=True)
)
```

### Metadata Filtering
```python
label_filter = Filter.by_property("doc_labels").contains_any(labels)
uuid_filter = Filter.by_property("doc_uuid").contains_any(document_uuids)
combined_filter = label_filter & uuid_filter
```

## Testing Patterns

```python
import pytest
from goldenverba.components.document import Document

class TestYourComponent:
    @pytest.fixture
    def component(self):
        return YourComponent()
    
    @pytest.mark.asyncio
    async def test_functionality(self, component):
        # Test implementation
        pass
```

## Common Troubleshooting

1. **Component not loading**: Check imports in `managers.py`
2. **Config errors**: Validate `InputConfig` definitions  
3. **Vector issues**: Verify dimensions and content
4. **Memory issues**: Implement batch processing
5. **Connection issues**: Check Weaviate credentials

For detailed examples and complete implementation guidance, see the full RAG_Dev.md documentation.