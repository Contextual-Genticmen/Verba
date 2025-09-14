# Verba RAG System Examples

This directory contains comprehensive examples demonstrating how to create and customize components for Verba's RAG (Retrieval-Augmented Generation) system.

## Overview

Verba's RAG system is built on a modular architecture that allows developers to easily extend and customize various components. These examples provide practical implementations and best practices for:

- **Custom Chunkers**: Advanced text splitting strategies
- **Custom Retrievers**: Sophisticated document retrieval methods  
- **Custom Generators**: Enhanced response generation with memory management

## Files in this Directory

### 1. `custom_chunker_example.py`
**Complete implementation of an advanced custom chunker**

Features demonstrated:
- Multiple chunking strategies (sentence, paragraph, semantic, content-aware)
- Configurable parameters (chunk size, overlap, minimum size)
- Boundary-aware splitting (respects sentence/paragraph boundaries)
- Content structure preservation (code blocks, lists, headers)
- Semantic clustering using embeddings
- Comprehensive error handling and validation

Key concepts:
```python
class ExampleCustomChunker(Chunker):
    async def chunk(self, config, documents, embedder=None, embedder_config=None):
        # Your chunking implementation
        pass
```

### 2. `custom_retriever_example.py`
**Complete implementation of an advanced custom retriever**

Features demonstrated:
- Multiple search strategies (semantic, keyword, hybrid, multi-vector)
- Query expansion with synonyms and semantic terms
- Result diversification to avoid redundancy
- Configurable reranking (cross-encoder, LLM-based)
- Window expansion for context
- Advanced filtering and scoring

Key concepts:
```python
class ExampleCustomRetriever(Retriever):
    async def retrieve(self, client, query, vector, config, weaviate_manager, labels=[], document_uuids=[]):
        # Your retrieval implementation
        return documents, context
```

### 3. `custom_generator_example.py`
**Complete implementation of an advanced custom generator**

Features demonstrated:
- Multiple memory management strategies (sliding window, summarization, entity tracking, adaptive)
- Context window management and truncation
- Streaming response generation
- Configurable response formatting
- Error handling and fallback models
- Advanced conversation analysis

Key concepts:
```python
class ExampleCustomGenerator(Generator):
    async def generate_stream(self, config, query, context, conversation):
        # Your generation implementation
        yield {"message": chunk, "finished": False}
```

## Getting Started

### Prerequisites

Before running these examples, ensure you have:

1. **Verba installed**: Follow the main installation guide
2. **Required dependencies**: Each example lists its `requires_library` 
3. **API keys**: Set up environment variables for external services (optional)
4. **Test environment**: A development Weaviate instance for testing

### Basic Usage

1. **Study the examples**: Review the code and comments to understand the patterns
2. **Copy and modify**: Use the examples as templates for your own components
3. **Test incrementally**: Start with basic functionality and add complexity
4. **Register components**: Add your components to the managers in `goldenverba/components/managers.py`

### Running the Examples

Each example file can be run independently for testing:

```bash
# Test the custom chunker
python examples/custom_chunker_example.py

# Test the custom retriever (requires Weaviate connection)
python examples/custom_retriever_example.py

# Test the custom generator (requires API keys for full functionality)
python examples/custom_generator_example.py
```

## Implementation Patterns

### 1. Component Structure

All components follow this basic structure:

```python
class YourCustomComponent(ComponentInterface):
    def __init__(self):
        super().__init__()
        self.name = "YourCustom"
        self.description = "Description of your component"
        self.requires_library = ["optional_dependencies"]
        self.requires_env = ["OPTIONAL_API_KEYS"]
        
        # Configuration schema
        self.config = {
            "Parameter": InputConfig(
                type="dropdown|number|text|textarea|password|multi",
                value="default_value",
                description="User-friendly description",
                values=["options", "for", "dropdown"],
            ),
        }
    
    async def main_method(self, *args, **kwargs):
        # Your implementation here
        pass
```

### 2. Configuration Management

Use `InputConfig` to define user-configurable parameters:

```python
from goldenverba.components.types import InputConfig

self.config = {
    "Strategy": InputConfig(
        type="dropdown",
        value="default_option",
        description="Choose the processing strategy",
        values=["option1", "option2", "option3"],
    ),
    "Size": InputConfig(
        type="number",
        value=500,
        description="Processing size parameter",
        values=[],  # Empty for number inputs
    ),
}
```

### 3. Error Handling

Implement robust error handling:

```python
from wasabi import msg

try:
    # Your implementation
    result = await self.process_data(data)
    msg.good("Processing completed successfully")
    return result
except ValueError as e:
    msg.warn(f"Configuration error: {e}")
    raise e
except Exception as e:
    msg.fail(f"Unexpected error in {self.name}: {e}")
    raise e
```

### 4. Async Patterns

Use proper async patterns for I/O operations:

```python
async def process_documents(self, documents):
    tasks = []
    for document in documents:
        task = self.process_single_document(document)
        tasks.append(task)
    
    results = await asyncio.gather(*tasks, return_exceptions=True)
    return [r for r in results if not isinstance(r, Exception)]
```

## Advanced Concepts

### Memory Management Strategies

The generator example demonstrates several memory management approaches:

1. **Sliding Window**: Keep only recent conversation exchanges
2. **Summarization**: Summarize old conversation and keep recent details
3. **Entity Tracking**: Track important entities and topics across conversation
4. **Adaptive**: Choose strategy based on conversation characteristics

### Retrieval Strategies

The retriever example shows advanced retrieval techniques:

1. **Hybrid Search**: Combine semantic and keyword search
2. **Query Expansion**: Add related terms to improve recall
3. **Result Diversification**: Ensure variety in retrieved results
4. **Reranking**: Improve relevance using secondary models

### Chunking Strategies

The chunker example implements various chunking approaches:

1. **Boundary-Aware**: Respect sentence and paragraph boundaries
2. **Semantic Clustering**: Group related sentences using embeddings
3. **Content-Aware**: Preserve structure of code, lists, headers
4. **Configurable Overlap**: Flexible overlap strategies

## Integration Guidelines

### 1. Component Registration

Add your components to the managers:

```python
# In goldenverba/components/managers.py
from goldenverba.components.chunking.YourChunker import YourChunker

chunkers = [
    TokenChunker(),
    SentenceChunker(),
    YourChunker(),  # Add here
]
```

### 2. Testing

Create comprehensive tests:

```python
import pytest
from goldenverba.components.document import Document

class TestYourComponent:
    @pytest.fixture
    def component(self):
        return YourComponent()
    
    @pytest.mark.asyncio
    async def test_functionality(self, component):
        # Test your component
        pass
```

### 3. Documentation

Document your components:

```python
class YourComponent(ComponentInterface):
    """
    Brief description of your component
    
    Features:
    - Feature 1
    - Feature 2
    
    Configuration:
    - parameter1: Description
    - parameter2: Description
    """
```

## Best Practices

1. **Start Simple**: Begin with basic functionality and iterate
2. **Follow Patterns**: Use the established patterns from existing components
3. **Handle Errors**: Implement comprehensive error handling
4. **Test Thoroughly**: Write tests for edge cases and error conditions
5. **Document Well**: Provide clear documentation and examples
6. **Performance**: Consider memory usage and processing time
7. **Configuration**: Make components highly configurable
8. **Backward Compatibility**: Maintain compatibility with existing interfaces

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all required libraries are installed
2. **Configuration Issues**: Validate `InputConfig` definitions
3. **Async Issues**: Use proper async/await patterns
4. **Memory Issues**: Implement batch processing for large datasets
5. **API Issues**: Handle rate limits and network errors gracefully

### Debug Tips

1. **Use Logging**: Leverage `wasabi.msg` for status messages
2. **Test Incrementally**: Build and test functionality step by step
3. **Check Types**: Ensure proper type annotations and validation
4. **Monitor Performance**: Track execution times and memory usage

## Contributing

When contributing your custom components:

1. **Follow Code Style**: Use Black for formatting
2. **Add Tests**: Include comprehensive test coverage
3. **Update Documentation**: Add your component to relevant docs
4. **Consider Dependencies**: Minimize external dependencies
5. **Handle Gracefully**: Ensure components fail gracefully

## Support

For questions and support:

1. **Review Documentation**: Check the main RAG_Dev.md guide
2. **Study Examples**: Use these examples as reference
3. **Community Discussion**: Participate in GitHub Discussions
4. **Issue Tracking**: Report bugs and feature requests

---

These examples provide a solid foundation for extending Verba's RAG capabilities. Use them as starting points and adapt them to your specific needs. Happy coding!