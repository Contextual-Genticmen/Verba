"""
Example Custom Chunker Implementation for Verba RAG System

This file demonstrates how to create a custom chunker that implements
advanced chunking strategies with configurable parameters.
"""

from typing import List, Optional
import re
from goldenverba.components.interfaces import Chunker, Embedding
from goldenverba.components.document import Document
from goldenverba.components.chunk import Chunk
from goldenverba.components.types import InputConfig
from wasabi import msg


class ExampleCustomChunker(Chunker):
    """
    Example custom chunker demonstrating advanced chunking techniques
    
    Features:
    - Configurable chunk size and overlap
    - Multiple chunking strategies (sentence, paragraph, semantic)
    - Boundary-aware splitting (respects sentence/paragraph boundaries)
    - Content-aware chunking (preserves code blocks, lists, etc.)
    """
    
    def __init__(self):
        super().__init__()
        self.name = "ExampleCustom"
        self.description = "Advanced custom chunker with multiple strategies"
        self.requires_library = ["nltk", "spacy"]  # Optional dependencies
        
        # Configuration schema
        self.config = {
            "Strategy": InputConfig(
                type="dropdown",
                value="sentence_boundary",
                description="Chunking strategy to use",
                values=[
                    "sentence_boundary", 
                    "paragraph_boundary", 
                    "semantic_clustering",
                    "content_aware"
                ],
            ),
            "Chunk Size": InputConfig(
                type="number",
                value=500,
                description="Target chunk size in characters",
                values=[],
            ),
            "Overlap Percentage": InputConfig(
                type="number", 
                value=10,
                description="Overlap between chunks as percentage (0-50)",
                values=[],
            ),
            "Min Chunk Size": InputConfig(
                type="number",
                value=100,
                description="Minimum chunk size to avoid very small chunks",
                values=[],
            ),
            "Preserve Structure": InputConfig(
                type="dropdown",
                value="enabled",
                description="Preserve document structure (headers, lists, etc.)",
                values=["enabled", "disabled"],
            ),
        }
    
    async def chunk(
        self,
        config: dict,
        documents: List[Document],
        embedder: Optional[Embedding] = None,
        embedder_config: Optional[dict] = None,
    ) -> List[Document]:
        """
        Main chunking method
        
        Args:
            config: Chunker configuration
            documents: List of documents to chunk
            embedder: Optional embedder for semantic chunking
            embedder_config: Optional embedder configuration
            
        Returns:
            List of documents with populated chunks
        """
        try:
            # Extract configuration
            strategy = config["Strategy"].value
            chunk_size = int(config["Chunk Size"].value)
            overlap_percentage = int(config["Overlap Percentage"].value)
            min_chunk_size = int(config["Min Chunk Size"].value)
            preserve_structure = config["Preserve Structure"].value == "enabled"
            
            # Validate configuration
            self._validate_config(chunk_size, overlap_percentage, min_chunk_size)
            
            msg.info(f"Chunking {len(documents)} documents with strategy: {strategy}")
            
            for document in documents:
                # Skip if already chunked
                if len(document.chunks) > 0:
                    msg.info(f"Skipping already chunked document: {document.title}")
                    continue
                
                # Choose chunking strategy
                if strategy == "sentence_boundary":
                    chunks = await self._sentence_boundary_chunking(
                        document.content, chunk_size, overlap_percentage, min_chunk_size
                    )
                elif strategy == "paragraph_boundary":
                    chunks = await self._paragraph_boundary_chunking(
                        document.content, chunk_size, overlap_percentage, min_chunk_size
                    )
                elif strategy == "semantic_clustering":
                    if embedder is None:
                        msg.warn("Semantic chunking requires embedder, falling back to sentence boundary")
                        chunks = await self._sentence_boundary_chunking(
                            document.content, chunk_size, overlap_percentage, min_chunk_size
                        )
                    else:
                        chunks = await self._semantic_clustering_chunking(
                            document.content, chunk_size, embedder, embedder_config
                        )
                elif strategy == "content_aware":
                    chunks = await self._content_aware_chunking(
                        document.content, chunk_size, overlap_percentage, min_chunk_size, preserve_structure
                    )
                else:
                    raise ValueError(f"Unknown chunking strategy: {strategy}")
                
                # Create Chunk objects
                for i, chunk_text in enumerate(chunks):
                    chunk = Chunk(
                        content=chunk_text,
                        chunk_id=i,
                        start_i=None,  # Could be implemented for character indexing
                        end_i=None,
                        content_without_overlap=chunk_text,  # Could implement overlap removal
                    )
                    document.chunks.append(chunk)
                
                msg.good(f"Created {len(chunks)} chunks for document: {document.title}")
            
            return documents
            
        except Exception as e:
            msg.fail(f"Error in {self.name} chunker: {str(e)}")
            raise e
    
    def _validate_config(self, chunk_size: int, overlap_percentage: int, min_chunk_size: int):
        """Validate configuration parameters"""
        if chunk_size <= 0:
            raise ValueError("Chunk size must be positive")
        if not (0 <= overlap_percentage <= 50):
            raise ValueError("Overlap percentage must be between 0 and 50")
        if min_chunk_size <= 0:
            raise ValueError("Minimum chunk size must be positive")
        if min_chunk_size >= chunk_size:
            raise ValueError("Minimum chunk size must be less than target chunk size")
    
    async def _sentence_boundary_chunking(
        self, 
        text: str, 
        chunk_size: int, 
        overlap_percentage: int, 
        min_chunk_size: int
    ) -> List[str]:
        """Chunk text respecting sentence boundaries"""
        # Simple sentence splitting (could be improved with spaCy/NLTK)
        sentences = self._split_into_sentences(text)
        
        chunks = []
        current_chunk = ""
        overlap_size = chunk_size * overlap_percentage // 100
        
        for sentence in sentences:
            # Check if adding this sentence would exceed chunk size
            if len(current_chunk) + len(sentence) > chunk_size and current_chunk:
                # Finalize current chunk if it meets minimum size
                if len(current_chunk.strip()) >= min_chunk_size:
                    chunks.append(current_chunk.strip())
                
                # Start new chunk with overlap
                if overlap_size > 0 and chunks:
                    overlap_text = self._get_overlap_text(current_chunk, overlap_size)
                    current_chunk = overlap_text + " " + sentence
                else:
                    current_chunk = sentence
            else:
                # Add sentence to current chunk
                if current_chunk:
                    current_chunk += " " + sentence
                else:
                    current_chunk = sentence
        
        # Add final chunk
        if current_chunk.strip() and len(current_chunk.strip()) >= min_chunk_size:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    async def _paragraph_boundary_chunking(
        self, 
        text: str, 
        chunk_size: int, 
        overlap_percentage: int, 
        min_chunk_size: int
    ) -> List[str]:
        """Chunk text respecting paragraph boundaries"""
        paragraphs = text.split('\n\n')
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        
        chunks = []
        current_chunk = ""
        overlap_size = chunk_size * overlap_percentage // 100
        
        for paragraph in paragraphs:
            if len(current_chunk) + len(paragraph) > chunk_size and current_chunk:
                if len(current_chunk.strip()) >= min_chunk_size:
                    chunks.append(current_chunk.strip())
                
                # Start new chunk with overlap
                if overlap_size > 0 and chunks:
                    overlap_text = self._get_overlap_text(current_chunk, overlap_size)
                    current_chunk = overlap_text + "\n\n" + paragraph
                else:
                    current_chunk = paragraph
            else:
                if current_chunk:
                    current_chunk += "\n\n" + paragraph
                else:
                    current_chunk = paragraph
        
        if current_chunk.strip() and len(current_chunk.strip()) >= min_chunk_size:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    async def _semantic_clustering_chunking(
        self, 
        text: str, 
        target_size: int, 
        embedder: Embedding, 
        embedder_config: dict
    ) -> List[str]:
        """Chunk text using semantic similarity clustering"""
        sentences = self._split_into_sentences(text)
        
        # Generate embeddings for sentences
        sentence_texts = [s.strip() for s in sentences if s.strip()]
        embeddings = await embedder.vectorize(embedder_config, sentence_texts)
        
        # Simple clustering based on cosine similarity
        chunks = []
        current_cluster = [sentence_texts[0]]
        current_size = len(sentence_texts[0])
        
        for i in range(1, len(sentence_texts)):
            sentence = sentence_texts[i]
            
            # Calculate similarity with current cluster (using last sentence as representative)
            similarity = self._calculate_cosine_similarity(embeddings[i-1], embeddings[i])
            
            # If similar enough and size allows, add to current cluster
            if similarity > 0.7 and current_size + len(sentence) <= target_size:
                current_cluster.append(sentence)
                current_size += len(sentence)
            else:
                # Finalize current cluster and start new one
                if current_cluster:
                    chunks.append(" ".join(current_cluster))
                current_cluster = [sentence]
                current_size = len(sentence)
        
        # Add final cluster
        if current_cluster:
            chunks.append(" ".join(current_cluster))
        
        return chunks
    
    async def _content_aware_chunking(
        self, 
        text: str, 
        chunk_size: int, 
        overlap_percentage: int, 
        min_chunk_size: int,
        preserve_structure: bool
    ) -> List[str]:
        """Chunk text while preserving content structure"""
        if not preserve_structure:
            return await self._sentence_boundary_chunking(text, chunk_size, overlap_percentage, min_chunk_size)
        
        # Identify structural elements
        sections = self._identify_sections(text)
        
        chunks = []
        current_chunk = ""
        overlap_size = chunk_size * overlap_percentage // 100
        
        for section in sections:
            section_text = section["content"]
            section_type = section["type"]
            
            # For code blocks and lists, try to keep them together
            if section_type in ["code", "list"] and len(section_text) <= chunk_size:
                if len(current_chunk) + len(section_text) > chunk_size and current_chunk:
                    # Finalize current chunk
                    if len(current_chunk.strip()) >= min_chunk_size:
                        chunks.append(current_chunk.strip())
                    
                    # Start new chunk with this section
                    current_chunk = section_text
                else:
                    if current_chunk:
                        current_chunk += "\n\n" + section_text
                    else:
                        current_chunk = section_text
            else:
                # For regular text, use sentence-based chunking
                section_chunks = await self._sentence_boundary_chunking(
                    section_text, chunk_size - len(current_chunk), overlap_percentage, min_chunk_size
                )
                
                for section_chunk in section_chunks:
                    if len(current_chunk) + len(section_chunk) > chunk_size and current_chunk:
                        if len(current_chunk.strip()) >= min_chunk_size:
                            chunks.append(current_chunk.strip())
                        current_chunk = section_chunk
                    else:
                        if current_chunk:
                            current_chunk += "\n\n" + section_chunk
                        else:
                            current_chunk = section_chunk
        
        # Add final chunk
        if current_chunk.strip() and len(current_chunk.strip()) >= min_chunk_size:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Simple sentence splitting (could be improved with spaCy/NLTK)"""
        # Basic sentence splitting - in practice, use a proper NLP library
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _get_overlap_text(self, text: str, overlap_size: int) -> str:
        """Get the last overlap_size characters from text"""
        if len(text) <= overlap_size:
            return text
        return text[-overlap_size:]
    
    def _calculate_cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        # Simple dot product implementation
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        magnitude1 = sum(a * a for a in vec1) ** 0.5
        magnitude2 = sum(b * b for b in vec2) ** 0.5
        
        if magnitude1 == 0 or magnitude2 == 0:
            return 0
        
        return dot_product / (magnitude1 * magnitude2)
    
    def _identify_sections(self, text: str) -> List[dict]:
        """Identify different types of content sections"""
        sections = []
        lines = text.split('\n')
        current_section = {"type": "text", "content": ""}
        
        for line in lines:
            line_stripped = line.strip()
            
            # Detect code blocks
            if line_stripped.startswith('```'):
                if current_section["content"]:
                    sections.append(current_section)
                current_section = {"type": "code", "content": ""}
                continue
            
            # Detect list items
            if re.match(r'^[*\-+]\s+', line_stripped) or re.match(r'^\d+\.\s+', line_stripped):
                if current_section["type"] != "list":
                    if current_section["content"]:
                        sections.append(current_section)
                    current_section = {"type": "list", "content": line}
                else:
                    current_section["content"] += "\n" + line
                continue
            
            # Detect headers
            if line_stripped.startswith('#'):
                if current_section["content"]:
                    sections.append(current_section)
                current_section = {"type": "header", "content": line}
                continue
            
            # Regular text
            if current_section["type"] in ["list", "code"] and line_stripped:
                if current_section["content"]:
                    sections.append(current_section)
                current_section = {"type": "text", "content": line}
            else:
                if current_section["content"]:
                    current_section["content"] += "\n" + line
                else:
                    current_section["content"] = line
        
        # Add final section
        if current_section["content"]:
            sections.append(current_section)
        
        return sections


# Example usage and testing
if __name__ == "__main__":
    import asyncio
    
    async def test_custom_chunker():
        """Test the custom chunker"""
        chunker = ExampleCustomChunker()
        
        # Create test document
        test_content = """
        This is the first paragraph with multiple sentences. 
        It contains some interesting information about the topic.
        
        This is the second paragraph. It has different content.
        The sentences here discuss various aspects of the subject.
        
        # Header Section
        
        This section starts with a header.
        
        - This is a list item
        - Another list item with some details
        - Third item in the list
        
        ```python
        def example_function():
            return "This is a code block"
        ```
        
        Final paragraph with concluding thoughts.
        """
        
        document = Document(
            title="Test Document",
            content=test_content,
            extension="md"
        )
        
        # Test different strategies
        strategies = ["sentence_boundary", "paragraph_boundary", "content_aware"]
        
        for strategy in strategies:
            print(f"\n--- Testing {strategy} strategy ---")
            
            config = {
                "Strategy": type('obj', (object,), {'value': strategy}),
                "Chunk Size": type('obj', (object,), {'value': 200}),
                "Overlap Percentage": type('obj', (object,), {'value': 10}),
                "Min Chunk Size": type('obj', (object,), {'value': 50}),
                "Preserve Structure": type('obj', (object,), {'value': "enabled"}),
            }
            
            # Reset document chunks
            document.chunks = []
            
            # Run chunking
            result = await chunker.chunk(config, [document])
            
            print(f"Generated {len(result[0].chunks)} chunks:")
            for i, chunk in enumerate(result[0].chunks):
                print(f"Chunk {i+1} ({len(chunk.content)} chars): {chunk.content[:100]}...")
    
    # Run test
    asyncio.run(test_custom_chunker())