"""
Sample Python code file for testing MCP RAG server file operations.

This file contains example code that will be used to test:
- File reading via MCP filesystem tools
- Document chunking on real code content
- RAG operations (indexing and retrieval) on source code
"""


class DocumentProcessor:
    """A sample class that processes documents for RAG operations."""
    
    def __init__(self, chunk_size=500):
        """
        Initialize the document processor.
        
        Args:
            chunk_size: Size of chunks for document processing
        """
        self.chunk_size = chunk_size
        self.documents = []
    
    def add_document(self, title, content):
        """
        Add a document to the processor.
        
        Args:
            title: Document title
            content: Document content
            
        Returns:
            Document ID
        """
        doc_id = len(self.documents)
        self.documents.append({
            'id': doc_id,
            'title': title,
            'content': content
        })
        return doc_id
    
    def chunk_document(self, doc_id):
        """
        Chunk a document into smaller pieces.
        
        Args:
            doc_id: ID of document to chunk
            
        Returns:
            List of chunks
        """
        if doc_id >= len(self.documents):
            raise ValueError(f"Document {doc_id} not found")
        
        doc = self.documents[doc_id]
        content = doc['content']
        chunks = []
        
        for i in range(0, len(content), self.chunk_size):
            chunk = content[i:i + self.chunk_size]
            chunks.append({
                'doc_id': doc_id,
                'chunk_id': len(chunks),
                'content': chunk
            })
        
        return chunks
    
    def retrieve_relevant_chunks(self, query, top_k=3):
        """
        Retrieve relevant chunks for a query.
        
        Args:
            query: Search query
            top_k: Number of results to return
            
        Returns:
            List of relevant chunks
        """
        # Simple keyword matching for demo
        results = []
        for doc in self.documents:
            if query.lower() in doc['content'].lower():
                results.append(doc)
        
        return results[:top_k]


def main():
    """Example usage of DocumentProcessor."""
    processor = DocumentProcessor(chunk_size=100)
    
    # Add some sample documents
    doc_id = processor.add_document(
        "Sample Doc",
        "This is a test document for RAG operations. "
        "It demonstrates how documents can be processed and chunked."
    )
    
    # Chunk the document
    chunks = processor.chunk_document(doc_id)
    print(f"Created {len(chunks)} chunks from document {doc_id}")
    
    # Retrieve relevant chunks
    results = processor.retrieve_relevant_chunks("RAG operations")
    print(f"Found {len(results)} relevant documents")


if __name__ == "__main__":
    main()
