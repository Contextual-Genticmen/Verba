"""
Example Custom Retriever Implementation for Verba RAG System

This file demonstrates how to create a custom retriever that implements
advanced retrieval strategies with configurable parameters.
"""

from typing import List, Dict, Tuple, Optional
import asyncio
from goldenverba.components.interfaces import Retriever
from goldenverba.components.types import InputConfig
from goldenverba.components.managers import WeaviateManager
from goldenverba.components.document import Document
from goldenverba.server.types import ChunkScore
from weaviate.classes.query import Filter, MetadataQuery
from wasabi import msg


class ExampleCustomRetriever(Retriever):
    """
    Example custom retriever demonstrating advanced retrieval techniques
    
    Features:
    - Multiple search strategies (semantic, keyword, hybrid, multi-vector)
    - Configurable ranking and filtering
    - Result diversification
    - Query expansion
    - Reranking capabilities
    """
    
    def __init__(self):
        super().__init__()
        self.name = "ExampleCustom"
        self.description = "Advanced custom retriever with multiple search strategies"
        self.requires_library = ["sentence-transformers"]  # Optional for reranking
        
        # Configuration schema
        self.config = {
            "Search Strategy": InputConfig(
                type="dropdown",
                value="hybrid",
                description="Primary search strategy",
                values=["semantic", "keyword", "hybrid", "multi_vector"],
            ),
            "Max Results": InputConfig(
                type="number",
                value=10,
                description="Maximum number of chunks to retrieve",
                values=[],
            ),
            "Min Score": InputConfig(
                type="number",
                value=0.7,
                description="Minimum similarity score (0-1)",
                values=[],
            ),
            "Query Expansion": InputConfig(
                type="dropdown",
                value="disabled",
                description="Expand query with synonyms/related terms",
                values=["disabled", "synonyms", "semantic"],
            ),
            "Diversification": InputConfig(
                type="dropdown",
                value="enabled",
                description="Ensure result diversity",
                values=["enabled", "disabled"],
            ),
            "Reranking": InputConfig(
                type="dropdown",
                value="disabled",
                description="Rerank results for better relevance",
                values=["disabled", "cross_encoder", "llm_based"],
            ),
            "Window Size": InputConfig(
                type="number",
                value=1,
                description="Number of surrounding chunks to include",
                values=[],
            ),
            "Score Threshold": InputConfig(
                type="number",
                value=80,
                description="Threshold for applying window expansion (1-100)",
                values=[],
            ),
        }
    
    async def retrieve(
        self,
        client,
        query: str,
        vector: List[float],
        config: dict,
        weaviate_manager: WeaviateManager,
        labels: List[str] = [],
        document_uuids: List[str] = [],
    ) -> Tuple[List[Document], str]:
        """
        Main retrieval method
        
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
        try:
            # Extract configuration
            strategy = config["Search Strategy"].value
            max_results = int(config["Max Results"].value)
            min_score = float(config["Min Score"].value)
            query_expansion = config["Query Expansion"].value
            diversification = config["Diversification"].value == "enabled"
            reranking = config["Reranking"].value
            window_size = int(config["Window Size"].value)
            score_threshold = float(config["Score Threshold"].value) / 100.0
            
            msg.info(f"Retrieving with strategy: {strategy}, max_results: {max_results}")
            
            # Step 1: Query expansion
            expanded_queries = await self._expand_query(query, query_expansion)
            
            # Step 2: Retrieve chunks using selected strategy
            if strategy == "semantic":
                results = await self._semantic_search(
                    client, vector, max_results, min_score, labels, document_uuids
                )
            elif strategy == "keyword":
                results = await self._keyword_search(
                    client, query, max_results, labels, document_uuids
                )
            elif strategy == "hybrid":
                results = await self._hybrid_search(
                    client, query, vector, max_results, min_score, labels, document_uuids
                )
            elif strategy == "multi_vector":
                results = await self._multi_vector_search(
                    client, expanded_queries, vector, max_results, min_score, labels, document_uuids
                )
            else:
                raise ValueError(f"Unknown search strategy: {strategy}")
            
            # Step 3: Apply diversification
            if diversification:
                results = await self._diversify_results(results, max_results)
            
            # Step 4: Rerank results
            if reranking != "disabled":
                results = await self._rerank_results(query, results, reranking)
            
            # Step 5: Apply window expansion for high-scoring chunks
            results = await self._apply_window_expansion(
                client, results, window_size, score_threshold
            )
            
            # Step 6: Format results
            documents, context = await self._format_results(results, weaviate_manager)
            
            msg.good(f"Retrieved {len(documents)} documents with {sum(len(doc.chunks) for doc in documents)} chunks")
            
            return documents, context
            
        except Exception as e:
            msg.fail(f"Error in {self.name} retriever: {str(e)}")
            raise e
    
    async def _expand_query(self, query: str, expansion_type: str) -> List[str]:
        """Expand query with related terms"""
        if expansion_type == "disabled":
            return [query]
        
        expanded = [query]
        
        if expansion_type == "synonyms":
            # Simple synonym expansion (in practice, use a thesaurus API)
            synonyms = self._get_synonyms(query)
            expanded.extend(synonyms)
        
        elif expansion_type == "semantic":
            # Semantic expansion using embeddings (simplified)
            semantic_terms = await self._get_semantic_expansions(query)
            expanded.extend(semantic_terms)
        
        return expanded[:3]  # Limit to avoid noise
    
    async def _semantic_search(
        self, 
        client, 
        vector: List[float], 
        max_results: int, 
        min_score: float,
        labels: List[str], 
        document_uuids: List[str]
    ) -> List[dict]:
        """Perform semantic vector search"""
        try:
            collection = client.collections.get("Verba")
            
            # Build filters
            filters = self._build_filters(labels, document_uuids)
            
            # Perform search
            response = collection.query.near_vector(
                near_vector=vector,
                limit=max_results * 2,  # Get more for filtering
                where=filters,
                return_metadata=MetadataQuery(score=True, distance=True)
            )
            
            # Filter by minimum score
            results = []
            for obj in response.objects:
                score = obj.metadata.score or 0
                if score >= min_score:
                    results.append({
                        "chunk": obj,
                        "score": score,
                        "document_uuid": obj.properties.get("doc_uuid", ""),
                        "chunk_id": obj.properties.get("chunk_id", 0),
                    })
            
            # Sort by score and limit
            results.sort(key=lambda x: x["score"], reverse=True)
            return results[:max_results]
            
        except Exception as e:
            msg.fail(f"Semantic search failed: {str(e)}")
            return []
    
    async def _keyword_search(
        self, 
        client, 
        query: str, 
        max_results: int,
        labels: List[str], 
        document_uuids: List[str]
    ) -> List[dict]:
        """Perform keyword-based search"""
        try:
            collection = client.collections.get("Verba")
            
            # Build filters
            filters = self._build_filters(labels, document_uuids)
            
            # Perform BM25 search
            response = collection.query.bm25(
                query=query,
                limit=max_results,
                where=filters,
                return_metadata=MetadataQuery(score=True)
            )
            
            results = []
            for obj in response.objects:
                results.append({
                    "chunk": obj,
                    "score": obj.metadata.score or 0,
                    "document_uuid": obj.properties.get("doc_uuid", ""),
                    "chunk_id": obj.properties.get("chunk_id", 0),
                })
            
            return results
            
        except Exception as e:
            msg.fail(f"Keyword search failed: {str(e)}")
            return []
    
    async def _hybrid_search(
        self, 
        client, 
        query: str, 
        vector: List[float], 
        max_results: int, 
        min_score: float,
        labels: List[str], 
        document_uuids: List[str]
    ) -> List[dict]:
        """Perform hybrid search combining semantic and keyword"""
        try:
            collection = client.collections.get("Verba")
            
            # Build filters
            filters = self._build_filters(labels, document_uuids)
            
            # Perform hybrid search
            response = collection.query.hybrid(
                query=query,
                vector=vector,
                limit=max_results,
                where=filters,
                return_metadata=MetadataQuery(score=True)
            )
            
            results = []
            for obj in response.objects:
                score = obj.metadata.score or 0
                if score >= min_score:
                    results.append({
                        "chunk": obj,
                        "score": score,
                        "document_uuid": obj.properties.get("doc_uuid", ""),
                        "chunk_id": obj.properties.get("chunk_id", 0),
                    })
            
            return results
            
        except Exception as e:
            msg.fail(f"Hybrid search failed: {str(e)}")
            return []
    
    async def _multi_vector_search(
        self, 
        client, 
        queries: List[str], 
        main_vector: List[float], 
        max_results: int, 
        min_score: float,
        labels: List[str], 
        document_uuids: List[str]
    ) -> List[dict]:
        """Perform search using multiple query vectors"""
        all_results = []
        
        # Search with main vector
        semantic_results = await self._semantic_search(
            client, main_vector, max_results, min_score, labels, document_uuids
        )
        
        # Weight main results higher
        for result in semantic_results:
            result["score"] *= 1.0  # Full weight for main query
            all_results.append(result)
        
        # Search with expanded queries (if different from main)
        for i, query in enumerate(queries[1:], 1):  # Skip first (main) query
            keyword_results = await self._keyword_search(
                client, query, max_results // 2, labels, document_uuids
            )
            
            # Weight expanded results lower
            weight = 0.7 / i  # Decreasing weight for each expansion
            for result in keyword_results:
                result["score"] *= weight
                all_results.append(result)
        
        # Merge and deduplicate
        merged_results = self._merge_and_deduplicate(all_results)
        
        # Sort by combined score and limit
        merged_results.sort(key=lambda x: x["score"], reverse=True)
        return merged_results[:max_results]
    
    async def _diversify_results(self, results: List[dict], max_results: int) -> List[dict]:
        """Ensure diversity in results to avoid redundancy"""
        if not results or len(results) <= max_results:
            return results
        
        diversified = [results[0]]  # Always include top result
        
        for result in results[1:]:
            if len(diversified) >= max_results:
                break
            
            # Check if this result is sufficiently different from selected ones
            is_diverse = True
            result_content = result["chunk"].properties.get("text", "").lower()
            
            for selected in diversified:
                selected_content = selected["chunk"].properties.get("text", "").lower()
                
                # Simple diversity check using word overlap
                if self._calculate_content_overlap(result_content, selected_content) > 0.8:
                    is_diverse = False
                    break
            
            if is_diverse:
                diversified.append(result)
        
        return diversified
    
    async def _rerank_results(self, query: str, results: List[dict], reranking_type: str) -> List[dict]:
        """Rerank results for better relevance"""
        if not results or reranking_type == "disabled":
            return results
        
        if reranking_type == "cross_encoder":
            return await self._cross_encoder_rerank(query, results)
        elif reranking_type == "llm_based":
            return await self._llm_based_rerank(query, results)
        
        return results
    
    async def _cross_encoder_rerank(self, query: str, results: List[dict]) -> List[dict]:
        """Rerank using a cross-encoder model"""
        try:
            # This would require sentence-transformers
            # For now, implement a simple relevance scoring
            
            for result in results:
                content = result["chunk"].properties.get("text", "")
                # Simple relevance scoring based on query term overlap
                relevance = self._calculate_query_relevance(query, content)
                # Combine with original score
                result["score"] = (result["score"] + relevance) / 2
            
            results.sort(key=lambda x: x["score"], reverse=True)
            return results
            
        except Exception as e:
            msg.warn(f"Cross-encoder reranking failed: {str(e)}")
            return results
    
    async def _llm_based_rerank(self, query: str, results: List[dict]) -> List[dict]:
        """Rerank using LLM-based relevance scoring"""
        # This would involve calling an LLM to score relevance
        # For now, return original results
        msg.info("LLM-based reranking not implemented in this example")
        return results
    
    async def _apply_window_expansion(
        self, 
        client, 
        results: List[dict], 
        window_size: int, 
        score_threshold: float
    ) -> List[dict]:
        """Apply window expansion for high-scoring chunks"""
        if window_size <= 0:
            return results
        
        expanded_results = []
        
        for result in results:
            chunk_score = result["score"]
            expanded_results.append(result)
            
            # Apply window expansion if score is above threshold
            if chunk_score >= score_threshold:
                document_uuid = result["document_uuid"]
                chunk_id = result["chunk_id"]
                
                # Get surrounding chunks
                surrounding_chunks = await self._get_surrounding_chunks(
                    client, document_uuid, chunk_id, window_size
                )
                
                # Add surrounding chunks with lower scores
                for surrounding_chunk in surrounding_chunks:
                    expanded_results.append({
                        "chunk": surrounding_chunk,
                        "score": chunk_score * 0.8,  # Lower score for context
                        "document_uuid": document_uuid,
                        "chunk_id": surrounding_chunk.properties.get("chunk_id", 0),
                        "is_context": True,
                    })
        
        return expanded_results
    
    async def _get_surrounding_chunks(
        self, 
        client, 
        document_uuid: str, 
        chunk_id: int, 
        window_size: int
    ) -> List:
        """Get chunks surrounding the given chunk"""
        try:
            collection = client.collections.get("Verba")
            
            # Get chunks from same document with nearby chunk IDs
            chunk_ids = []
            for i in range(max(0, chunk_id - window_size), chunk_id + window_size + 1):
                if i != chunk_id:  # Exclude the original chunk
                    chunk_ids.append(i)
            
            if not chunk_ids:
                return []
            
            # Build filter for document and chunk IDs
            doc_filter = Filter.by_property("doc_uuid").equal(document_uuid)
            chunk_filter = Filter.by_property("chunk_id").contains_any(chunk_ids)
            combined_filter = doc_filter & chunk_filter
            
            response = collection.query.fetch_objects(
                where=combined_filter,
                limit=window_size * 2
            )
            
            return list(response.objects)
            
        except Exception as e:
            msg.warn(f"Failed to get surrounding chunks: {str(e)}")
            return []
    
    async def _format_results(
        self, 
        results: List[dict], 
        weaviate_manager: WeaviateManager
    ) -> Tuple[List[Document], str]:
        """Format results into documents and context string"""
        documents = {}
        context_parts = []
        chunk_scores = []
        
        for result in results:
            chunk_obj = result["chunk"]
            score = result["score"]
            doc_uuid = result["document_uuid"]
            is_context = result.get("is_context", False)
            
            # Extract chunk properties
            chunk_text = chunk_obj.properties.get("text", "")
            chunk_id = chunk_obj.properties.get("chunk_id", 0)
            doc_title = chunk_obj.properties.get("doc_title", "Unknown")
            doc_type = chunk_obj.properties.get("doc_type", "Document")
            
            # Create or update document
            if doc_uuid not in documents:
                documents[doc_uuid] = Document(
                    title=doc_title,
                    content="",  # Will be populated from chunks
                    extension=doc_type.lower(),
                    uuid=doc_uuid,
                )
            
            # Create chunk score
            chunk_score = ChunkScore(
                text=chunk_text,
                score=score,
                uuid=doc_uuid,
                chunk_id=chunk_id,
                doc_title=doc_title,
                doc_type=doc_type,
            )
            chunk_scores.append(chunk_score)
            
            # Add to context (mark context chunks differently)
            if is_context:
                context_parts.append(f"[Context] {chunk_text}")
            else:
                context_parts.append(chunk_text)
        
        # Convert to document list
        document_list = list(documents.values())
        
        # Add chunk scores to documents
        for doc in document_list:
            doc.chunk_scores = [cs for cs in chunk_scores if cs.uuid == doc.uuid]
        
        # Create context string
        context = "\n\n".join(context_parts)
        
        return document_list, context
    
    def _build_filters(self, labels: List[str], document_uuids: List[str]) -> Optional[Filter]:
        """Build Weaviate filters from labels and document UUIDs"""
        filters = []
        
        if labels:
            label_filter = Filter.by_property("doc_labels").contains_any(labels)
            filters.append(label_filter)
        
        if document_uuids:
            uuid_filter = Filter.by_property("doc_uuid").contains_any(document_uuids)
            filters.append(uuid_filter)
        
        if not filters:
            return None
        
        # Combine filters with AND
        combined_filter = filters[0]
        for filter_obj in filters[1:]:
            combined_filter = combined_filter & filter_obj
        
        return combined_filter
    
    def _get_synonyms(self, query: str) -> List[str]:
        """Get synonyms for query terms (simplified implementation)"""
        # This would use a thesaurus API or WordNet in practice
        synonym_map = {
            "fast": ["quick", "rapid", "speedy"],
            "big": ["large", "huge", "massive"],
            "small": ["tiny", "little", "miniature"],
            # Add more mappings as needed
        }
        
        synonyms = []
        words = query.lower().split()
        
        for word in words:
            if word in synonym_map:
                synonyms.extend(synonym_map[word])
        
        return synonyms[:2]  # Limit to avoid noise
    
    async def _get_semantic_expansions(self, query: str) -> List[str]:
        """Get semantically related terms (simplified implementation)"""
        # This would use word embeddings or a language model in practice
        expansion_map = {
            "machine learning": ["artificial intelligence", "deep learning", "neural networks"],
            "database": ["data storage", "SQL", "NoSQL"],
            "programming": ["coding", "software development", "algorithms"],
        }
        
        query_lower = query.lower()
        for key, expansions in expansion_map.items():
            if key in query_lower:
                return expansions[:2]
        
        return []
    
    def _calculate_content_overlap(self, content1: str, content2: str) -> float:
        """Calculate content overlap between two texts"""
        words1 = set(content1.split())
        words2 = set(content2.split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        return intersection / union if union > 0 else 0.0
    
    def _calculate_query_relevance(self, query: str, content: str) -> float:
        """Calculate relevance score between query and content"""
        query_words = set(query.lower().split())
        content_words = set(content.lower().split())
        
        if not query_words:
            return 0.0
        
        matches = len(query_words & content_words)
        return matches / len(query_words)
    
    def _merge_and_deduplicate(self, results: List[dict]) -> List[dict]:
        """Merge results and deduplicate by chunk ID"""
        seen_chunks = {}
        
        for result in results:
            chunk_id = result["chunk_id"]
            doc_uuid = result["document_uuid"]
            key = f"{doc_uuid}_{chunk_id}"
            
            if key not in seen_chunks or result["score"] > seen_chunks[key]["score"]:
                seen_chunks[key] = result
        
        return list(seen_chunks.values())


# Example usage and testing
if __name__ == "__main__":
    import asyncio
    
    async def test_custom_retriever():
        """Test the custom retriever"""
        retriever = ExampleCustomRetriever()
        
        print("Custom Retriever Configuration:")
        for key, config in retriever.config.items():
            print(f"  {key}: {config.value} ({config.type})")
        
        print(f"\nRetriever: {retriever.name}")
        print(f"Description: {retriever.description}")
        print(f"Required libraries: {retriever.requires_library}")
    
    # Run test
    asyncio.run(test_custom_retriever())