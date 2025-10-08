"""
Example Custom Generator Implementation for Verba RAG System

This file demonstrates how to create a custom generator that implements
advanced conversation memory management and generation strategies.
"""

from typing import List, Dict, Optional, AsyncGenerator
import json
import re
from goldenverba.components.interfaces import Generator
from goldenverba.components.types import InputConfig
from goldenverba.components.util import get_environment, get_token
from wasabi import msg


class ExampleCustomGenerator(Generator):
    """
    Example custom generator demonstrating advanced memory management and generation
    
    Features:
    - Multiple memory strategies (sliding window, summarization, entity tracking)
    - Context window management
    - Streaming response generation
    - Configurable response formatting
    - Error handling and fallbacks
    """
    
    def __init__(self):
        super().__init__()
        self.name = "ExampleCustom"
        self.description = "Advanced custom generator with memory management"
        self.context_window = 8000  # Model context limit
        self.requires_env = ["CUSTOM_API_KEY"]  # If using external API
        self.requires_library = ["aiohttp", "tiktoken"]  # For API calls and token counting
        
        # Configuration schema
        self.config = {
            "Model": InputConfig(
                type="dropdown",
                value="gpt-3.5-turbo",
                description="Language model to use",
                values=["gpt-3.5-turbo", "gpt-4", "claude-3", "local-llama"],
            ),
            "Memory Strategy": InputConfig(
                type="dropdown",
                value="sliding_window",
                description="Conversation memory management strategy",
                values=["sliding_window", "summarization", "entity_tracking", "adaptive"],
            ),
            "Memory Length": InputConfig(
                type="number",
                value=6,
                description="Number of previous exchanges to remember",
                values=[],
            ),
            "Max Tokens": InputConfig(
                type="number",
                value=2000,
                description="Maximum tokens in response",
                values=[],
            ),
            "Temperature": InputConfig(
                type="number",
                value=0.7,
                description="Response creativity (0.0-1.0)",
                values=[],
            ),
            "Response Format": InputConfig(
                type="dropdown",
                value="conversational",
                description="Response formatting style",
                values=["conversational", "academic", "bullet_points", "structured"],
            ),
            "Citations": InputConfig(
                type="dropdown",
                value="enabled",
                description="Include source citations in responses",
                values=["enabled", "disabled", "footnotes"],
            ),
            "Fallback Model": InputConfig(
                type="dropdown",
                value="gpt-3.5-turbo",
                description="Fallback model if primary fails",
                values=["gpt-3.5-turbo", "claude-3", "local-llama"],
            ),
        }
        
        # Initialize system message with context placeholder
        default_prompt = """You are Verba, an advanced AI assistant specialized in Retrieval Augmented Generation (RAG). 

You have access to relevant context from a knowledge base. Your primary responsibilities are:

1. Answer questions accurately using the provided context
2. Cite sources when using specific information
3. Acknowledge when information is incomplete or unavailable
4. Maintain conversation coherence across multiple exchanges
5. Provide helpful, clear, and well-structured responses

Context Guidelines:
- Use the provided context as your primary information source
- Clearly distinguish between information from the context vs. your general knowledge
- When citing sources, reference the document titles and relevant sections
- If the context doesn't contain enough information, say so explicitly

Response Guidelines:
- Structure your responses clearly with appropriate formatting
- Use code blocks for code examples with proper language tags
- Break down complex topics into digestible sections
- Maintain a helpful and professional tone"""

        self.config["System Message"] = InputConfig(
            type="textarea",
            value=default_prompt,
            description="System message for the AI assistant",
            values=[],
        )
    
    async def generate_stream(
        self, 
        config: dict, 
        query: str, 
        context: str, 
        conversation: List[dict]
    ) -> AsyncGenerator[dict, None]:
        """
        Generate streaming response with advanced memory management
        
        Args:
            config: Generator configuration
            query: Current user query
            context: Retrieved context from RAG
            conversation: Previous conversation history
            
        Yields:
            Dict with response chunks: {"message": str, "finished": bool}
        """
        try:
            # Extract configuration
            model = config["Model"].value
            memory_strategy = config["Memory Strategy"].value
            memory_length = int(config["Memory Length"].value)
            max_tokens = int(config["Max Tokens"].value)
            temperature = float(config["Temperature"].value)
            response_format = config["Response Format"].value
            citations_enabled = config["Citations"].value
            fallback_model = config["Fallback Model"].value
            
            msg.info(f"Generating response with {model}, memory: {memory_strategy}")
            
            # Prepare messages with memory management
            messages = await self._prepare_messages_with_memory(
                query, context, conversation, memory_strategy, memory_length, config
            )
            
            # Add response formatting instructions
            messages = self._add_formatting_instructions(messages, response_format, citations_enabled)
            
            # Generate response with primary model
            try:
                async for chunk in self._generate_with_model(
                    model, messages, max_tokens, temperature
                ):
                    yield chunk
                    
            except Exception as e:
                msg.warn(f"Primary model {model} failed: {str(e)}, trying fallback")
                
                # Fallback to secondary model
                async for chunk in self._generate_with_model(
                    fallback_model, messages, max_tokens, temperature
                ):
                    yield chunk
            
        except Exception as e:
            msg.fail(f"Error in {self.name} generator: {str(e)}")
            yield {
                "message": f"I apologize, but I encountered an error while generating a response: {str(e)}",
                "finished": True
            }
    
    async def _prepare_messages_with_memory(
        self, 
        query: str, 
        context: str, 
        conversation: List[dict], 
        memory_strategy: str, 
        memory_length: int,
        config: dict
    ) -> List[dict]:
        """Prepare messages using the specified memory strategy"""
        
        if memory_strategy == "sliding_window":
            return await self._sliding_window_memory(query, context, conversation, memory_length, config)
        elif memory_strategy == "summarization":
            return await self._summarization_memory(query, context, conversation, config)
        elif memory_strategy == "entity_tracking":
            return await self._entity_tracking_memory(query, context, conversation, config)
        elif memory_strategy == "adaptive":
            return await self._adaptive_memory(query, context, conversation, memory_length, config)
        else:
            # Default to sliding window
            return await self._sliding_window_memory(query, context, conversation, memory_length, config)
    
    async def _sliding_window_memory(
        self, 
        query: str, 
        context: str, 
        conversation: List[dict], 
        window_size: int,
        config: dict
    ) -> List[dict]:
        """Sliding window memory: keep only recent exchanges"""
        messages = []
        
        # System message with context
        system_content = self._build_system_message(context, config)
        messages.append({"role": "system", "content": system_content})
        
        if conversation:
            # Take only recent exchanges (user + assistant pairs)
            recent_conversation = conversation[-(window_size * 2):]
            
            for msg in recent_conversation[:-1]:  # Exclude current query
                role = "user" if msg["type"] == "user" else "assistant"
                messages.append({"role": role, "content": msg["content"]})
        
        # Current query
        messages.append({"role": "user", "content": query})
        
        # Ensure within context window
        return await self._truncate_to_context_window(messages)
    
    async def _summarization_memory(
        self, 
        query: str, 
        context: str, 
        conversation: List[dict],
        config: dict
    ) -> List[dict]:
        """Summarization memory: summarize old conversation and keep recent"""
        messages = []
        
        if not conversation or len(conversation) <= 6:
            # Not enough history to summarize, use sliding window
            return await self._sliding_window_memory(query, context, conversation, 3, config)
        
        # Split conversation into old and recent parts
        old_conversation = conversation[:-6]  # Older exchanges to summarize
        recent_conversation = conversation[-6:]  # Recent exchanges to keep
        
        # Generate summary of old conversation
        summary = await self._generate_conversation_summary(old_conversation)
        
        # System message with context and summary
        system_content = self._build_system_message(context, config)
        system_content += f"\n\nConversation Summary:\n{summary}"
        messages.append({"role": "system", "content": system_content})
        
        # Add recent conversation
        for msg in recent_conversation[:-1]:
            role = "user" if msg["type"] == "user" else "assistant"
            messages.append({"role": role, "content": msg["content"]})
        
        # Current query
        messages.append({"role": "user", "content": query})
        
        return await self._truncate_to_context_window(messages)
    
    async def _entity_tracking_memory(
        self, 
        query: str, 
        context: str, 
        conversation: List[dict],
        config: dict
    ) -> List[dict]:
        """Entity tracking memory: maintain important entities and topics"""
        messages = []
        
        # Extract entities and topics from conversation
        entities = await self._extract_entities_from_conversation(conversation)
        topics = await self._extract_topics_from_conversation(conversation)
        
        # Build enhanced system message
        system_content = self._build_system_message(context, config)
        
        if entities:
            system_content += f"\n\nKey entities in this conversation: {', '.join(entities)}"
        if topics:
            system_content += f"\nMain topics discussed: {', '.join(topics)}"
        
        messages.append({"role": "system", "content": system_content})
        
        # Add recent conversation (last 3 exchanges)
        if conversation:
            recent = conversation[-6:]
            for msg in recent[:-1]:
                role = "user" if msg["type"] == "user" else "assistant"
                messages.append({"role": role, "content": msg["content"]})
        
        # Current query
        messages.append({"role": "user", "content": query})
        
        return await self._truncate_to_context_window(messages)
    
    async def _adaptive_memory(
        self, 
        query: str, 
        context: str, 
        conversation: List[dict], 
        base_memory_length: int,
        config: dict
    ) -> List[dict]:
        """Adaptive memory: adjust strategy based on conversation characteristics"""
        
        # Analyze conversation to choose best strategy
        if not conversation:
            # No history, use simple approach
            return await self._sliding_window_memory(query, context, conversation, base_memory_length, config)
        
        conversation_length = len(conversation)
        
        # Check for entity-heavy conversation
        entities = await self._extract_entities_from_conversation(conversation)
        if len(entities) > 5:
            msg.info("Using entity tracking memory for entity-heavy conversation")
            return await self._entity_tracking_memory(query, context, conversation, config)
        
        # Check for long conversation
        if conversation_length > 12:
            msg.info("Using summarization memory for long conversation")
            return await self._summarization_memory(query, context, conversation, config)
        
        # Default to sliding window
        msg.info("Using sliding window memory")
        return await self._sliding_window_memory(query, context, conversation, base_memory_length, config)
    
    def _build_system_message(self, context: str, config: dict) -> str:
        """Build system message with context"""
        base_message = config["System Message"].value
        
        if context.strip():
            return f"{base_message}\n\nRelevant Context:\n{context}"
        else:
            return f"{base_message}\n\nNote: No specific context was retrieved for this query."
    
    def _add_formatting_instructions(
        self, 
        messages: List[dict], 
        response_format: str, 
        citations_enabled: str
    ) -> List[dict]:
        """Add response formatting instructions"""
        
        format_instructions = {
            "conversational": "Respond in a natural, conversational tone.",
            "academic": "Respond in a formal, academic style with proper structure.",
            "bullet_points": "Structure your response using bullet points and clear sections.",
            "structured": "Use clear headings and well-organized sections in your response."
        }
        
        citation_instructions = {
            "enabled": "Include citations to source documents when using specific information.",
            "disabled": "Do not include explicit citations in your response.",
            "footnotes": "Include citations as footnotes at the end of your response."
        }
        
        # Add instructions to system message
        system_msg = messages[0]["content"]
        system_msg += f"\n\nResponse Format: {format_instructions.get(response_format, '')}"
        system_msg += f"\nCitation Style: {citation_instructions.get(citations_enabled, '')}"
        
        messages[0]["content"] = system_msg
        return messages
    
    async def _truncate_to_context_window(self, messages: List[dict]) -> List[dict]:
        """Truncate messages to fit within context window"""
        
        # Estimate token count (rough approximation)
        total_tokens = sum(len(msg["content"]) // 4 for msg in messages)
        
        if total_tokens <= self.context_window:
            return messages
        
        msg.info(f"Truncating conversation: {total_tokens} tokens > {self.context_window} limit")
        
        # Keep system message and current query, truncate middle
        system_msg = messages[0]
        current_query = messages[-1]
        middle_messages = messages[1:-1]
        
        # Calculate available tokens for middle messages
        reserved_tokens = len(system_msg["content"]) // 4 + len(current_query["content"]) // 4 + 200
        available_tokens = self.context_window - reserved_tokens
        
        # Truncate middle messages from the beginning
        truncated_middle = []
        current_tokens = 0
        
        for msg in reversed(middle_messages):  # Start from most recent
            msg_tokens = len(msg["content"]) // 4
            if current_tokens + msg_tokens <= available_tokens:
                truncated_middle.insert(0, msg)  # Insert at beginning to maintain order
                current_tokens += msg_tokens
            else:
                break
        
        return [system_msg] + truncated_middle + [current_query]
    
    async def _generate_with_model(
        self, 
        model: str, 
        messages: List[dict], 
        max_tokens: int, 
        temperature: float
    ) -> AsyncGenerator[dict, None]:
        """Generate response using specified model"""
        
        if model.startswith("gpt"):
            async for chunk in self._generate_openai(messages, model, max_tokens, temperature):
                yield chunk
        elif model.startswith("claude"):
            async for chunk in self._generate_anthropic(messages, model, max_tokens, temperature):
                yield chunk
        elif model.startswith("local"):
            async for chunk in self._generate_local(messages, model, max_tokens, temperature):
                yield chunk
        else:
            raise ValueError(f"Unsupported model: {model}")
    
    async def _generate_openai(
        self, 
        messages: List[dict], 
        model: str, 
        max_tokens: int, 
        temperature: float
    ) -> AsyncGenerator[dict, None]:
        """Generate response using OpenAI API"""
        try:
            import aiohttp
            import json
            
            api_key = get_token("OPENAI_API_KEY") or get_token("CUSTOM_API_KEY")
            if not api_key:
                raise ValueError("OpenAI API key not found")
            
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "stream": True
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers=headers,
                    json=payload
                ) as response:
                    
                    if response.status != 200:
                        error_text = await response.text()
                        raise Exception(f"OpenAI API error: {error_text}")
                    
                    async for line in response.content:
                        line = line.decode('utf-8').strip()
                        if line.startswith('data: '):
                            data = line[6:]
                            if data == '[DONE]':
                                yield {"message": "", "finished": True}
                                break
                            
                            try:
                                json_data = json.loads(data)
                                if 'choices' in json_data and json_data['choices']:
                                    delta = json_data['choices'][0].get('delta', {})
                                    content = delta.get('content', '')
                                    if content:
                                        yield {"message": content, "finished": False}
                            except json.JSONDecodeError:
                                continue
                                
        except Exception as e:
            msg.fail(f"OpenAI generation failed: {str(e)}")
            yield {"message": f"Error: {str(e)}", "finished": True}
    
    async def _generate_anthropic(
        self, 
        messages: List[dict], 
        model: str, 
        max_tokens: int, 
        temperature: float
    ) -> AsyncGenerator[dict, None]:
        """Generate response using Anthropic API (placeholder)"""
        # This would implement Anthropic's Claude API
        yield {"message": "Anthropic API integration not implemented in this example.", "finished": True}
    
    async def _generate_local(
        self, 
        messages: List[dict], 
        model: str, 
        max_tokens: int, 
        temperature: float
    ) -> AsyncGenerator[dict, None]:
        """Generate response using local model (placeholder)"""
        # This would implement local model inference (e.g., Ollama)
        yield {"message": "Local model integration not implemented in this example.", "finished": True}
    
    async def _generate_conversation_summary(self, conversation: List[dict]) -> str:
        """Generate a summary of conversation history"""
        # Extract key points from conversation
        user_queries = [msg["content"] for msg in conversation if msg["type"] == "user"]
        assistant_responses = [msg["content"] for msg in conversation if msg["type"] == "assistant"]
        
        # Simple extractive summary (in practice, use a summarization model)
        key_topics = []
        for query in user_queries[:3]:  # Take first 3 queries
            # Extract key terms (simplified)
            words = query.split()
            key_words = [w for w in words if len(w) > 4 and w.isalpha()]
            key_topics.extend(key_words[:2])
        
        if key_topics:
            return f"Previous discussion covered topics including: {', '.join(set(key_topics))}."
        else:
            return "Previous general discussion about various topics."
    
    async def _extract_entities_from_conversation(self, conversation: List[dict]) -> List[str]:
        """Extract named entities from conversation"""
        entities = set()
        
        for msg in conversation:
            content = msg["content"]
            # Simple entity extraction (in practice, use NER)
            words = content.split()
            # Look for capitalized words that might be entities
            for word in words:
                if (word[0].isupper() and 
                    len(word) > 2 and 
                    word.isalpha() and 
                    word not in ["The", "This", "That", "What", "How", "When", "Where"]):
                    entities.add(word)
        
        return list(entities)[:8]  # Return top 8 entities
    
    async def _extract_topics_from_conversation(self, conversation: List[dict]) -> List[str]:
        """Extract main topics from conversation"""
        # Simple keyword-based topic extraction
        topic_keywords = {
            "technology": ["software", "programming", "computer", "algorithm", "data"],
            "science": ["research", "experiment", "theory", "analysis", "study"],
            "business": ["company", "market", "strategy", "revenue", "customer"],
            "education": ["learning", "teaching", "student", "course", "knowledge"]
        }
        
        topics = set()
        all_text = " ".join([msg["content"].lower() for msg in conversation])
        
        for topic, keywords in topic_keywords.items():
            if any(keyword in all_text for keyword in keywords):
                topics.add(topic)
        
        return list(topics)


# Example usage and testing
if __name__ == "__main__":
    import asyncio
    
    async def test_custom_generator():
        """Test the custom generator"""
        generator = ExampleCustomGenerator()
        
        print("Custom Generator Configuration:")
        for key, config in generator.config.items():
            print(f"  {key}: {config.value} ({config.type})")
        
        print(f"\nGenerator: {generator.name}")
        print(f"Description: {generator.description}")
        print(f"Context window: {generator.context_window}")
        print(f"Required environment: {generator.requires_env}")
        print(f"Required libraries: {generator.requires_library}")
        
        # Test memory strategies
        conversation = [
            {"type": "user", "content": "What is machine learning?"},
            {"type": "assistant", "content": "Machine learning is a subset of AI that enables computers to learn and improve from experience without being explicitly programmed."},
            {"type": "user", "content": "Can you explain neural networks?"},
            {"type": "assistant", "content": "Neural networks are computing systems inspired by biological neural networks..."},
        ]
        
        config = {
            "Memory Strategy": type('obj', (object,), {'value': 'sliding_window'}),
            "Memory Length": type('obj', (object,), {'value': 3}),
            "System Message": type('obj', (object,), {'value': generator.config["System Message"].value}),
        }
        
        # Test message preparation
        messages = await generator._prepare_messages_with_memory(
            "How does backpropagation work?",
            "Backpropagation is an algorithm used to train neural networks...",
            conversation,
            "sliding_window",
            3,
            config
        )
        
        print(f"\nPrepared {len(messages)} messages for generation:")
        for i, msg in enumerate(messages):
            content_preview = msg["content"][:100] + "..." if len(msg["content"]) > 100 else msg["content"]
            print(f"  {i+1}. {msg['role']}: {content_preview}")
    
    # Run test
    asyncio.run(test_custom_generator())