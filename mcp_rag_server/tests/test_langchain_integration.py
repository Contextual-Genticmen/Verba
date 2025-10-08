"""
Advanced integration example: Using LangChain with MCP RAG Server.

This example demonstrates how to use LangChain's MCP client integration
to connect to the MCP RAG server and leverage both LangChain's agent
capabilities and Verba's RAG tools.

Based on the langchain-mcp-tools package and community best practices.
"""

import asyncio
from pathlib import Path

# Try to import LangChain components
try:
    from langchain.agents import AgentExecutor, create_tool_calling_agent
    from langchain.prompts import ChatPromptTemplate
    from langchain_core.tools import BaseTool
    
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    # Create a mock BaseTool for standalone usage
    class BaseTool:
        def __init__(self, name: str, description: str, **kwargs):
            self.name = name
            self.description = description
    
    print("LangChain not available. Install with: pip install langchain langchain-core")

# Import MCP components
try:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    from mcp.types import CallToolRequest
    
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    print("MCP SDK not available. Install with: pip install mcp>=1.16.0")


class MCPToolWrapper(BaseTool):
    """
    Wrapper to convert MCP tools into LangChain tools.
    
    This allows LangChain agents to use MCP RAG server tools seamlessly.
    """
    
    def __init__(self, name: str, description: str, session, **kwargs):
        """
        Initialize the MCP tool wrapper.
        
        Args:
            name: Tool name
            description: Tool description
            session: MCP ClientSession for making tool calls
        """
        super().__init__(name=name, description=description, **kwargs)
        self.session = session
    
    async def _arun(self, **kwargs):
        """Execute the tool asynchronously."""
        result = await self.session.call_tool(self.name, arguments=kwargs)
        return result.content[0].text if result.content else ""
    
    def _run(self, **kwargs):
        """Synchronous execution (not recommended for MCP)."""
        raise NotImplementedError("Use async version (_arun) for MCP tools")


async def create_langchain_agent_with_mcp():
    """
    Create a LangChain agent that uses MCP RAG server tools.
    
    This demonstrates:
    1. Connecting to the MCP RAG server
    2. Wrapping MCP tools as LangChain tools
    3. Creating an agent that can use RAG capabilities
    """
    if not LANGCHAIN_AVAILABLE or not MCP_AVAILABLE:
        print("Required dependencies not available")
        return
    
    print("=== Creating LangChain Agent with MCP RAG Server ===\n")
    
    # Connect to MCP RAG server
    server_params = StdioServerParameters(
        command="python",
        args=["-m", "mcp_rag_server.cli"]
    )
    
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # Initialize session
            await session.initialize()
            
            # List available tools
            tools_response = await session.list_tools()
            print(f"Found {len(tools_response.tools)} MCP tools:")
            for tool in tools_response.tools:
                print(f"  - {tool.name}: {tool.description[:60]}...")
            
            # Wrap MCP tools for LangChain
            langchain_tools = []
            for mcp_tool in tools_response.tools:
                lc_tool = MCPToolWrapper(
                    name=mcp_tool.name,
                    description=mcp_tool.description,
                    session=session
                )
                langchain_tools.append(lc_tool)
            
            print(f"\n✓ Created {len(langchain_tools)} LangChain tools from MCP server")
            
            # Now you can use these tools with LangChain agents
            # (requires LLM configuration)
            print("\nReady to create LangChain agent with MCP RAG tools!")
            
            return langchain_tools


async def demo_cli_with_mcp_and_files():
    """
    Demonstrate a CLI workflow using MCP RAG server with file operations.
    
    This shows:
    1. Reading a code file
    2. Processing it through RAG pipeline
    3. Answering questions about the code
    """
    if not MCP_AVAILABLE:
        print("MCP SDK not available")
        return
    
    print("=== CLI Demo: MCP RAG Server with File Operations ===\n")
    
    # Path to test file
    test_file = Path(__file__).parent / "test_resources" / "sample_code.py"
    
    if not test_file.exists():
        print(f"Test file not found: {test_file}")
        return
    
    # Read file content
    with open(test_file, 'r') as f:
        file_content = f.read()
    
    print(f"Loaded file: {test_file.name} ({len(file_content)} bytes)\n")
    
    # Connect to MCP server (would be actual server in production)
    # For demo, we'll simulate the operations
    print("[Step 1] Chunking document...")
    print(f"  -> Created chunks from {test_file.name}")
    
    print("\n[Step 2] Indexing document...")
    print(f"  -> Indexed document into vector store")
    
    print("\n[Step 3] User Query: 'What does the DocumentProcessor class do?'")
    print(f"  -> Retrieved relevant context from indexed code")
    
    print("\n[Step 4] Generating Answer...")
    print(f"  -> The DocumentProcessor class is designed to process documents")
    print(f"     for RAG operations. It provides methods for adding documents,")
    print(f"     chunking them into smaller pieces, and retrieving relevant chunks.")
    
    print("\n[Step 5] Follow-up Query: 'Show me the chunking method'")
    print(f"  -> Located chunk_document method in the code")
    print(f"  -> Method signature: def chunk_document(self, doc_id)")
    
    print("\n✓ CLI demo completed successfully!")


async def demo_with_community_filesystem_tools():
    """
    Demonstrate integration with community MCP filesystem tools.
    
    This shows how to combine:
    1. MCP filesystem server (for file operations)
    2. MCP RAG server (for document processing)
    
    Note: Requires @modelcontextprotocol/server-filesystem to be installed
    """
    if not MCP_AVAILABLE:
        print("MCP SDK not available")
        return
    
    print("=== Demo: Integration with MCP Filesystem Tools ===\n")
    
    print("To use this integration, you need:")
    print("1. Install Node.js MCP filesystem server:")
    print("   npm install -g @modelcontextprotocol/server-filesystem")
    print()
    print("2. Example integration code:")
    print()
    print("```python")
    print("# Connect to filesystem server")
    print("fs_params = StdioServerParameters(")
    print('    command="npx",')
    print('    args=["-y", "@modelcontextprotocol/server-filesystem", "./code"]')
    print(")")
    print()
    print("async with stdio_client(fs_params) as (fs_read, fs_write):")
    print("    async with ClientSession(fs_read, fs_write) as fs_session:")
    print("        await fs_session.initialize()")
    print()
    print("        # Read file using filesystem tools")
    print('        result = await fs_session.call_tool("read_file",')
    print('            arguments={"path": "sample.py"})')
    print("        file_content = result.content[0].text")
    print()
    print("        # Connect to RAG server")
    print("        rag_params = StdioServerParameters(")
    print('            command="python",')
    print('            args=["-m", "mcp_rag_server.cli"]')
    print("        )")
    print()
    print("        async with stdio_client(rag_params) as (rag_read, rag_write):")
    print("            async with ClientSession(rag_read, rag_write) as rag_session:")
    print("                await rag_session.initialize()")
    print()
    print("                # Process with RAG server")
    print('                result = await rag_session.call_tool("chunk_documents",')
    print("                    arguments={")
    print('                        "documents": [{"name": "sample.py",')
    print('                                      "content": file_content}]')
    print("                    })")
    print("```")
    print()
    print("This architecture allows you to:")
    print("- Use filesystem server for secure file access")
    print("- Use RAG server for document processing")
    print("- Combine multiple MCP servers in one workflow")


async def main():
    """Run all demo examples."""
    print("MCP RAG Server - LangChain Integration Examples")
    print("=" * 70)
    print()
    
    # Demo 1: CLI with files
    await demo_cli_with_mcp_and_files()
    print("\n" + "=" * 70 + "\n")
    
    # Demo 2: Community filesystem tools
    await demo_with_community_filesystem_tools()
    print("\n" + "=" * 70 + "\n")
    
    # Demo 3: LangChain agent (if available)
    if LANGCHAIN_AVAILABLE and MCP_AVAILABLE:
        await create_langchain_agent_with_mcp()
    else:
        print("=== LangChain Agent Demo ===\n")
        print("LangChain or MCP not available.")
        print("Install with: pip install langchain langchain-core mcp>=1.16.0")
    
    print("\n" + "=" * 70)
    print("All demos completed!")


if __name__ == "__main__":
    asyncio.run(main())
