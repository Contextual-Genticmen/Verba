#!/usr/bin/env python3
"""
MCP RAG Server CLI entry point.

Provides command-line interface for running the Verba MCP RAG server.
"""

import argparse
import asyncio
import json
import os
import sys
from typing import Dict, Any

from .server import MCPRAGServer


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from JSON file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Configuration file not found: {config_path}")
        return {}
    except json.JSONDecodeError as e:
        print(f"Invalid JSON in configuration file: {e}")
        return {}


def create_default_config() -> Dict[str, Any]:
    """
    Create default configuration.
    
    Returns:
        Default configuration dictionary
    """
    return {
        "server": {
            "name": "verba-rag-server",
            "version": "1.0.0"
        },
        "weaviate": {
            "url": os.getenv("WEAVIATE_URL", "http://localhost:8080"),
            "key": os.getenv("WEAVIATE_API_KEY")
        },
        "mem0": {
            "vector_store_provider": "qdrant",
            "vector_store_config": {
                "collection_name": "verba_memories",
                "host": "localhost",
                "port": 6333
            },
            "llm_provider": "openai",
            "llm_config": {
                "model": "gpt-4o-mini",
                "temperature": 0.1
            },
            "embedder_provider": "openai",
            "embedder_config": {
                "model": "text-embedding-ada-002"
            }
        }
    }


async def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Verba MCP RAG Server - Expose Verba's RAG capabilities via Model Context Protocol"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        help="Path to configuration file (JSON format)"
    )
    
    parser.add_argument(
        "--create-config",
        type=str,
        help="Create a default configuration file at the specified path"
    )
    
    parser.add_argument(
        "--transport",
        type=str,
        choices=["stdio"],
        default="stdio",
        help="Transport protocol to use (default: stdio)"
    )
    
    parser.add_argument(
        "--name",
        type=str,
        default="verba-rag-server",
        help="Server name (default: verba-rag-server)"
    )
    
    parser.add_argument(
        "--version",
        type=str,
        default="1.0.0",
        help="Server version (default: 1.0.0)"
    )
    
    args = parser.parse_args()
    
    # Handle config creation
    if args.create_config:
        config = create_default_config()
        try:
            with open(args.create_config, 'w') as f:
                json.dump(config, f, indent=2)
            print(f"Created default configuration file: {args.create_config}")
            return
        except Exception as e:
            print(f"Error creating configuration file: {e}")
            sys.exit(1)
    
    # Load configuration
    if args.config:
        config = load_config(args.config)
    else:
        config = create_default_config()
    
    # Extract configuration components
    server_config = config.get("server", {})
    weaviate_config = config.get("weaviate", {})
    mem0_config = config.get("mem0", {})
    
    # Override with CLI arguments
    server_name = args.name or server_config.get("name", "verba-rag-server")
    server_version = args.version or server_config.get("version", "1.0.0")
    
    # Initialize and run server
    try:
        server = MCPRAGServer(
            name=server_name,
            version=server_version,
            weaviate_url=weaviate_config.get("url"),
            weaviate_key=weaviate_config.get("key"),
            mem0_config=mem0_config
        )
        
        print(f"Starting {server_name} v{server_version} with {args.transport} transport...")
        await server.run(transport_type=args.transport)
        
    except KeyboardInterrupt:
        print("\nServer stopped by user")
    except Exception as e:
        print(f"Error running server: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())