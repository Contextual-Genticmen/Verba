"""
Configuration management for MCP RAG Server.

Handles loading and validation of server configuration from various sources.
"""

import os
import json
from typing import Dict, Any, Optional
from pathlib import Path


class MCPRAGConfig:
    """
    Configuration manager for MCP RAG Server.
    
    Handles configuration loading from:
    - Environment variables
    - JSON configuration files
    - Default values
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration manager.
        
        Args:
            config_path: Path to JSON configuration file
        """
        self.config_path = config_path
        self._config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """
        Load configuration from various sources.
        
        Returns:
            Merged configuration dictionary
        """
        # Start with defaults
        config = self._get_default_config()
        
        # Load from file if specified
        if self.config_path:
            file_config = self._load_from_file(self.config_path)
            config = self._merge_configs(config, file_config)
        
        # Override with environment variables
        env_config = self._load_from_env()
        config = self._merge_configs(config, env_config)
        
        return config
    
    def _get_default_config(self) -> Dict[str, Any]:
        """
        Get default configuration values.
        
        Returns:
            Default configuration dictionary
        """
        return {
            "server": {
                "name": "verba-rag-server",
                "version": "1.0.0",
                "log_level": "INFO"
            },
            "weaviate": {
                "url": "http://localhost:8080",
                "key": None,
                "timeout": 30
            },
            "mem0": {
                "vector_store": {
                    "provider": "qdrant",
                    "config": {
                        "collection_name": "verba_memories",
                        "host": "localhost",
                        "port": 6333
                    }
                },
                "llm": {
                    "provider": "openai",
                    "config": {
                        "model": "gpt-4o-mini",
                        "temperature": 0.1,
                        "max_tokens": 1000
                    }
                },
                "embedder": {
                    "provider": "openai",
                    "config": {
                        "model": "text-embedding-ada-002"
                    }
                }
            },
            "rag": {
                "default_chunker": "TokenChunker",
                "default_embedder": "OpenAIEmbedder", 
                "default_retriever": "BasicRetriever",
                "default_generator": "OpenAIGenerator",
                "chunk_size": 500,
                "chunk_overlap": 50,
                "max_results": 10
            }
        }
    
    def _load_from_file(self, config_path: str) -> Dict[str, Any]:
        """
        Load configuration from JSON file.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Configuration dictionary from file
        """
        try:
            path = Path(config_path)
            if path.exists():
                with open(path, 'r') as f:
                    return json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            print(f"Warning: Could not load config file {config_path}: {e}")
        return {}
    
    def _load_from_env(self) -> Dict[str, Any]:
        """
        Load configuration from environment variables.
        
        Returns:
            Configuration dictionary from environment
        """
        config = {}
        
        # Server configuration
        if os.getenv("MCP_RAG_SERVER_NAME"):
            config.setdefault("server", {})["name"] = os.getenv("MCP_RAG_SERVER_NAME")
        
        if os.getenv("MCP_RAG_LOG_LEVEL"):
            config.setdefault("server", {})["log_level"] = os.getenv("MCP_RAG_LOG_LEVEL")
        
        # Weaviate configuration
        if os.getenv("WEAVIATE_URL"):
            config.setdefault("weaviate", {})["url"] = os.getenv("WEAVIATE_URL")
        
        if os.getenv("WEAVIATE_API_KEY"):
            config.setdefault("weaviate", {})["key"] = os.getenv("WEAVIATE_API_KEY")
        
        # OpenAI configuration (commonly used)
        if os.getenv("OPENAI_API_KEY"):
            config.setdefault("mem0", {}).setdefault("llm", {}).setdefault("config", {})["api_key"] = os.getenv("OPENAI_API_KEY")
            config.setdefault("mem0", {}).setdefault("embedder", {}).setdefault("config", {})["api_key"] = os.getenv("OPENAI_API_KEY")
        
        # Qdrant configuration (for mem0)
        if os.getenv("QDRANT_URL"):
            config.setdefault("mem0", {}).setdefault("vector_store", {}).setdefault("config", {})["url"] = os.getenv("QDRANT_URL")
        
        if os.getenv("QDRANT_API_KEY"):
            config.setdefault("mem0", {}).setdefault("vector_store", {}).setdefault("config", {})["api_key"] = os.getenv("QDRANT_API_KEY")
        
        return config
    
    def _merge_configs(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """
        Recursively merge configuration dictionaries.
        
        Args:
            base: Base configuration
            override: Override configuration
            
        Returns:
            Merged configuration
        """
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_configs(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.
        
        Args:
            key: Configuration key (e.g., "server.name")
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key.split(".")
        value = self._config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def get_server_config(self) -> Dict[str, Any]:
        """Get server configuration section."""
        return self._config.get("server", {})
    
    def get_weaviate_config(self) -> Dict[str, Any]:
        """Get Weaviate configuration section."""
        return self._config.get("weaviate", {})
    
    def get_mem0_config(self) -> Dict[str, Any]:
        """Get mem0 configuration section."""
        return self._config.get("mem0", {})
    
    def get_rag_config(self) -> Dict[str, Any]:
        """Get RAG configuration section."""
        return self._config.get("rag", {})
    
    def save_to_file(self, file_path: str) -> None:
        """
        Save current configuration to file.
        
        Args:
            file_path: Path to save configuration file
        """
        try:
            path = Path(file_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(path, 'w') as f:
                json.dump(self._config, f, indent=2)
        except OSError as e:
            raise RuntimeError(f"Could not save config to {file_path}: {e}")
    
    def validate(self) -> bool:
        """
        Validate configuration.
        
        Returns:
            True if configuration is valid
        """
        required_sections = ["server", "weaviate", "mem0", "rag"]
        
        for section in required_sections:
            if section not in self._config:
                print(f"Warning: Missing configuration section: {section}")
                return False
        
        # Validate server section
        server_config = self.get_server_config()
        if not server_config.get("name"):
            print("Error: Server name is required")
            return False
        
        return True
    
    @property
    def config(self) -> Dict[str, Any]:
        """Get the full configuration dictionary."""
        return self._config.copy()