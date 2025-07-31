"""
Configuration management for RAG MCP Server.
Handles environment variables and application settings.
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Config:
    """Configuration class for RAG MCP Server."""
    
    # Google AI Configuration
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "gemini-embedding-001")
    LLM_MODEL = os.getenv("LLM_MODEL", "gemini-2.0-flash")
    
    # Pinecone Configuration
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT", "us-west1-gcp")
    PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "rag-documents")
    
    # Server Configuration
    SERVER_NAME = os.getenv("MCP_SERVER_NAME", "rag-mcp-server")
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    
    # Text Processing Configuration
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
    MAX_CHUNKS_PER_DOCUMENT = int(os.getenv("MAX_CHUNKS_PER_DOCUMENT", "100"))
    
    # Development Configuration
    DEV_MODE = os.getenv("DEV_MODE", "false").lower() == "true"
    VERBOSE_LOGGING = os.getenv("VERBOSE_LOGGING", "false").lower() == "true"
    
    @classmethod
    def validate(cls) -> bool:
        """Validate that all required configuration is present."""
        required_vars = [
            ("GOOGLE_API_KEY", cls.GOOGLE_API_KEY),
            ("PINECONE_API_KEY", cls.PINECONE_API_KEY),
        ]
        
        missing_vars = []
        for var_name, var_value in required_vars:
            if not var_value:
                missing_vars.append(var_name)
        
        if missing_vars:
            raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")
        
        return True
    
    @classmethod
    def get_summary(cls) -> dict:
        """Get a summary of current configuration (without sensitive data)."""
        return {
            "server_name": cls.SERVER_NAME,
            "embedding_model": cls.EMBEDDING_MODEL,
            "llm_model": cls.LLM_MODEL,
            "pinecone_environment": cls.PINECONE_ENVIRONMENT,
            "pinecone_index": cls.PINECONE_INDEX_NAME,
            "chunk_size": cls.CHUNK_SIZE,
            "chunk_overlap": cls.CHUNK_OVERLAP,
            "max_chunks": cls.MAX_CHUNKS_PER_DOCUMENT,
            "dev_mode": cls.DEV_MODE,
            "log_level": cls.LOG_LEVEL
        }
