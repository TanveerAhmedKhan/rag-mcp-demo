"""
RAG MCP Server - Main Entry Point

A Model Context Protocol (MCP) server that provides Retrieval-Augmented Generation
capabilities for PDF documents using Google Gemini AI and Pinecone vector database.

Features:
- PDF document ingestion and processing
- Semantic search across document collections
- AI-powered document retrieval and question answering
- Integration with VSCode and other MCP-compatible clients

Author: RAG MCP Demo Project
Version: 1.0.0
"""

import logging
import sys
import asyncio
from mcp.server.fastmcp import FastMCP
from rag_mcp_server.config import Config
from rag_mcp_server.tools import pdf_ingestion, semantic_search, document_retrieval

# Configure logging
logging.basicConfig(
    level=getattr(logging, Config.LOG_LEVEL.upper()),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stderr)  # Use stderr for MCP compatibility
    ]
)

logger = logging.getLogger(__name__)

# Initialize FastMCP server
mcp = FastMCP(Config.SERVER_NAME)


@mcp.tool()
async def upload_pdf(file_path: str, document_id: str, chunking_strategy: str = "sliding_window") -> str:
    """
    Upload and process PDF document into vector database.
    
    Args:
        file_path: Path to the PDF file to process
        document_id: Unique identifier for the document
        chunking_strategy: Text chunking strategy ('sliding_window', 'sentence_aware', 'paragraph_aware')
    """
    return await pdf_ingestion.upload_pdf(file_path, document_id, chunking_strategy)


@mcp.tool()
async def get_pdf_info(file_path: str) -> str:
    """
    Get information about a PDF file without processing it.
    
    Args:
        file_path: Path to the PDF file
    """
    return await pdf_ingestion.get_pdf_info(file_path)


@mcp.tool()
async def list_processed_documents() -> str:
    """
    List documents that have been processed and stored in the vector database.
    """
    return await pdf_ingestion.list_processed_documents()


@mcp.tool()
async def search_documents(query: str, top_k: int = 5, min_score: float = 0.0, 
                         document_filter: str = None) -> str:
    """
    Search for relevant document chunks using semantic similarity.
    
    Args:
        query: Search query text
        top_k: Number of top results to return (default: 5, max: 20)
        min_score: Minimum similarity score threshold (0.0 to 1.0)
        document_filter: Optional document ID to filter results
    """
    return await semantic_search.search_documents(query, top_k, min_score, document_filter)


@mcp.tool()
async def search_by_keywords(keywords: str, top_k: int = 5) -> str:
    """
    Search documents using keyword-based approach combined with semantic search.
    
    Args:
        keywords: Comma-separated keywords to search for
        top_k: Number of top results to return
    """
    return await semantic_search.search_by_keywords(keywords, top_k)


@mcp.tool()
async def get_search_suggestions(partial_query: str) -> str:
    """
    Get search suggestions based on a partial query.
    
    Args:
        partial_query: Partial search query
    """
    return await semantic_search.get_search_suggestions(partial_query)


@mcp.tool()
async def retrieve_documents(query: str, generate_answer: bool = True, top_k: int = 3, 
                           include_sources: bool = True, document_filter: str = None) -> str:
    """
    Retrieve relevant document chunks and optionally generate an AI response.
    
    Args:
        query: Question or search query
        generate_answer: Whether to generate an AI response using retrieved context
        top_k: Number of document chunks to retrieve for context
        include_sources: Whether to include source information in the response
        document_filter: Optional document ID to filter results
    """
    return await document_retrieval.retrieve_documents(
        query, generate_answer, top_k, include_sources, document_filter
    )


@mcp.tool()
async def ask_question(question: str, document_id: str = None) -> str:
    """
    Ask a specific question and get an AI-generated answer based on document content.
    
    Args:
        question: The question to ask
        document_id: Optional specific document to search in
    """
    return await document_retrieval.ask_question(question, document_id)


@mcp.tool()
async def summarize_document(document_id: str, max_length: int = 500) -> str:
    """
    Generate a summary of a specific document.
    
    Args:
        document_id: ID of the document to summarize
        max_length: Maximum length of the summary
    """
    return await document_retrieval.summarize_document(document_id, max_length)


def main():
    """Main function to run the MCP server."""
    try:
        # Validate configuration
        logger.info("Starting RAG MCP Server...")
        logger.info(f"Server name: {Config.SERVER_NAME}")

        if not Config.validate():
            logger.error("Configuration validation failed")
            sys.exit(1)

        logger.info("Configuration validated successfully")

        # Log configuration summary (without sensitive data)
        config_summary = Config.get_summary()
        logger.info(f"Configuration: {config_summary}")

        # Run the MCP server
        logger.info("Starting MCP server with STDIO transport...")
        mcp.run(transport='stdio')

    except KeyboardInterrupt:
        logger.info("Server shutdown requested")
    except Exception as e:
        logger.error(f"Server error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
