"""
PDF ingestion tool for uploading and processing PDF documents.
"""

import logging
import os
from typing import Dict, Any
from rag_mcp_server.services.pdf_processor import PDFProcessor
from rag_mcp_server.services.gemini_service import GeminiService
from rag_mcp_server.services.pinecone_service import PineconeService

logger = logging.getLogger(__name__)

# Initialize services
pdf_processor = PDFProcessor()
gemini_service = GeminiService()
pinecone_service = PineconeService()


async def upload_pdf(file_path: str, document_id: str, chunking_strategy: str = "sliding_window") -> str:
    """
    Upload and process PDF document into vector database.
    
    Args:
        file_path: Path to the PDF file to process
        document_id: Unique identifier for the document
        chunking_strategy: Text chunking strategy ('sliding_window', 'sentence_aware', 'paragraph_aware')
    
    Returns:
        Status message indicating success or failure
    """
    try:
        logger.info(f"Starting PDF ingestion for document: {document_id}")
        
        # Validate PDF file
        if not await pdf_processor.validate_pdf_file(file_path):
            return f"❌ Error: Invalid PDF file at {file_path}"
        
        # Process PDF into chunks
        logger.info(f"Processing PDF into chunks using {chunking_strategy} strategy")
        chunks = await pdf_processor.process_pdf_to_chunks(
            file_path, document_id, chunking_strategy
        )
        
        if not chunks:
            return f"❌ Error: No content could be extracted from PDF {file_path}"
        
        logger.info(f"Generated {len(chunks)} chunks from PDF")
        
        # Generate embeddings for chunks
        logger.info("Generating embeddings for document chunks")

        # Limit text length for embedding generation (Gemini API has ~36KB limit)
        max_embedding_text_length = 30000  # Conservative limit for Gemini API
        chunk_texts = []

        for chunk in chunks:
            chunk_text = chunk["text"]
            if len(chunk_text.encode('utf-8')) > max_embedding_text_length:
                # Truncate text for embedding generation
                truncated_text = chunk_text[:max_embedding_text_length]
                logger.warning(f"Truncated chunk {chunk['id']} for embedding generation: {len(chunk_text)} -> {len(truncated_text)} characters")
                chunk_texts.append(truncated_text)
            else:
                chunk_texts.append(chunk_text)

        try:
            embeddings = await gemini_service.generate_embeddings(chunk_texts)
        except Exception as e:
            logger.error(f"Embedding generation failed: {str(e)}")
            return f"❌ Error: Failed to generate embeddings. {str(e)}"

        if not embeddings:
            return f"❌ Error: No embeddings generated for document chunks"

        if len(embeddings) != len(chunks):
            return f"❌ Error: Embedding generation failed. Expected {len(chunks)} embeddings, got {len(embeddings)}"
        
        # Prepare vectors for Pinecone
        vectors = []
        for chunk, embedding in zip(chunks, embeddings):
            # Include the actual text content in metadata for retrieval
            metadata_with_text = chunk["metadata"].copy()

            # Pinecone has metadata size limits (~40KB), so truncate text if needed
            chunk_text = chunk["text"]
            max_text_length = 30000  # Conservative limit to leave room for other metadata

            if len(chunk_text) > max_text_length:
                chunk_text = chunk_text[:max_text_length] + "... [truncated]"
                logger.warning(f"Truncated text for chunk {chunk['id']} from {len(chunk['text'])} to {len(chunk_text)} characters")

            metadata_with_text["text"] = chunk_text

            vectors.append({
                "id": chunk["id"],
                "values": embedding,
                "metadata": metadata_with_text
            })
        
        # Validate vectors before upsert
        logger.info(f"Validating {len(vectors)} vectors before Pinecone storage")
        for i, vector in enumerate(vectors):
            if not vector.get('id'):
                return f"❌ Error: Vector {i} missing ID"
            if not vector.get('values'):
                return f"❌ Error: Vector {i} missing embedding values"
            if not isinstance(vector.get('metadata'), dict):
                return f"❌ Error: Vector {i} metadata is not a dictionary"

        # Upsert to Pinecone
        logger.info("Storing vectors in Pinecone database")
        try:
            success = await pinecone_service.upsert_vectors(vectors)

            if not success:
                return f"❌ Error: Failed to store vectors in Pinecone database. Check server logs for detailed error information."
        except Exception as e:
            logger.error(f"Exception during Pinecone upsert: {str(e)}")
            return f"❌ Error: Pinecone storage failed with exception: {str(e)}"
        
        # Get file info for summary
        file_name = os.path.basename(file_path)
        file_size = os.path.getsize(file_path)
        
        success_message = f"""✅ Successfully processed PDF document!

📄 Document Details:
   • Document ID: {document_id}
   • File: {file_name}
   • Size: {file_size:,} bytes
   • Chunks Generated: {len(chunks)}
   • Chunking Strategy: {chunking_strategy}
   • Embeddings Created: {len(embeddings)}
   
🔍 The document is now searchable in the RAG system.
You can search for content using the semantic_search or retrieve_documents tools."""
        
        logger.info(f"PDF ingestion completed successfully for document: {document_id}")
        return success_message
        
    except FileNotFoundError:
        error_msg = f"❌ Error: PDF file not found at {file_path}"
        logger.error(error_msg)
        return error_msg
    except Exception as e:
        error_msg = f"❌ Error processing PDF: {str(e)}"
        logger.error(f"PDF ingestion failed for {document_id}: {str(e)}")
        return error_msg


async def get_pdf_info(file_path: str) -> str:
    """
    Get information about a PDF file without processing it.
    
    Args:
        file_path: Path to the PDF file
    
    Returns:
        PDF information summary
    """
    try:
        logger.info(f"Getting PDF info for: {file_path}")
        
        pdf_info = await pdf_processor.get_pdf_info(file_path)
        
        if "error" in pdf_info:
            return f"❌ Error: {pdf_info['error']}"
        
        info_message = f"""📄 PDF Information:

📁 File Details:
   • Name: {pdf_info.get('file_name', 'Unknown')}
   • Size: {pdf_info.get('file_size', 0):,} bytes
   • Pages: {pdf_info.get('page_count', 0)}

📋 Metadata:
   • Title: {pdf_info.get('title', 'Not specified')}
   • Author: {pdf_info.get('author', 'Not specified')}
   • Subject: {pdf_info.get('subject', 'Not specified')}
   • Creator: {pdf_info.get('creator', 'Not specified')}

📊 Processing Estimate:
   • Estimated Chunks: {pdf_info.get('estimated_chunks', 'Unknown')}
   • Text Sample: {pdf_info.get('text_sample', 'No text extracted')[:200]}...

✅ File is valid and ready for processing."""
        
        return info_message
        
    except Exception as e:
        error_msg = f"❌ Error getting PDF info: {str(e)}"
        logger.error(error_msg)
        return error_msg


async def list_processed_documents() -> str:
    """
    List documents that have been processed and stored in the vector database.
    
    Returns:
        List of processed documents
    """
    try:
        logger.info("Retrieving list of processed documents")
        
        # Get index statistics
        stats = await pinecone_service.get_index_stats()
        
        if not stats:
            return "❌ Error: Could not retrieve index statistics"
        
        total_vectors = stats.get('total_vector_count', 0)
        
        if total_vectors == 0:
            return """📋 Document Database Status:

🔍 No documents have been processed yet.

To get started:
1. Use the upload_pdf tool to process your first PDF document
2. Documents will appear here once they're successfully ingested"""
        
        # Note: This is a simplified implementation
        # In a production system, you'd want to maintain a separate index of documents
        # or use Pinecone's metadata filtering to get unique document IDs
        
        status_message = f"""📋 Document Database Status:

📊 Statistics:
   • Total Chunks Stored: {total_vectors:,}
   • Index Name: {pinecone_service.index_name}
   
💡 Note: Use semantic_search or retrieve_documents to query the stored content.
   
🔧 For detailed document management, consider implementing a document metadata store."""
        
        return status_message
        
    except Exception as e:
        error_msg = f"❌ Error listing documents: {str(e)}"
        logger.error(error_msg)
        return error_msg
