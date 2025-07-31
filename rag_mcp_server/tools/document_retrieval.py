"""
Document retrieval tool for fetching relevant content and generating AI responses.
"""

import logging
from typing import List, Dict, Any, Optional
from rag_mcp_server.services.gemini_service import GeminiService
from rag_mcp_server.services.pinecone_service import PineconeService

logger = logging.getLogger(__name__)

# Initialize services
gemini_service = GeminiService()
pinecone_service = PineconeService()


async def retrieve_documents(query: str, generate_answer: bool = True, top_k: int = 3, 
                           include_sources: bool = True, document_filter: Optional[str] = None) -> str:
    """
    Retrieve relevant document chunks and optionally generate an AI response.
    
    Args:
        query: Question or search query
        generate_answer: Whether to generate an AI response using retrieved context
        top_k: Number of document chunks to retrieve for context
        include_sources: Whether to include source information in the response
        document_filter: Optional document ID to filter results
    
    Returns:
        AI-generated response with sources or retrieved context
    """
    try:
        logger.info(f"Retrieving documents for query: {query[:50]}...")
        
        # Validate inputs
        if not query.strip():
            return "❌ Error: Query cannot be empty"
        
        if top_k < 1 or top_k > 10:
            return "❌ Error: top_k must be between 1 and 10"
        
        # Generate embedding for the query
        query_vector = await gemini_service.generate_query_embedding(query)
        
        if not query_vector:
            return "❌ Error: Failed to generate embedding for the query"
        
        # Prepare filter if document_filter is provided
        filter_dict = None
        if document_filter:
            filter_dict = {"document_id": {"$eq": document_filter}}
        
        # Search in Pinecone
        search_results = await pinecone_service.query_vectors(
            query_vector=query_vector,
            top_k=top_k,
            filter_dict=filter_dict
        )
        
        matches = search_results.get('matches', [])
        
        if not matches:
            filter_msg = f" in document '{document_filter}'" if document_filter else ""
            return f"""❌ No relevant documents found to answer your query{filter_msg}.

💡 Suggestions:
   • Check if documents have been uploaded using the upload_pdf tool
   • Try rephrasing your question
   • Use broader search terms"""
        
        # Extract context and source information
        context_chunks = []
        source_info = []
        
        for i, match in enumerate(matches, 1):
            metadata = match.get('metadata', {})
            score = match.get('score', 0)
            
            document_id = metadata.get('document_id', 'Unknown')
            chunk_index = metadata.get('chunk_index', 0)
            file_name = metadata.get('file_name', 'Unknown')
            word_count = metadata.get('word_count', 0)
            
            # Extract the actual text content from metadata
            chunk_text = metadata.get('text', '')

            # If no text content is found, create a descriptive placeholder
            if not chunk_text:
                chunk_text = f"[Content from document '{document_id}', chunk {chunk_index + 1} of file '{file_name}' - {word_count} words, similarity score: {score:.4f}]"
            context_chunks.append(chunk_text)
            
            source_info.append({
                "document_id": document_id,
                "file_name": file_name,
                "chunk_index": chunk_index + 1,
                "similarity_score": score,
                "word_count": word_count
            })
        
        # Combine context
        combined_context = "\n\n".join(context_chunks)
        
        if generate_answer:
            # Generate AI response using retrieved context
            logger.info("Generating AI response using retrieved context")
            
            ai_response = await gemini_service.generate_response(
                prompt=query,
                context=combined_context,
                max_tokens=1000
            )
            
            if not ai_response:
                return "❌ Error: Failed to generate AI response"
            
            # Format the response
            result = f"""🤖 AI Response:

{ai_response}"""
            
            if include_sources:
                sources_text = "\n".join([
                    f"📄 {i}. {source['file_name']} (Document: {source['document_id']})"
                    f"\n   • Chunk: {source['chunk_index']}"
                    f"\n   • Similarity: {source['similarity_score']:.4f}"
                    f"\n   • Words: {source['word_count']}"
                    for i, source in enumerate(source_info, 1)
                ])
                
                result += f"""

📚 Sources:
{sources_text}

💡 The response above is based on {len(matches)} relevant document chunks."""
            
            return result
            
        else:
            # Return just the retrieved context without AI generation
            sources_text = "\n".join([
                f"📄 Source {i}: {source['file_name']} (Document: {source['document_id']})"
                f"\n   • Chunk: {source['chunk_index']}"
                f"\n   • Similarity: {source['similarity_score']:.4f}"
                f"\n   • Words: {source['word_count']}"
                for i, source in enumerate(source_info, 1)
            ])
            
            result = f"""📋 Retrieved Context for: "{query}"

🔍 Found {len(matches)} relevant chunks:

{combined_context}

📚 Sources:
{sources_text}

💡 Use generate_answer=true to get an AI-generated response based on this context."""
            
            return result
            
    except Exception as e:
        error_msg = f"❌ Error retrieving documents: {str(e)}"
        logger.error(f"Document retrieval failed for query '{query}': {str(e)}")
        return error_msg


async def ask_question(question: str, document_id: Optional[str] = None) -> str:
    """
    Ask a specific question and get an AI-generated answer based on document content.
    
    Args:
        question: The question to ask
        document_id: Optional specific document to search in
    
    Returns:
        AI-generated answer with sources
    """
    try:
        logger.info(f"Processing question: {question[:50]}...")
        
        if not question.strip():
            return "❌ Error: Question cannot be empty"
        
        # Use retrieve_documents with answer generation enabled
        result = await retrieve_documents(
            query=question,
            generate_answer=True,
            top_k=5,  # Use more context for questions
            include_sources=True,
            document_filter=document_id
        )
        
        return result
        
    except Exception as e:
        error_msg = f"❌ Error processing question: {str(e)}"
        logger.error(f"Question processing failed: {str(e)}")
        return error_msg


async def summarize_document(document_id: str, max_length: int = 500) -> str:
    """
    Generate a summary of a specific document.
    
    Args:
        document_id: ID of the document to summarize
        max_length: Maximum length of the summary
    
    Returns:
        Document summary
    """
    try:
        logger.info(f"Generating summary for document: {document_id}")
        
        if not document_id.strip():
            return "❌ Error: Document ID cannot be empty"
        
        # Retrieve multiple chunks from the specific document
        # Use a broad query to get representative content
        broad_query = f"overview summary content main points key information from {document_id}"
        
        result = await retrieve_documents(
            query=broad_query,
            generate_answer=False,  # Get raw content first
            top_k=8,  # Get more chunks for better summary
            include_sources=False,
            document_filter=document_id
        )
        
        if "❌" in result:
            return f"❌ Error: Could not find document '{document_id}' or no content available"
        
        # Extract the context from the result
        # This is a simplified approach - in practice you'd parse the result better
        context_start = result.find("📋 Retrieved Context")
        if context_start == -1:
            return "❌ Error: Could not extract content for summarization"
        
        context = result[context_start:]
        
        # Generate summary using Gemini
        summary = await gemini_service.summarize_text(context, max_length)
        
        if not summary:
            return "❌ Error: Failed to generate document summary"
        
        summary_result = f"""📄 Document Summary: {document_id}

{summary}

📊 Summary Details:
   • Target Length: {max_length} characters
   • Actual Length: {len(summary)} characters
   • Based on multiple document chunks

💡 Use ask_question to get specific information from this document."""
        
        return summary_result
        
    except Exception as e:
        error_msg = f"❌ Error generating document summary: {str(e)}"
        logger.error(f"Document summarization failed for '{document_id}': {str(e)}")
        return error_msg
