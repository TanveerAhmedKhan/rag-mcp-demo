"""
Semantic search tool for querying documents using vector similarity.
"""

import logging
from typing import Dict, Any, Optional
from rag_mcp_server.services.gemini_service import GeminiService
from rag_mcp_server.services.pinecone_service import PineconeService

logger = logging.getLogger(__name__)

# Initialize services
gemini_service = GeminiService()
pinecone_service = PineconeService()


async def search_documents(query: str, top_k: int = 5, min_score: float = 0.0, 
                         document_filter: Optional[str] = None) -> str:
    """
    Search for relevant document chunks using semantic similarity.
    
    Args:
        query: Search query text
        top_k: Number of top results to return (default: 5, max: 20)
        min_score: Minimum similarity score threshold (0.0 to 1.0)
        document_filter: Optional document ID to filter results
    
    Returns:
        Formatted search results
    """
    try:
        logger.info(f"Performing semantic search for query: {query[:50]}...")
        
        # Validate inputs
        if not query.strip():
            return "❌ Error: Search query cannot be empty"
        
        if top_k < 1 or top_k > 20:
            return "❌ Error: top_k must be between 1 and 20"
        
        if min_score < 0.0 or min_score > 1.0:
            return "❌ Error: min_score must be between 0.0 and 1.0"
        
        # Generate embedding for the query
        query_vector = await gemini_service.generate_query_embedding(query)
        
        if not query_vector:
            return "❌ Error: Failed to generate embedding for the search query"
        
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
        
        # Process results
        matches = search_results.get('matches', [])
        
        if not matches:
            filter_msg = f" in document '{document_filter}'" if document_filter else ""
            return f"""🔍 Search Results:

❌ No relevant documents found for your query{filter_msg}.

💡 Suggestions:
   • Try different keywords or phrases
   • Check if documents have been uploaded using the upload_pdf tool
   • Use broader search terms"""
        
        # Filter by minimum score
        filtered_matches = [match for match in matches if match.get('score', 0) >= min_score]
        
        if not filtered_matches:
            return f"""🔍 Search Results:

❌ No results found above the minimum score threshold of {min_score:.2f}.

📊 Found {len(matches)} results with lower scores. Consider lowering the min_score parameter."""
        
        # Format results
        formatted_results = []
        for i, match in enumerate(filtered_matches, 1):
            score = match.get('score', 0)
            metadata = match.get('metadata', {})
            
            document_id = metadata.get('document_id', 'Unknown')
            chunk_index = metadata.get('chunk_index', 0)
            file_name = metadata.get('file_name', 'Unknown')
            word_count = metadata.get('word_count', 0)
            
            # Extract the actual text content from metadata
            chunk_text = metadata.get('text', '')

            # Create a preview of the content (first 200 characters)
            text_preview = ""
            if chunk_text:
                text_preview = chunk_text[:200] + "..." if len(chunk_text) > 200 else chunk_text
            else:
                text_preview = "[Text content not available]"

            result_entry = f"""📄 Result {i}:
   • Document: {document_id}
   • File: {file_name}
   • Chunk: {chunk_index + 1}
   • Similarity Score: {score:.4f}
   • Word Count: {word_count}
   • Preview: {text_preview}
   • Match ID: {match['id']}"""
            
            formatted_results.append(result_entry)
        
        # Create summary
        total_results = len(filtered_matches)
        avg_score = sum(match.get('score', 0) for match in filtered_matches) / total_results
        
        result_text = f"""🔍 Search Results for: "{query}"

📊 Summary:
   • Found: {total_results} relevant chunks
   • Average Similarity: {avg_score:.4f}
   • Score Threshold: {min_score:.2f}

📋 Results:

{chr(10).join(formatted_results)}

💡 Use the retrieve_documents tool to get the full content and AI-generated answers."""
        
        logger.info(f"Search completed: {total_results} results found")
        return result_text
        
    except Exception as e:
        error_msg = f"❌ Error performing semantic search: {str(e)}"
        logger.error(f"Semantic search failed for query '{query}': {str(e)}")
        return error_msg


async def search_by_keywords(keywords: str, top_k: int = 5) -> str:
    """
    Search documents using keyword-based approach combined with semantic search.
    
    Args:
        keywords: Comma-separated keywords to search for
        top_k: Number of top results to return
    
    Returns:
        Formatted search results
    """
    try:
        logger.info(f"Performing keyword-based search for: {keywords}")
        
        if not keywords.strip():
            return "❌ Error: Keywords cannot be empty"
        
        # Convert keywords to a search query
        keyword_list = [kw.strip() for kw in keywords.split(',') if kw.strip()]
        
        if not keyword_list:
            return "❌ Error: No valid keywords provided"
        
        # Create a search query from keywords
        search_query = f"Find information about: {', '.join(keyword_list)}"
        
        # Perform semantic search
        results = await search_documents(
            query=search_query,
            top_k=top_k,
            min_score=0.1  # Lower threshold for keyword search
        )
        
        # Add keyword context to the results
        keyword_context = f"""🔑 Keyword Search for: {', '.join(keyword_list)}
(Converted to semantic search query: "{search_query}")

{results}"""
        
        return keyword_context
        
    except Exception as e:
        error_msg = f"❌ Error performing keyword search: {str(e)}"
        logger.error(f"Keyword search failed for '{keywords}': {str(e)}")
        return error_msg


async def get_search_suggestions(partial_query: str) -> str:
    """
    Get search suggestions based on a partial query.
    
    Args:
        partial_query: Partial search query
    
    Returns:
        Search suggestions
    """
    try:
        logger.info(f"Generating search suggestions for: {partial_query}")
        
        if not partial_query.strip():
            return "❌ Error: Partial query cannot be empty"
        
        # Use Gemini to generate search suggestions
        suggestions_prompt = f"""Based on the partial query "{partial_query}", suggest 5 related search queries that might help find relevant information in a document database. Focus on:
1. Expanding the query with related terms
2. Different ways to phrase the same question
3. More specific variations
4. Broader context variations

Return only the suggestions, one per line, without numbering."""
        
        suggestions = await gemini_service.generate_response(
            prompt=suggestions_prompt,
            context="",
            max_tokens=200
        )
        
        if not suggestions:
            return f"❌ Error: Could not generate suggestions for '{partial_query}'"
        
        suggestion_text = f"""💡 Search Suggestions for: "{partial_query}"

{suggestions}

🔍 Try using any of these suggestions with the search_documents tool."""
        
        return suggestion_text
        
    except Exception as e:
        error_msg = f"❌ Error generating search suggestions: {str(e)}"
        logger.error(f"Search suggestions failed for '{partial_query}': {str(e)}")
        return error_msg
