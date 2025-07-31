"""
Google Gemini AI service for embeddings and text generation.
"""

import logging
from typing import List, Optional, Dict, Any
import google.generativeai as genai
from rag_mcp_server.config import Config

logger = logging.getLogger(__name__)


class GeminiService:
    """Service for interacting with Google Gemini AI models."""
    
    def __init__(self):
        """Initialize Gemini service with configuration."""
        try:
            genai.configure(api_key=Config.GOOGLE_API_KEY)
            self.embedding_model = Config.EMBEDDING_MODEL
            self.llm_model = Config.LLM_MODEL
            logger.info(f"Initialized Gemini service with models: {self.embedding_model}, {self.llm_model}")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini service: {str(e)}")
            raise
    
    async def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings using Gemini embedding model.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            List of embedding vectors
        """
        try:
            embeddings = []
            
            for i, text in enumerate(texts):
                if not text.strip():
                    logger.warning(f"Empty text provided for embedding at index {i}, skipping")
                    continue

                # Check text size before sending to API
                text_size = len(text.encode('utf-8'))
                if text_size > 35000:  # Conservative limit for Gemini API
                    logger.error(f"Text at index {i} too large for embedding: {text_size} bytes")
                    logger.error(f"Text preview: {text[:100]}...")
                    continue

                try:
                    result = genai.embed_content(
                        model=f"models/{self.embedding_model}",
                        content=text,
                        task_type="retrieval_document"
                    )

                    if 'embedding' in result:
                        embeddings.append(result['embedding'])
                        logger.debug(f"Generated embedding for text {i} ({text_size} bytes)")
                    else:
                        logger.error(f"No embedding returned for text {i}: {text[:50]}...")

                except Exception as e:
                    logger.error(f"Failed to generate embedding for text {i}: {str(e)}")
                    logger.error(f"Text size: {text_size} bytes, preview: {text[:100]}...")
                    continue
            
            logger.info(f"Generated {len(embeddings)} embeddings from {len(texts)} texts")
            return embeddings
            
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {str(e)}")
            return []
    
    async def generate_query_embedding(self, query: str) -> Optional[List[float]]:
        """
        Generate embedding for a search query.
        
        Args:
            query: Search query text
            
        Returns:
            Query embedding vector or None if failed
        """
        try:
            if not query.strip():
                logger.warning("Empty query provided for embedding")
                return None
            
            result = genai.embed_content(
                model=f"models/{self.embedding_model}",
                content=query,
                task_type="retrieval_query"
            )
            
            if 'embedding' in result:
                logger.debug(f"Generated query embedding for: {query[:50]}...")
                return result['embedding']
            else:
                logger.error("No embedding returned for query")
                return None
                
        except Exception as e:
            logger.error(f"Failed to generate query embedding: {str(e)}")
            return None
    
    async def generate_response(self, prompt: str, context: str = "", 
                              max_tokens: int = 1000) -> Optional[str]:
        """
        Generate response using Gemini LLM.
        
        Args:
            prompt: User prompt/question
            context: Additional context for the response
            max_tokens: Maximum tokens in response
            
        Returns:
            Generated response text or None if failed
        """
        try:
            model = genai.GenerativeModel(self.llm_model)
            
            # Construct the full prompt
            if context.strip():
                full_prompt = f"""Context: {context}

Question: {prompt}

Please provide a comprehensive answer based on the context provided. If the context doesn't contain enough information to answer the question, please say so clearly.

Answer:"""
            else:
                full_prompt = prompt
            
            # Generate response
            response = model.generate_content(
                full_prompt,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=max_tokens,
                    temperature=0.7,
                )
            )
            
            if response.text:
                logger.debug(f"Generated response for prompt: {prompt[:50]}...")
                return response.text
            else:
                logger.error("No text returned from Gemini model")
                return None
                
        except Exception as e:
            logger.error(f"Failed to generate response: {str(e)}")
            return None
    
    async def summarize_text(self, text: str, max_length: int = 200) -> Optional[str]:
        """
        Summarize text using Gemini LLM.
        
        Args:
            text: Text to summarize
            max_length: Maximum length of summary
            
        Returns:
            Summary text or None if failed
        """
        try:
            model = genai.GenerativeModel(self.llm_model)
            
            prompt = f"""Please provide a concise summary of the following text in approximately {max_length} characters:

{text}

Summary:"""
            
            response = model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=max_length // 4,  # Rough estimate of tokens
                    temperature=0.3,
                )
            )
            
            if response.text:
                logger.debug(f"Generated summary for text: {text[:50]}...")
                return response.text
            else:
                logger.error("No summary returned from Gemini model")
                return None
                
        except Exception as e:
            logger.error(f"Failed to generate summary: {str(e)}")
            return None
    
    async def extract_keywords(self, text: str, max_keywords: int = 10) -> List[str]:
        """
        Extract keywords from text using Gemini LLM.
        
        Args:
            text: Text to extract keywords from
            max_keywords: Maximum number of keywords to extract
            
        Returns:
            List of extracted keywords
        """
        try:
            model = genai.GenerativeModel(self.llm_model)
            
            prompt = f"""Extract the {max_keywords} most important keywords from the following text. Return only the keywords, separated by commas:

{text}

Keywords:"""
            
            response = model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=100,
                    temperature=0.3,
                )
            )
            
            if response.text:
                keywords = [kw.strip() for kw in response.text.split(',')]
                keywords = [kw for kw in keywords if kw]  # Remove empty strings
                logger.debug(f"Extracted {len(keywords)} keywords from text")
                return keywords[:max_keywords]
            else:
                logger.error("No keywords returned from Gemini model")
                return []
                
        except Exception as e:
            logger.error(f"Failed to extract keywords: {str(e)}")
            return []
