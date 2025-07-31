"""
Text chunking utilities for processing documents into manageable chunks.
Supports various chunking strategies for optimal RAG performance.
"""

import re
from typing import List, Optional
from rag_mcp_server.config import Config


class TextChunker:
    """Handles text chunking with configurable strategies."""
    
    def __init__(self, 
                 chunk_size: int = None, 
                 chunk_overlap: int = None,
                 max_chunks: int = None):
        """
        Initialize text chunker with configuration.
        
        Args:
            chunk_size: Maximum size of each chunk in characters
            chunk_overlap: Number of characters to overlap between chunks
            max_chunks: Maximum number of chunks to generate per document
        """
        self.chunk_size = chunk_size or Config.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or Config.CHUNK_OVERLAP
        self.max_chunks = max_chunks or Config.MAX_CHUNKS_PER_DOCUMENT
    
    def chunk_text(self, text: str, strategy: str = "sliding_window") -> List[str]:
        """
        Chunk text using the specified strategy.
        
        Args:
            text: Input text to chunk
            strategy: Chunking strategy ('sliding_window', 'sentence_aware', 'paragraph_aware')
            
        Returns:
            List of text chunks
        """
        # Clean and normalize text
        cleaned_text = self._clean_text(text)
        
        if strategy == "sliding_window":
            return self._sliding_window_chunk(cleaned_text)
        elif strategy == "sentence_aware":
            return self._sentence_aware_chunk(cleaned_text)
        elif strategy == "paragraph_aware":
            return self._paragraph_aware_chunk(cleaned_text)
        else:
            raise ValueError(f"Unknown chunking strategy: {strategy}")
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text for processing."""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters that might interfere with processing
        text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)\[\]\{\}\"\'\/]', '', text)
        
        # Strip leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def _sliding_window_chunk(self, text: str) -> List[str]:
        """
        Simple sliding window chunking strategy.
        
        Args:
            text: Input text to chunk
            
        Returns:
            List of text chunks
        """
        chunks = []
        start = 0
        
        while start < len(text) and len(chunks) < self.max_chunks:
            end = start + self.chunk_size
            
            # If this is not the last chunk, try to end at a word boundary
            if end < len(text):
                # Look for the last space within the chunk
                last_space = text.rfind(' ', start, end)
                if last_space > start:
                    end = last_space
            
            chunk = text[start:end].strip()
            if chunk:  # Only add non-empty chunks
                chunks.append(chunk)
            
            # Move start position with overlap
            start = end - self.chunk_overlap
            if start <= 0:
                start = end
        
        return chunks
    
    def _sentence_aware_chunk(self, text: str) -> List[str]:
        """
        Sentence-aware chunking strategy that tries to preserve sentence boundaries.
        
        Args:
            text: Input text to chunk
            
        Returns:
            List of text chunks
        """
        # Split text into sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            if len(chunks) >= self.max_chunks:
                break
                
            # Check if adding this sentence would exceed chunk size
            if len(current_chunk) + len(sentence) + 1 <= self.chunk_size:
                if current_chunk:
                    current_chunk += " " + sentence
                else:
                    current_chunk = sentence
            else:
                # Save current chunk if it's not empty
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                
                # Start new chunk with current sentence
                current_chunk = sentence
        
        # Add the last chunk if it's not empty
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def _paragraph_aware_chunk(self, text: str) -> List[str]:
        """
        Paragraph-aware chunking strategy that preserves paragraph boundaries.
        
        Args:
            text: Input text to chunk
            
        Returns:
            List of text chunks
        """
        # Split text into paragraphs
        paragraphs = re.split(r'\n\s*\n', text)
        
        chunks = []
        current_chunk = ""
        
        for paragraph in paragraphs:
            if len(chunks) >= self.max_chunks:
                break
                
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            
            # Check if adding this paragraph would exceed chunk size
            if len(current_chunk) + len(paragraph) + 2 <= self.chunk_size:
                if current_chunk:
                    current_chunk += "\n\n" + paragraph
                else:
                    current_chunk = paragraph
            else:
                # Save current chunk if it's not empty
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                
                # If paragraph itself is too long, split it using sentence-aware strategy
                if len(paragraph) > self.chunk_size:
                    para_chunks = self._sentence_aware_chunk(paragraph)
                    chunks.extend(para_chunks)
                    current_chunk = ""
                else:
                    current_chunk = paragraph
        
        # Add the last chunk if it's not empty
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def get_chunk_metadata(self, chunk: str, chunk_index: int, total_chunks: int) -> dict:
        """
        Generate metadata for a text chunk.
        
        Args:
            chunk: The text chunk
            chunk_index: Index of this chunk
            total_chunks: Total number of chunks in the document
            
        Returns:
            Dictionary containing chunk metadata
        """
        return {
            "chunk_index": chunk_index,
            "total_chunks": total_chunks,
            "chunk_length": len(chunk),
            "word_count": len(chunk.split()),
            "char_count": len(chunk)
        }
