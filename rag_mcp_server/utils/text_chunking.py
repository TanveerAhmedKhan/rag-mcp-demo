"""
Enhanced text chunking utilities for processing documents into manageable chunks.
Supports various chunking strategies including recursive character text splitting
for optimal RAG performance and semantic coherence.
"""

import re
import logging
from typing import List, Optional, Dict, Any
from rag_mcp_server.config import Config

logger = logging.getLogger(__name__)


class RecursiveCharacterTextSplitter:
    """
    Enhanced recursive character text splitter that maintains semantic coherence
    by using hierarchical separators and intelligent boundary detection.
    """

    def __init__(self,
                 chunk_size: int = None,
                 chunk_overlap: int = None,
                 max_chunks: int = None,
                 min_chunk_size: int = 50,
                 separators: Optional[List[str]] = None):
        """
        Initialize recursive character text splitter.

        Args:
            chunk_size: Maximum size of each chunk in characters
            chunk_overlap: Number of characters to overlap between chunks
            max_chunks: Maximum number of chunks to generate per document
            min_chunk_size: Minimum size for a chunk to be considered valid
            separators: Custom list of separators in priority order
        """
        self.chunk_size = chunk_size or Config.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or Config.CHUNK_OVERLAP
        self.max_chunks = max_chunks or Config.MAX_CHUNKS_PER_DOCUMENT
        self.min_chunk_size = min_chunk_size

        # Hierarchical separators in priority order
        self.separators = separators or [
            "\n\n",      # Paragraph breaks (highest priority)
            "\n",        # Line breaks
            ". ",        # Sentence endings with space
            "! ",        # Exclamation with space
            "? ",        # Question with space
            "; ",        # Semicolon with space
            ", ",        # Comma with space
            " ",         # Word boundaries
            ""           # Character level (last resort)
        ]

        # Maximum text size for embedding API (from recent Pinecone fixes)
        self.max_embedding_size = 30000  # Conservative limit for Gemini API

        logger.debug(f"Initialized RecursiveCharacterTextSplitter: chunk_size={self.chunk_size}, "
                    f"overlap={self.chunk_overlap}, max_chunks={self.max_chunks}, "
                    f"min_chunk_size={self.min_chunk_size}")

    def split_text(self, text: str) -> List[str]:
        """
        Split text using recursive character splitting with hierarchical separators.

        Args:
            text: Input text to split

        Returns:
            List of text chunks maintaining semantic coherence
        """
        # Clean and normalize text
        cleaned_text = self._clean_text(text)

        if not cleaned_text:
            return []

        # Check if text is within embedding size limit
        if len(cleaned_text.encode('utf-8')) > self.max_embedding_size:
            logger.warning(f"Text size ({len(cleaned_text.encode('utf-8'))} bytes) exceeds embedding limit")

        # Start recursive splitting
        chunks = self._recursive_split(cleaned_text, self.separators)

        # Apply overlap and finalize chunks
        final_chunks = self._apply_overlap(chunks)

        # Limit number of chunks
        if len(final_chunks) > self.max_chunks:
            logger.warning(f"Generated {len(final_chunks)} chunks, limiting to {self.max_chunks}")
            final_chunks = final_chunks[:self.max_chunks]

        logger.info(f"Split text into {len(final_chunks)} chunks using recursive character splitting")
        return final_chunks

    def _recursive_split(self, text: str, separators: List[str]) -> List[str]:
        """
        Recursively split text using hierarchical separators.

        Args:
            text: Text to split
            separators: List of separators in priority order

        Returns:
            List of text chunks
        """
        # Base case: if text is small enough, return as single chunk
        if len(text) <= self.chunk_size:
            return [text] if len(text) >= self.min_chunk_size else []

        # Try each separator in order
        for separator in separators:
            if separator in text:
                chunks = self._split_by_separator(text, separator, separators)
                if chunks:  # If splitting was successful
                    return chunks

        # If no separator worked, force split at chunk_size
        logger.debug("No suitable separator found, forcing character-level split")
        return self._force_split(text)

    def _split_by_separator(self, text: str, separator: str, remaining_separators: List[str]) -> List[str]:
        """
        Split text by a specific separator and recursively process large chunks.

        Args:
            text: Text to split
            separator: Separator to use for splitting
            remaining_separators: Remaining separators for recursive splitting

        Returns:
            List of text chunks
        """
        if separator == "":
            # Character-level splitting (last resort)
            return self._force_split(text)

        # Split by separator
        parts = text.split(separator)

        chunks = []
        current_chunk = ""

        for i, part in enumerate(parts):
            # Add separator back (except for last part)
            if i < len(parts) - 1:
                part_with_sep = part + separator
            else:
                part_with_sep = part

            # Check if adding this part would exceed chunk size
            potential_chunk = current_chunk + part_with_sep

            if len(potential_chunk) <= self.chunk_size:
                current_chunk = potential_chunk
            else:
                # Save current chunk if it's not empty and meets minimum size
                if current_chunk.strip() and len(current_chunk.strip()) >= self.min_chunk_size:
                    chunks.append(current_chunk.strip())

                # If the part itself is too large, recursively split it
                if len(part_with_sep) > self.chunk_size:
                    # Find next separator in hierarchy
                    next_separators = remaining_separators[remaining_separators.index(separator) + 1:]
                    if next_separators:
                        sub_chunks = self._recursive_split(part_with_sep, next_separators)
                        chunks.extend(sub_chunks)
                    else:
                        # Force split if no more separators
                        sub_chunks = self._force_split(part_with_sep)
                        chunks.extend(sub_chunks)
                    current_chunk = ""
                else:
                    current_chunk = part_with_sep

        # Add the last chunk if it's not empty and meets minimum size
        if current_chunk.strip() and len(current_chunk.strip()) >= self.min_chunk_size:
            chunks.append(current_chunk.strip())

        return chunks

    def _force_split(self, text: str) -> List[str]:
        """
        Force split text at character boundaries when no separator works.

        Args:
            text: Text to split

        Returns:
            List of text chunks
        """
        chunks = []
        start = 0

        while start < len(text):
            end = start + self.chunk_size

            # Try to end at a word boundary if possible
            if end < len(text):
                # Look for the last space within the chunk
                last_space = text.rfind(' ', start, end)
                if last_space > start:
                    end = last_space

            chunk = text[start:end].strip()
            if chunk and len(chunk) >= self.min_chunk_size:
                chunks.append(chunk)

            start = end

        return chunks

    def _apply_overlap(self, chunks: List[str]) -> List[str]:
        """
        Apply overlap between chunks to maintain context.

        Args:
            chunks: List of text chunks

        Returns:
            List of chunks with overlap applied
        """
        if len(chunks) <= 1 or self.chunk_overlap <= 0:
            return chunks

        overlapped_chunks = []

        for i, chunk in enumerate(chunks):
            if i == 0:
                # First chunk - no overlap needed
                overlapped_chunks.append(chunk)
            else:
                # Add overlap from previous chunk
                prev_chunk = chunks[i - 1]

                # Get overlap text from end of previous chunk
                overlap_text = ""
                if len(prev_chunk) > self.chunk_overlap:
                    overlap_text = prev_chunk[-self.chunk_overlap:]

                    # Try to start overlap at a word boundary
                    first_space = overlap_text.find(' ')
                    if first_space > 0:
                        overlap_text = overlap_text[first_space + 1:]

                # Combine overlap with current chunk
                if overlap_text:
                    combined_chunk = overlap_text + " " + chunk

                    # Ensure combined chunk doesn't exceed size limits
                    if len(combined_chunk) <= self.chunk_size:
                        overlapped_chunks.append(combined_chunk)
                    else:
                        # If overlap makes chunk too large, use original chunk
                        overlapped_chunks.append(chunk)
                else:
                    overlapped_chunks.append(chunk)

        return overlapped_chunks

    def _clean_text(self, text: str) -> str:
        """Clean and normalize text for processing while preserving semantic structure."""
        if not text:
            return ""

        # Remove excessive whitespace while preserving paragraph breaks
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)  # Normalize paragraph breaks
        text = re.sub(r'[ \t]+', ' ', text)  # Normalize spaces and tabs
        text = re.sub(r' *\n *', '\n', text)  # Clean line breaks

        # Remove special characters that might interfere with processing
        # Keep more punctuation for better semantic boundaries
        text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)\[\]\{\}\"\'\/\n]', '', text)

        # Strip leading/trailing whitespace
        text = text.strip()

        return text

    def get_chunk_info(self, chunks: List[str]) -> Dict[str, Any]:
        """
        Get information about the generated chunks.

        Args:
            chunks: List of text chunks

        Returns:
            Dictionary with chunk statistics
        """
        if not chunks:
            return {
                "total_chunks": 0,
                "total_characters": 0,
                "average_chunk_size": 0,
                "min_chunk_size": 0,
                "max_chunk_size": 0
            }

        chunk_sizes = [len(chunk) for chunk in chunks]

        return {
            "total_chunks": len(chunks),
            "total_characters": sum(chunk_sizes),
            "average_chunk_size": sum(chunk_sizes) / len(chunks),
            "min_chunk_size": min(chunk_sizes),
            "max_chunk_size": max(chunk_sizes),
            "chunks_over_embedding_limit": sum(1 for size in chunk_sizes if size > self.max_embedding_size)
        }


class TextChunker:
    """
    Legacy text chunker class for backward compatibility.
    Wraps the new RecursiveCharacterTextSplitter.
    """

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
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            max_chunks=max_chunks
        )

    def chunk_text(self, text: str, strategy: str = "recursive") -> List[str]:
        """
        Chunk text using the specified strategy.

        Args:
            text: Input text to chunk
            strategy: Chunking strategy ('recursive', 'sliding_window', 'sentence_aware', 'paragraph_aware')

        Returns:
            List of text chunks
        """
        if strategy == "recursive":
            return self.splitter.split_text(text)
        elif strategy == "sliding_window":
            return self._sliding_window_chunk(text)
        elif strategy == "sentence_aware":
            return self._sentence_aware_chunk(text)
        elif strategy == "paragraph_aware":
            return self._paragraph_aware_chunk(text)
        else:
            raise ValueError(f"Unknown chunking strategy: {strategy}")

    def _recursive_split(self, text: str, separators: List[str]) -> List[str]:
        """
        Recursively split text using hierarchical separators.

        Args:
            text: Text to split
            separators: List of separators in priority order

        Returns:
            List of text chunks
        """
        # Base case: if text is small enough, return as single chunk
        if len(text) <= self.chunk_size:
            return [text] if len(text) >= self.min_chunk_size else []

        # Try each separator in order
        for separator in separators:
            if separator in text:
                chunks = self._split_by_separator(text, separator, separators)
                if chunks:  # If splitting was successful
                    return chunks

        # If no separator worked, force split at chunk_size
        logger.debug("No suitable separator found, forcing character-level split")
        return self._force_split(text)

    def _split_by_separator(self, text: str, separator: str, remaining_separators: List[str]) -> List[str]:
        """
        Split text by a specific separator and recursively process large chunks.

        Args:
            text: Text to split
            separator: Separator to use for splitting
            remaining_separators: Remaining separators for recursive splitting

        Returns:
            List of text chunks
        """
        if separator == "":
            # Character-level splitting (last resort)
            return self._force_split(text)

        # Split by separator
        parts = text.split(separator)

        chunks = []
        current_chunk = ""

        for i, part in enumerate(parts):
            # Add separator back (except for last part)
            if i < len(parts) - 1:
                part_with_sep = part + separator
            else:
                part_with_sep = part

            # Check if adding this part would exceed chunk size
            potential_chunk = current_chunk + part_with_sep

            if len(potential_chunk) <= self.chunk_size:
                current_chunk = potential_chunk
            else:
                # Save current chunk if it's not empty and meets minimum size
                if current_chunk.strip() and len(current_chunk.strip()) >= self.min_chunk_size:
                    chunks.append(current_chunk.strip())

                # If the part itself is too large, recursively split it
                if len(part_with_sep) > self.chunk_size:
                    # Find next separator in hierarchy
                    next_separators = remaining_separators[remaining_separators.index(separator) + 1:]
                    if next_separators:
                        sub_chunks = self._recursive_split(part_with_sep, next_separators)
                        chunks.extend(sub_chunks)
                    else:
                        # Force split if no more separators
                        sub_chunks = self._force_split(part_with_sep)
                        chunks.extend(sub_chunks)
                    current_chunk = ""
                else:
                    current_chunk = part_with_sep

        # Add the last chunk if it's not empty and meets minimum size
        if current_chunk.strip() and len(current_chunk.strip()) >= self.min_chunk_size:
            chunks.append(current_chunk.strip())

        return chunks

    def _force_split(self, text: str) -> List[str]:
        """
        Force split text at character boundaries when no separator works.

        Args:
            text: Text to split

        Returns:
            List of text chunks
        """
        chunks = []
        start = 0

        while start < len(text):
            end = start + self.chunk_size

            # Try to end at a word boundary if possible
            if end < len(text):
                # Look for the last space within the chunk
                last_space = text.rfind(' ', start, end)
                if last_space > start:
                    end = last_space

            chunk = text[start:end].strip()
            if chunk and len(chunk) >= self.min_chunk_size:
                chunks.append(chunk)

            start = end

        return chunks

    def _apply_overlap(self, chunks: List[str]) -> List[str]:
        """
        Apply overlap between chunks to maintain context.

        Args:
            chunks: List of text chunks

        Returns:
            List of chunks with overlap applied
        """
        if len(chunks) <= 1 or self.chunk_overlap <= 0:
            return chunks

        overlapped_chunks = []

        for i, chunk in enumerate(chunks):
            if i == 0:
                # First chunk - no overlap needed
                overlapped_chunks.append(chunk)
            else:
                # Add overlap from previous chunk
                prev_chunk = chunks[i - 1]

                # Get overlap text from end of previous chunk
                overlap_text = ""
                if len(prev_chunk) > self.chunk_overlap:
                    overlap_text = prev_chunk[-self.chunk_overlap:]

                    # Try to start overlap at a word boundary
                    first_space = overlap_text.find(' ')
                    if first_space > 0:
                        overlap_text = overlap_text[first_space + 1:]

                # Combine overlap with current chunk
                if overlap_text:
                    combined_chunk = overlap_text + " " + chunk

                    # Ensure combined chunk doesn't exceed size limits
                    if len(combined_chunk) <= self.chunk_size:
                        overlapped_chunks.append(combined_chunk)
                    else:
                        # If overlap makes chunk too large, use original chunk
                        overlapped_chunks.append(chunk)
                else:
                    overlapped_chunks.append(chunk)

        return overlapped_chunks

    def _clean_text(self, text: str) -> str:
        """Clean and normalize text for processing while preserving semantic structure."""
        if not text:
            return ""

        # Remove excessive whitespace while preserving paragraph breaks
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)  # Normalize paragraph breaks
        text = re.sub(r'[ \t]+', ' ', text)  # Normalize spaces and tabs
        text = re.sub(r' *\n *', '\n', text)  # Clean line breaks

        # Remove special characters that might interfere with processing
        # Keep more punctuation for better semantic boundaries
        text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)\[\]\{\}\"\'\/\n]', '', text)

        # Strip leading/trailing whitespace
        text = text.strip()

        return text

    def get_chunk_info(self, chunks: List[str]) -> Dict[str, Any]:
        """
        Get information about the generated chunks.

        Args:
            chunks: List of text chunks

        Returns:
            Dictionary with chunk statistics
        """
        if not chunks:
            return {
                "total_chunks": 0,
                "total_characters": 0,
                "average_chunk_size": 0,
                "min_chunk_size": 0,
                "max_chunk_size": 0
            }

        chunk_sizes = [len(chunk) for chunk in chunks]

        return {
            "total_chunks": len(chunks),
            "total_characters": sum(chunk_sizes),
            "average_chunk_size": sum(chunk_sizes) / len(chunks),
            "min_chunk_size": min(chunk_sizes),
            "max_chunk_size": max(chunk_sizes),
            "chunks_over_embedding_limit": sum(1 for size in chunk_sizes if size > self.max_embedding_size)
        }

    def chunk_text(self, text: str, strategy: str = "recursive") -> List[str]:
        """
        Chunk text using the specified strategy.

        Args:
            text: Input text to chunk
            strategy: Chunking strategy ('recursive', 'sliding_window', 'sentence_aware', 'paragraph_aware')

        Returns:
            List of text chunks
        """
        if strategy == "recursive":
            return self.splitter.split_text(text)
        elif strategy == "sliding_window":
            return self._sliding_window_chunk(text)
        elif strategy == "sentence_aware":
            return self._sentence_aware_chunk(text)
        elif strategy == "paragraph_aware":
            return self._paragraph_aware_chunk(text)
        else:
            raise ValueError(f"Unknown chunking strategy: {strategy}")

    def _clean_text(self, text: str) -> str:
        """Clean and normalize text for processing (legacy method)."""
        return self.splitter._clean_text(text)
    
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
        Simple sliding window chunking strategy (legacy method).

        Args:
            text: Input text to chunk

        Returns:
            List of text chunks
        """
        cleaned_text = self._clean_text(text)
        chunks = []
        start = 0

        while start < len(cleaned_text) and len(chunks) < self.splitter.max_chunks:
            end = start + self.splitter.chunk_size

            # If this is not the last chunk, try to end at a word boundary
            if end < len(cleaned_text):
                # Look for the last space within the chunk
                last_space = cleaned_text.rfind(' ', start, end)
                if last_space > start:
                    end = last_space

            chunk = cleaned_text[start:end].strip()
            if chunk and len(chunk) >= self.splitter.min_chunk_size:
                chunks.append(chunk)

            # Move start position with overlap
            start = end - self.splitter.chunk_overlap
            if start <= 0:
                start = end

        return chunks
    
    def _sentence_aware_chunk(self, text: str) -> List[str]:
        """
        Sentence-aware chunking strategy that tries to preserve sentence boundaries (legacy method).

        Args:
            text: Input text to chunk

        Returns:
            List of text chunks
        """
        cleaned_text = self._clean_text(text)

        # Split text into sentences
        sentences = re.split(r'(?<=[.!?])\s+', cleaned_text)

        chunks = []
        current_chunk = ""

        for sentence in sentences:
            if len(chunks) >= self.splitter.max_chunks:
                break

            # Check if adding this sentence would exceed chunk size
            if len(current_chunk) + len(sentence) + 1 <= self.splitter.chunk_size:
                if current_chunk:
                    current_chunk += " " + sentence
                else:
                    current_chunk = sentence
            else:
                # Save current chunk if it's not empty and meets minimum size
                if current_chunk.strip() and len(current_chunk.strip()) >= self.splitter.min_chunk_size:
                    chunks.append(current_chunk.strip())

                # Start new chunk with current sentence
                current_chunk = sentence

        # Add the last chunk if it's not empty and meets minimum size
        if current_chunk.strip() and len(current_chunk.strip()) >= self.splitter.min_chunk_size:
            chunks.append(current_chunk.strip())

        return chunks
    
    def _paragraph_aware_chunk(self, text: str) -> List[str]:
        """
        Paragraph-aware chunking strategy that preserves paragraph boundaries (legacy method).

        Args:
            text: Input text to chunk

        Returns:
            List of text chunks
        """
        cleaned_text = self._clean_text(text)

        # Split text into paragraphs
        paragraphs = re.split(r'\n\s*\n', cleaned_text)

        chunks = []
        current_chunk = ""

        for paragraph in paragraphs:
            if len(chunks) >= self.splitter.max_chunks:
                break

            paragraph = paragraph.strip()
            if not paragraph:
                continue

            # Check if adding this paragraph would exceed chunk size
            if len(current_chunk) + len(paragraph) + 2 <= self.splitter.chunk_size:
                if current_chunk:
                    current_chunk += "\n\n" + paragraph
                else:
                    current_chunk = paragraph
            else:
                # Save current chunk if it's not empty and meets minimum size
                if current_chunk.strip() and len(current_chunk.strip()) >= self.splitter.min_chunk_size:
                    chunks.append(current_chunk.strip())

                # If paragraph itself is too long, split it using sentence-aware strategy
                if len(paragraph) > self.splitter.chunk_size:
                    para_chunks = self._sentence_aware_chunk(paragraph)
                    chunks.extend(para_chunks)
                    current_chunk = ""
                else:
                    current_chunk = paragraph

        # Add the last chunk if it's not empty and meets minimum size
        if current_chunk.strip() and len(current_chunk.strip()) >= self.splitter.min_chunk_size:
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
        # Count sentences and paragraphs for better metadata
        sentence_count = len(re.split(r'[.!?]+', chunk)) - 1  # -1 for empty last element
        paragraph_count = len(re.split(r'\n\s*\n', chunk))

        return {
            "chunk_index": chunk_index,
            "total_chunks": total_chunks,
            "chunk_length": len(chunk),
            "word_count": len(chunk.split()),
            "char_count": len(chunk),
            "sentence_count": max(1, sentence_count),  # At least 1 sentence
            "paragraph_count": max(1, paragraph_count),  # At least 1 paragraph
            "byte_size": len(chunk.encode('utf-8')),
            "exceeds_embedding_limit": len(chunk.encode('utf-8')) > 30000
        }
