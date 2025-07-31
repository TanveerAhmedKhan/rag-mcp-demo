"""
PDF processing service for text extraction and document handling.
"""

import logging
import os
from typing import List, Dict, Any, Optional
import pdfplumber
from rag_mcp_server.config import Config
from rag_mcp_server.utils.text_chunking import TextChunker

logger = logging.getLogger(__name__)


class PDFProcessor:
    """Service for processing PDF documents."""
    
    def __init__(self):
        """Initialize PDF processor with text chunker."""
        self.text_chunker = TextChunker(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP,
            max_chunks=Config.MAX_CHUNKS_PER_DOCUMENT
        )
        logger.info("Initialized PDF processor")
    
    async def extract_text_from_pdf(self, pdf_path: str) -> str:
        """
        Extract text content from PDF file.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Extracted text content
        """
        try:
            if not os.path.exists(pdf_path):
                raise FileNotFoundError(f"PDF file not found: {pdf_path}")
            
            text_content = ""
            page_count = 0
            
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages, 1):
                    try:
                        page_text = page.extract_text()
                        if page_text:
                            text_content += f"\n--- Page {page_num} ---\n"
                            text_content += page_text + "\n"
                            page_count += 1
                    except Exception as e:
                        logger.warning(f"Failed to extract text from page {page_num}: {str(e)}")
                        continue
            
            if not text_content.strip():
                raise ValueError("No text content extracted from PDF")
            
            logger.info(f"Extracted text from {page_count} pages of PDF: {pdf_path}")
            return text_content.strip()
            
        except Exception as e:
            logger.error(f"Failed to extract text from PDF {pdf_path}: {str(e)}")
            raise
    
    async def extract_metadata_from_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """
        Extract metadata from PDF file.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Dictionary containing PDF metadata
        """
        try:
            metadata = {
                "file_path": pdf_path,
                "file_name": os.path.basename(pdf_path),
                "file_size": os.path.getsize(pdf_path),
                "page_count": 0,
                "title": "",
                "author": "",
                "subject": "",
                "creator": "",
                "creation_date": "",
                "modification_date": ""
            }
            
            with pdfplumber.open(pdf_path) as pdf:
                metadata["page_count"] = len(pdf.pages)
                
                # Extract PDF metadata if available
                if pdf.metadata:
                    metadata.update({
                        "title": pdf.metadata.get("Title", ""),
                        "author": pdf.metadata.get("Author", ""),
                        "subject": pdf.metadata.get("Subject", ""),
                        "creator": pdf.metadata.get("Creator", ""),
                        "creation_date": str(pdf.metadata.get("CreationDate", "")),
                        "modification_date": str(pdf.metadata.get("ModDate", ""))
                    })
            
            logger.debug(f"Extracted metadata from PDF: {pdf_path}")
            return metadata
            
        except Exception as e:
            logger.error(f"Failed to extract metadata from PDF {pdf_path}: {str(e)}")
            return {"file_path": pdf_path, "error": str(e)}
    
    async def process_pdf_to_chunks(self, pdf_path: str, document_id: str, 
                                  chunking_strategy: str = "sliding_window") -> List[Dict[str, Any]]:
        """
        Process PDF into text chunks with metadata.
        
        Args:
            pdf_path: Path to the PDF file
            document_id: Unique identifier for the document
            chunking_strategy: Strategy for text chunking
            
        Returns:
            List of processed chunks with metadata
        """
        try:
            # Extract text content
            text_content = await self.extract_text_from_pdf(pdf_path)
            
            # Extract metadata
            pdf_metadata = await self.extract_metadata_from_pdf(pdf_path)
            
            # Chunk the text
            chunks = self.text_chunker.chunk_text(text_content, strategy=chunking_strategy)
            
            if not chunks:
                raise ValueError("No chunks generated from PDF content")
            
            # Process chunks with metadata
            processed_chunks = []
            for i, chunk in enumerate(chunks):
                chunk_metadata = {
                    "document_id": document_id,
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "source_file": pdf_path,
                    "file_name": os.path.basename(pdf_path),
                    "chunk_length": len(chunk),
                    "word_count": len(chunk.split()),
                    "chunking_strategy": chunking_strategy,
                    **pdf_metadata  # Include PDF metadata
                }
                
                processed_chunks.append({
                    "id": f"{document_id}_chunk_{i}",
                    "text": chunk,
                    "metadata": chunk_metadata
                })
            
            logger.info(f"Processed PDF into {len(processed_chunks)} chunks: {document_id}")
            return processed_chunks
            
        except Exception as e:
            logger.error(f"Failed to process PDF to chunks {pdf_path}: {str(e)}")
            raise
    
    async def validate_pdf_file(self, pdf_path: str) -> bool:
        """
        Validate that the file is a readable PDF.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            True if valid PDF, False otherwise
        """
        try:
            if not os.path.exists(pdf_path):
                logger.error(f"PDF file does not exist: {pdf_path}")
                return False
            
            if not pdf_path.lower().endswith('.pdf'):
                logger.error(f"File is not a PDF: {pdf_path}")
                return False
            
            # Try to open the PDF
            with pdfplumber.open(pdf_path) as pdf:
                if len(pdf.pages) == 0:
                    logger.error(f"PDF has no pages: {pdf_path}")
                    return False
            
            logger.debug(f"PDF validation successful: {pdf_path}")
            return True
            
        except Exception as e:
            logger.error(f"PDF validation failed for {pdf_path}: {str(e)}")
            return False
    
    async def get_pdf_info(self, pdf_path: str) -> Dict[str, Any]:
        """
        Get comprehensive information about a PDF file.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Dictionary containing PDF information
        """
        try:
            if not await self.validate_pdf_file(pdf_path):
                return {"error": "Invalid PDF file"}
            
            metadata = await self.extract_metadata_from_pdf(pdf_path)
            
            # Get text sample
            text_sample = ""
            try:
                full_text = await self.extract_text_from_pdf(pdf_path)
                text_sample = full_text[:500] + "..." if len(full_text) > 500 else full_text
            except Exception as e:
                logger.warning(f"Could not extract text sample: {str(e)}")
            
            info = {
                **metadata,
                "text_sample": text_sample,
                "estimated_chunks": len(self.text_chunker.chunk_text(text_sample)) if text_sample else 0,
                "is_valid": True
            }
            
            return info
            
        except Exception as e:
            logger.error(f"Failed to get PDF info for {pdf_path}: {str(e)}")
            return {"error": str(e), "is_valid": False}
