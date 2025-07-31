"""
Pinecone vector database service for document storage and retrieval.
"""

import logging
from typing import List, Dict, Any, Optional
from pinecone import Pinecone, ServerlessSpec
from rag_mcp_server.config import Config

logger = logging.getLogger(__name__)


class PineconeService:
    """Service for interacting with Pinecone vector database."""
    
    def __init__(self):
        """Initialize Pinecone service with configuration."""
        try:
            self.pc = Pinecone(api_key=Config.PINECONE_API_KEY)
            self.index_name = Config.PINECONE_INDEX_NAME
            self.index = None
            self._initialize_index()
        except Exception as e:
            logger.error(f"Failed to initialize Pinecone service: {str(e)}")
            raise
    
    def _initialize_index(self):
        """Initialize or connect to the Pinecone index."""
        try:
            # Check if index exists
            existing_indexes = [index.name for index in self.pc.list_indexes()]

            if self.index_name not in existing_indexes:
                logger.info(f"Creating Pinecone index: {self.index_name}")
                self.pc.create_index(
                    name=self.index_name,
                    dimension=3072,  # Gemini embedding dimension (gemini-embedding-001)
                    metric="cosine",
                    spec=ServerlessSpec(
                        cloud="aws",
                        region="us-east-1"
                    )
                )
                logger.info(f"Index {self.index_name} created successfully")

            # Connect to the index
            self.index = self.pc.Index(self.index_name)
            logger.info(f"Connected to Pinecone index: {self.index_name}")

        except Exception as e:
            logger.error(f"Failed to initialize Pinecone index: {str(e)}")
            raise
    
    async def upsert_vectors(self, vectors: List[Dict[str, Any]]) -> bool:
        """
        Upsert document vectors to Pinecone index.
        
        Args:
            vectors: List of vector dictionaries with id, values, and metadata
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if not self.index:
                raise Exception("Pinecone index not initialized")

            if not vectors:
                logger.warning("No vectors provided for upsert")
                return True

            logger.info(f"Starting upsert of {len(vectors)} vectors to Pinecone")

            # Validate vectors before upsert
            for i, vector in enumerate(vectors):
                if not isinstance(vector, dict):
                    raise ValueError(f"Vector {i} is not a dictionary: {type(vector)}")

                required_fields = ['id', 'values', 'metadata']
                missing_fields = [field for field in required_fields if field not in vector]
                if missing_fields:
                    raise ValueError(f"Vector {i} missing required fields: {missing_fields}")

                if not isinstance(vector['values'], list):
                    raise ValueError(f"Vector {i} 'values' must be a list, got {type(vector['values'])}")

                if len(vector['values']) != 3072:  # Expected dimension
                    raise ValueError(f"Vector {i} has wrong dimension: {len(vector['values'])}, expected 3072")

            # Upsert vectors in batches to avoid API limits
            batch_size = 100
            total_upserted = 0

            for i in range(0, len(vectors), batch_size):
                batch = vectors[i:i + batch_size]
                batch_num = i//batch_size + 1

                logger.debug(f"Processing batch {batch_num} with {len(batch)} vectors")

                # Log metadata size for debugging
                for vector in batch:
                    metadata = vector.get('metadata', {})
                    text_content = metadata.get('text', '')
                    metadata_size = len(str(metadata).encode('utf-8'))
                    logger.debug(f"Vector {vector['id']}: metadata size = {metadata_size} bytes, text length = {len(text_content)}")

                    # Check metadata size limit
                    if metadata_size > 40000:  # 40KB limit
                        logger.warning(f"Vector {vector['id']} metadata size ({metadata_size} bytes) exceeds 40KB limit")

                try:
                    result = self.index.upsert(vectors=batch)
                    upserted_count = result.get('upserted_count', len(batch))
                    total_upserted += upserted_count
                    logger.debug(f"Batch {batch_num} upserted {upserted_count} vectors successfully")

                except Exception as batch_error:
                    logger.error(f"Failed to upsert batch {batch_num}: {str(batch_error)}")
                    logger.error(f"Batch error type: {type(batch_error)}")

                    # Try to identify the problematic vector
                    for j, vector in enumerate(batch):
                        try:
                            single_result = self.index.upsert(vectors=[vector])
                            logger.debug(f"Vector {vector['id']} upserted individually")
                        except Exception as single_error:
                            logger.error(f"Vector {vector['id']} failed individual upsert: {str(single_error)}")
                            raise single_error

                    raise batch_error

            logger.info(f"Successfully upserted {total_upserted} vectors to Pinecone")
            return True

        except Exception as e:
            logger.error(f"Failed to upsert vectors to Pinecone: {str(e)}")
            logger.error(f"Error type: {type(e)}")

            # Log additional context for debugging
            if vectors:
                logger.error(f"Number of vectors: {len(vectors)}")
                logger.error(f"First vector ID: {vectors[0].get('id', 'Unknown')}")
                logger.error(f"First vector structure: {list(vectors[0].keys()) if isinstance(vectors[0], dict) else 'Not a dict'}")

            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            return False
    
    async def query_vectors(self, query_vector: List[float], top_k: int = 5, 
                          filter_dict: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Query similar vectors from Pinecone index.
        
        Args:
            query_vector: Query vector for similarity search
            top_k: Number of top results to return
            filter_dict: Optional metadata filter
            
        Returns:
            Dict containing query results
        """
        try:
            if not self.index:
                raise Exception("Pinecone index not initialized")
            
            query_params = {
                "vector": query_vector,
                "top_k": top_k,
                "include_metadata": True,
                "include_values": False
            }
            
            if filter_dict:
                query_params["filter"] = filter_dict
            
            results = self.index.query(**query_params)
            logger.debug(f"Query returned {len(results.get('matches', []))} results")
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to query vectors from Pinecone: {str(e)}")
            return {"matches": []}
    
    async def delete_vectors(self, ids: List[str]) -> bool:
        """
        Delete vectors from Pinecone index.
        
        Args:
            ids: List of vector IDs to delete
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if not self.index:
                raise Exception("Pinecone index not initialized")
            
            self.index.delete(ids=ids)
            logger.info(f"Successfully deleted {len(ids)} vectors from Pinecone")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete vectors from Pinecone: {str(e)}")
            return False
    
    async def get_index_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the Pinecone index.
        
        Returns:
            Dict containing index statistics
        """
        try:
            if not self.index:
                raise Exception("Pinecone index not initialized")
            
            stats = self.index.describe_index_stats()
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get index stats: {str(e)}")
            return {}
    
    async def list_documents(self, document_id_prefix: Optional[str] = None) -> List[str]:
        """
        List document IDs stored in the index.
        
        Args:
            document_id_prefix: Optional prefix to filter document IDs
            
        Returns:
            List of document IDs
        """
        try:
            # Note: This is a simplified implementation
            # In practice, you might want to store document metadata separately
            # or use Pinecone's metadata filtering capabilities
            
            stats = await self.get_index_stats()
            namespaces = stats.get('namespaces', {})
            
            # For now, return a placeholder
            # You would implement proper document listing based on your metadata structure
            return []
            
        except Exception as e:
            logger.error(f"Failed to list documents: {str(e)}")
            return []
