#!/usr/bin/env python3
"""
Environment setup script for RAG MCP Server.
Initializes Pinecone index and validates API connections.
"""

import os
import sys
import asyncio
from typing import Optional
from dotenv import load_dotenv

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from pinecone import Pinecone, ServerlessSpec
    import google.generativeai as genai
    from rag_mcp_server.config import Config
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please install dependencies: uv pip install -r requirements.txt")
    sys.exit(1)


class EnvironmentSetup:
    """Handles environment setup and validation."""

    def __init__(self):
        load_dotenv()
        self.config = Config()
        self.pc = None
    
    async def setup(self):
        """Run complete environment setup."""
        print("🚀 Starting RAG MCP Server Environment Setup")
        print("=" * 50)
        
        # Validate configuration
        if not self._validate_config():
            return False
        
        # Test API connections
        if not await self._test_api_connections():
            return False
        
        # Setup Pinecone index
        if not await self._setup_pinecone_index():
            return False
        
        print("\n✅ Environment setup completed successfully!")
        print("\nNext steps:")
        print("1. Test the MCP server: python -m rag_mcp_server.main")
        print("2. Configure VSCode: Copy .vscode/mcp.json to your workspace")
        print("3. Start using the RAG system in VSCode agent mode")
        
        return True
    
    def _validate_config(self) -> bool:
        """Validate required configuration."""
        print("\n📋 Validating configuration...")
        
        required_vars = [
            ("GOOGLE_API_KEY", self.config.GOOGLE_API_KEY),
            ("PINECONE_API_KEY", self.config.PINECONE_API_KEY),
            ("PINECONE_ENVIRONMENT", self.config.PINECONE_ENVIRONMENT),
        ]
        
        missing_vars = []
        for var_name, var_value in required_vars:
            if not var_value:
                missing_vars.append(var_name)
        
        if missing_vars:
            print(f"❌ Missing required environment variables: {', '.join(missing_vars)}")
            print("Please check your .env file and ensure all required variables are set.")
            return False
        
        print("✅ Configuration validation passed")
        return True
    
    async def _test_api_connections(self) -> bool:
        """Test connections to external APIs."""
        print("\n🔌 Testing API connections...")
        
        # Test Google AI API
        try:
            genai.configure(api_key=self.config.GOOGLE_API_KEY)
            
            # Test embedding model
            test_result = genai.embed_content(
                model=f"models/{self.config.EMBEDDING_MODEL}",
                content="Test embedding"
            )
            
            if test_result and 'embedding' in test_result:
                print("✅ Google AI API connection successful")
            else:
                print("❌ Google AI API test failed - invalid response")
                return False
                
        except Exception as e:
            print(f"❌ Google AI API connection failed: {str(e)}")
            return False
        
        # Test Pinecone API
        try:
            self.pc = Pinecone(api_key=self.config.PINECONE_API_KEY)

            # List indexes to test connection
            indexes = self.pc.list_indexes()
            print("✅ Pinecone API connection successful")

        except Exception as e:
            print(f"❌ Pinecone API connection failed: {str(e)}")
            return False
        
        return True
    
    async def _setup_pinecone_index(self) -> bool:
        """Setup Pinecone index for document storage."""
        print(f"\n📊 Setting up Pinecone index: {self.config.PINECONE_INDEX_NAME}")
        
        try:
            # Ensure we have a Pinecone client
            if not self.pc:
                self.pc = Pinecone(api_key=self.config.PINECONE_API_KEY)

            # Check if index already exists
            existing_indexes = [index.name for index in self.pc.list_indexes()]

            if self.config.PINECONE_INDEX_NAME in existing_indexes:
                print(f"✅ Index '{self.config.PINECONE_INDEX_NAME}' already exists")

                # Test index connection
                index = self.pc.Index(self.config.PINECONE_INDEX_NAME)
                stats = index.describe_index_stats()
                print(f"   Index stats: {stats.get('total_vector_count', 0)} vectors")

            else:
                print(f"📝 Creating new index: {self.config.PINECONE_INDEX_NAME}")

                # Create index with appropriate dimensions for Gemini embeddings
                # gemini-embedding-001 produces 3072-dimensional vectors
                self.pc.create_index(
                    name=self.config.PINECONE_INDEX_NAME,
                    dimension=3072,  # Gemini embedding dimension
                    metric="cosine",
                    spec=ServerlessSpec(
                        cloud="aws",
                        region="us-east-1"
                    )
                )

                print(f"✅ Index '{self.config.PINECONE_INDEX_NAME}' created successfully")
            
            return True
            
        except Exception as e:
            print(f"❌ Pinecone index setup failed: {str(e)}")
            return False
    
    def _print_summary(self):
        """Print setup summary."""
        print("\n📋 Setup Summary:")
        print(f"   Google AI Model (Embedding): {self.config.EMBEDDING_MODEL}")
        print(f"   Google AI Model (LLM): {self.config.LLM_MODEL}")
        print(f"   Pinecone Environment: {self.config.PINECONE_ENVIRONMENT}")
        print(f"   Pinecone Index: {self.config.PINECONE_INDEX_NAME}")
        print(f"   MCP Server Name: {self.config.SERVER_NAME}")


async def main():
    """Main setup function."""
    setup = EnvironmentSetup()
    
    try:
        success = await setup.setup()
        if not success:
            print("\n❌ Setup failed. Please check the errors above and try again.")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n\n⚠️  Setup interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error during setup: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
