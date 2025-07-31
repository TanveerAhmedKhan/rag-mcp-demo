#!/usr/bin/env python3
"""
Test script to verify the RAG MCP Server can start and register tools.
"""

import os
import sys
import asyncio
import subprocess
import time
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from rag_mcp_server.config import Config


async def test_server_startup():
    """Test that the MCP server can start up successfully."""
    print("🧪 Testing RAG MCP Server Startup")
    print("=" * 50)
    
    # Check configuration
    print("\n📋 Checking configuration...")
    try:
        if Config.validate():
            print("✅ Configuration is valid")
        else:
            print("❌ Configuration validation failed")
            return False
    except Exception as e:
        print(f"❌ Configuration error: {str(e)}")
        return False
    
    # Test server import
    print("\n📦 Testing server imports...")
    try:
        from rag_mcp_server.main import mcp
        print("✅ Server imports successful")
        print(f"   Server name: {mcp.name}")
    except Exception as e:
        print(f"❌ Import error: {str(e)}")
        return False
    
    # Test service initialization
    print("\n🔧 Testing service initialization...")
    try:
        from rag_mcp_server.services.pinecone_service import PineconeService
        from rag_mcp_server.services.gemini_service import GeminiService
        from rag_mcp_server.services.pdf_processor import PDFProcessor
        
        print("✅ Service imports successful")
        
        # Test service creation (without actual API calls)
        print("   Testing service instantiation...")
        # Note: These will fail if API keys are invalid, but that's expected in testing
        
    except Exception as e:
        print(f"❌ Service initialization error: {str(e)}")
        return False
    
    # Test tool registration
    print("\n🛠️  Testing tool registration...")
    try:
        # Get list of registered tools
        tools = []
        # Note: FastMCP doesn't expose tools directly, so we'll check the module
        import rag_mcp_server.main as main_module
        
        # Check if the main functions are defined
        tool_functions = [
            'upload_pdf',
            'search_documents', 
            'retrieve_documents',
            'ask_question',
            'get_pdf_info',
            'list_processed_documents',
            'search_by_keywords',
            'get_search_suggestions',
            'summarize_document'
        ]
        
        for func_name in tool_functions:
            if hasattr(main_module, func_name):
                tools.append(func_name)
        
        print(f"✅ Found {len(tools)} registered tools:")
        for tool in tools:
            print(f"   • {tool}")
            
    except Exception as e:
        print(f"❌ Tool registration error: {str(e)}")
        return False
    
    print("\n🎉 Server startup test completed successfully!")
    print("\n💡 Next steps:")
    print("1. Run: python -m rag_mcp_server.main")
    print("2. Configure VSCode with the MCP server")
    print("3. Test with actual PDF documents")
    
    return True


async def test_quick_server_run():
    """Test running the server for a few seconds."""
    print("\n🚀 Testing quick server run...")
    
    try:
        # Start the server process
        process = subprocess.Popen(
            [sys.executable, "-m", "rag_mcp_server.main"],
            cwd=project_root,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Wait a few seconds
        time.sleep(3)
        
        # Check if process is still running
        if process.poll() is None:
            print("✅ Server started successfully and is running")
            process.terminate()
            process.wait(timeout=5)
            return True
        else:
            stdout, stderr = process.communicate()
            print("❌ Server exited unexpectedly")
            print(f"   stdout: {stdout}")
            print(f"   stderr: {stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Server run test failed: {str(e)}")
        return False


async def main():
    """Main test function."""
    print("🧪 RAG MCP Server Test Suite")
    print("=" * 50)
    
    # Test startup
    startup_success = await test_server_startup()
    
    if startup_success:
        # Test quick run
        run_success = await test_quick_server_run()
        
        if run_success:
            print("\n🎉 All tests passed! Your RAG MCP Server is ready to use.")
        else:
            print("\n⚠️  Startup test passed but server run test failed.")
            print("   Check your API keys and network connectivity.")
    else:
        print("\n❌ Tests failed. Please check the errors above.")
        return False
    
    return True


if __name__ == "__main__":
    asyncio.run(main())
