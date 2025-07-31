# RAG MCP Server Setup Guide

## 🚀 Quick Start

### Step 1: Environment Setup
```bash
# Activate virtual environment (if not already active)
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install all dependencies
uv pip install -r requirements.txt
```

### Step 2: Configure Environment Variables
```bash
# Copy the environment template
cp .env.example .env

# Edit .env file with your API keys
# Required variables:
# - GOOGLE_API_KEY (from https://ai.google.dev/)
# - PINECONE_API_KEY (from https://app.pinecone.io/)
# - PINECONE_ENVIRONMENT (e.g., us-west1-gcp)
```

### Step 3: Initialize Environment
```bash
# Run the setup script to initialize Pinecone index
python scripts/setup_environment.py
```

### Step 4: Test the Server
```bash
# Test server startup and configuration
python scripts/test_server.py

# Run the MCP server
python -m rag_mcp_server.main
```

### Step 5: Configure VSCode
```bash
# Copy MCP configuration to your workspace
cp vscode_integration/.vscode/mcp.json .vscode/

# Restart VSCode and enable agent mode
```

## 🛠️ Available MCP Tools

Your RAG MCP server provides the following tools:

### 📄 PDF Management Tools
- **`upload_pdf`**: Upload and process PDF documents into the vector database
- **`get_pdf_info`**: Get information about a PDF file without processing it
- **`list_processed_documents`**: List documents stored in the vector database

### 🔍 Search Tools
- **`search_documents`**: Semantic search across all documents
- **`search_by_keywords`**: Keyword-based search with semantic enhancement
- **`get_search_suggestions`**: Get AI-generated search suggestions

### 📋 Retrieval Tools
- **`retrieve_documents`**: Get relevant content with optional AI-generated answers
- **`ask_question`**: Ask specific questions and get AI responses
- **`summarize_document`**: Generate summaries of specific documents

## 📝 Usage Examples

### Upload a PDF Document
```
In VSCode agent mode:
"Upload the PDF document at /path/to/document.pdf with document ID 'research-paper-1'"
```

### Search for Information
```
"Search for information about machine learning algorithms"
```

### Ask Questions
```
"What are the main conclusions about neural networks in the uploaded research papers?"
```

### Get Document Summary
```
"Summarize the document with ID 'research-paper-1'"
```

## 🔧 Configuration Options

### Environment Variables (.env)
```env
# Required
GOOGLE_API_KEY=your_google_api_key
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_ENVIRONMENT=us-west1-gcp

# Optional
PINECONE_INDEX_NAME=rag-documents
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
MAX_CHUNKS_PER_DOCUMENT=100
LOG_LEVEL=INFO
```

### VSCode MCP Configuration (.vscode/mcp.json)
The MCP configuration file allows VSCode to connect to your RAG server. It includes:
- Input prompts for API keys (secure storage)
- Server command and environment setup
- Transport configuration (STDIO)

## 🧪 Testing

### Test Server Functionality
```bash
# Run comprehensive tests
python scripts/test_server.py

# Test individual components
pytest tests/
```

### Test with Sample PDF
1. Place a PDF file in the `examples/sample_pdfs/` directory
2. Use the upload_pdf tool to process it
3. Test search and retrieval functionality

## 🐛 Troubleshooting

### Common Issues

**1. "Missing required environment variables"**
- Check your .env file has all required API keys
- Ensure .env file is in the project root directory

**2. "Failed to initialize Pinecone service"**
- Verify your Pinecone API key and environment
- Check network connectivity
- Ensure Pinecone index exists (run setup_environment.py)

**3. "Failed to generate embeddings"**
- Verify Google AI API key
- Check API quota and billing
- Ensure network connectivity to Google AI services

**4. "VSCode doesn't detect MCP server"**
- Ensure .vscode/mcp.json is in your workspace
- Restart VSCode completely
- Check VSCode logs for MCP errors

**5. "Server exits unexpectedly"**
- Check server logs in VSCode output panel
- Verify all dependencies are installed
- Run test_server.py to diagnose issues

### Debug Mode
Enable verbose logging by setting in .env:
```env
LOG_LEVEL=DEBUG
VERBOSE_LOGGING=true
```

### Check Server Status
```bash
# View server logs
python -m rag_mcp_server.main 2>&1 | tee server.log

# Check Pinecone index status
python -c "
from rag_mcp_server.services.pinecone_service import PineconeService
import asyncio
async def check():
    try:
        service = PineconeService()
        stats = await service.get_index_stats()
        print(f'Index stats: {stats}')
    except Exception as e:
        print(f'Error: {e}')
asyncio.run(check())
"
```

## 🔄 Next Steps

1. **Upload your first PDF**: Use the upload_pdf tool with a sample document
2. **Test search functionality**: Try various search queries
3. **Experiment with different chunking strategies**: Use sentence_aware or paragraph_aware
4. **Integrate with your workflow**: Use the tools in VSCode agent mode for document analysis

## 📚 Additional Resources

- [MCP Documentation](https://modelcontextprotocol.io/)
- [VSCode MCP Guide](https://code.visualstudio.com/docs/copilot/chat/mcp-servers)
- [Google Gemini API](https://ai.google.dev/gemini-api/docs)
- [Pinecone Documentation](https://docs.pinecone.io/)

---

**Your RAG MCP Server is now ready to use! 🎉**
