# RAG MCP Demo - Retrieval-Augmented Generation with Model Context Protocol

A comprehensive implementation of a RAG (Retrieval-Augmented Generation) system using the Model Context Protocol (MCP) for seamless integration with VSCode and other MCP-compatible clients.

## 🌟 **Features**

### **Core Capabilities**
- **PDF Document Ingestion**: Upload and process PDF documents into a vector database
- **Semantic Search**: Query documents using natural language with vector similarity search
- **Document Retrieval**: Fetch relevant document chunks with AI-generated responses
- **VSCode Integration**: Native MCP client support for seamless developer experience

### **Technology Stack**
- **MCP Server**: Python with FastMCP framework
- **Vector Database**: Pinecone v7+ for scalable document storage and retrieval
- **AI Models**: Google Gemini (embedding-001 for embeddings, 2.0-flash for responses)
- **PDF Processing**: pdfplumber for robust text extraction
- **Client Integration**: VSCode with built-in MCP support

## 🚀 **Quick Start**

### **Prerequisites**
- Python 3.10 or higher
- VSCode (latest version)
- Google AI API key ([Get one here](https://ai.google.dev/))
- Pinecone account and API key ([Sign up here](https://www.pinecone.io/))

### **Installation**

1. **Clone and Setup Environment**
   ```bash
   git clone <repository-url>
   cd RAG_MCP_demo
   
   # Create virtual environment using uv
   uv venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   
   # Install dependencies
   uv pip install -r requirements.txt
   ```

2. **Configure Environment Variables**
   ```bash
   cp .env.example .env
   # Edit .env file with your API keys
   ```

3. **Initialize Pinecone Index**
   ```bash
   python scripts/setup_environment.py
   ```

4. **Test MCP Server**
   ```bash
   python -m rag_mcp_server.main
   ```

5. **Configure VSCode Integration**
   ```bash
   # Copy MCP configuration to your workspace
   cp vscode_integration/.vscode/mcp.json .vscode/
   ```

## 🛠️ **Usage**

### **1. PDF Document Ingestion**
```python
# In VSCode with agent mode enabled
"Upload and process the PDF document at /path/to/document.pdf with ID 'research-paper-1'"
```

### **2. Semantic Search**
```python
# Search for relevant documents
"Search for documents related to 'machine learning algorithms'"
```

### **3. Document Retrieval with AI Response**
```python
# Get AI-generated answers based on document content
"What are the main findings about neural networks in the uploaded research papers?"
```

## 📁 **Project Structure**

```
RAG_MCP_demo/
├── rag_mcp_server/           # Core MCP server implementation
│   ├── main.py              # Server entry point
│   ├── config.py            # Configuration management
│   ├── tools/               # MCP tools (PDF ingestion, search, retrieval)
│   ├── services/            # Business logic (Pinecone, Gemini, PDF processing)
│   └── utils/               # Utility functions
├── vscode_integration/       # VSCode MCP client configuration
├── tests/                   # Comprehensive test suite
├── docs/                    # Documentation
├── examples/                # Example files and usage demos
└── scripts/                 # Setup and utility scripts
```

## 🔧 **Configuration**

### **Environment Variables**
Key configuration options in `.env`:

- `GOOGLE_API_KEY`: Your Google AI API key
- `PINECONE_API_KEY`: Your Pinecone API key
- `PINECONE_ENVIRONMENT`: Pinecone environment (e.g., us-west1-gcp)
- `PINECONE_INDEX_NAME`: Name of your Pinecone index

### **VSCode MCP Configuration**
The `.vscode/mcp.json` file configures VSCode to connect to your RAG MCP server:

```json
{
  "servers": {
    "rag-mcp-server": {
      "type": "stdio",
      "command": "python",
      "args": ["-m", "rag_mcp_server.main"],
      "env": { /* environment variables */ }
    }
  }
}
```

## 🧪 **Testing**

Run the comprehensive test suite:

```bash
# Run all tests
pytest

# Run specific test categories
pytest tests/test_tools/          # Tool-specific tests
pytest tests/test_services/       # Service layer tests
pytest tests/integration/         # Integration tests
```

## 📚 **Documentation**

- [Setup Guide](docs/setup_guide.md) - Detailed installation and configuration
- [API Reference](docs/api_reference.md) - MCP tools and methods
- [Architecture Overview](docs/architecture.md) - System design and components
- [Troubleshooting](docs/troubleshooting.md) - Common issues and solutions

## 🤝 **Contributing**

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## 📄 **License**

This project is licensed under the MIT License - see the LICENSE file for details.

## 🔗 **Related Resources**

- [Model Context Protocol Documentation](https://modelcontextprotocol.io/)
- [VSCode MCP Integration Guide](https://code.visualstudio.com/docs/copilot/chat/mcp-servers)
- [Google Gemini API Documentation](https://ai.google.dev/gemini-api/docs)
- [Pinecone Documentation](https://docs.pinecone.io/)

---

**Built with ❤️ using the Model Context Protocol**
