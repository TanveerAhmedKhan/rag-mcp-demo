@echo off
REM Batch script to start RAG MCP Server with correct Python environment

REM Change to project directory
cd /d "%~dp0\.."

REM Set Python path to include current directory
set PYTHONPATH=%CD%;%PYTHONPATH%

REM Use virtual environment Python directly
set PYTHON_EXE=%CD%\.venv\Scripts\python.exe

REM Verify environment
echo Starting RAG MCP Server... >&2
echo Python executable: %PYTHON_EXE% >&2
echo Working directory: %CD% >&2
echo PYTHONPATH: %PYTHONPATH% >&2
echo. >&2

REM Check if MCP module is available
"%PYTHON_EXE%" -c "import mcp; print('MCP module found')" 2>nul
if errorlevel 1 (
    echo ERROR: MCP module not found >&2
    exit /b 1
)

REM Check if rag_mcp_server module is available
"%PYTHON_EXE%" -c "import rag_mcp_server; print('rag_mcp_server module found')" 2>nul
if errorlevel 1 (
    echo ERROR: rag_mcp_server module not found >&2
    exit /b 1
)

REM Start the MCP server
"%PYTHON_EXE%" -m rag_mcp_server.main
