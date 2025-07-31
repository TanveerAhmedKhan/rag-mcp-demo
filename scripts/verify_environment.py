#!/usr/bin/env python3
"""
Comprehensive environment verification script for RAG MCP Server.
"""

import sys
import os
import subprocess
from pathlib import Path

def print_section(title):
    print(f"\n{'='*50}")
    print(f" {title}")
    print(f"{'='*50}")

def check_python_environment():
    print_section("PYTHON ENVIRONMENT CHECK")
    
    print(f"Python executable: {sys.executable}")
    print(f"Python version: {sys.version}")
    print(f"Python path: {sys.path[0]}")
    
    # Check if we're in a virtual environment
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("SUCCESS: Virtual environment detected")
        print(f"   Virtual env path: {sys.prefix}")
    else:
        print("WARNING: No virtual environment detected")
        return False
    
    return True

def check_required_packages():
    print_section("PACKAGE DEPENDENCY CHECK")
    
    required_packages = [
        'mcp',
        'pinecone',
        'google.generativeai',
        'pdfplumber',
        'dotenv',  # python-dotenv imports as 'dotenv'
        'fastapi',
        'uvicorn'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"SUCCESS: {package}")
        except ImportError:
            print(f"ERROR: {package} - NOT FOUND")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\nWARNING: Missing packages: {', '.join(missing_packages)}")
        return False

    print("\nSUCCESS: All required packages found")
    return True

def check_mcp_specific():
    print_section("MCP SPECIFIC CHECK")
    
    try:
        from mcp.server.fastmcp import FastMCP
        print("SUCCESS: FastMCP import successful")

        # Try to create a FastMCP instance
        test_server = FastMCP("test-server")
        print("SUCCESS: FastMCP instantiation successful")
        
        return True
    except Exception as e:
        print(f"ERROR: MCP check failed: {str(e)}")
        return False

def check_project_structure():
    print_section("PROJECT STRUCTURE CHECK")
    
    required_files = [
        'rag_mcp_server/__init__.py',
        'rag_mcp_server/main.py',
        'rag_mcp_server/config.py',
        'rag_mcp_server/services/pinecone_service.py',
        'rag_mcp_server/services/gemini_service.py',
        'rag_mcp_server/tools/pdf_ingestion.py',
        'requirements.txt',
        '.env'
    ]
    
    missing_files = []
    
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"SUCCESS: {file_path}")
        else:
            print(f"ERROR: {file_path} - NOT FOUND")
            missing_files.append(file_path)
    
    if missing_files:
        print(f"\nWARNING: Missing files: {', '.join(missing_files)}")
        return False
    
    return True

def check_configuration():
    print_section("CONFIGURATION CHECK")
    
    try:
        from rag_mcp_server.config import Config
        
        # Check if config loads
        config_valid = Config.validate()
        if config_valid:
            print("SUCCESS: Configuration validation passed")
        else:
            print("ERROR: Configuration validation failed")
            return False

        # Print config summary
        summary = Config.get_summary()
        print(f"Config summary: {summary}")
        
        return True
    except Exception as e:
        print(f"ERROR: Configuration check failed: {str(e)}")
        return False

def test_server_import():
    print_section("SERVER IMPORT TEST")
    
    try:
        import rag_mcp_server.main
        print("SUCCESS: Main server module import successful")

        # Try to access the mcp instance
        if hasattr(rag_mcp_server.main, 'mcp'):
            print("SUCCESS: MCP server instance found")
        else:
            print("ERROR: MCP server instance not found")
            return False

        return True
    except Exception as e:
        print(f"ERROR: Server import failed: {str(e)}")
        return False

def run_installation_fix():
    print_section("ATTEMPTING DEPENDENCY FIX")
    
    try:
        print("Running: uv pip install -r requirements.txt")
        result = subprocess.run([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ], capture_output=True, text=True, cwd=os.getcwd())
        
        if result.returncode == 0:
            print("SUCCESS: Dependencies installed successfully")
            return True
        else:
            print(f"ERROR: Installation failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"ERROR: Installation error: {str(e)}")
        return False

def main():
    print("RAG MCP Server Environment Verification")
    print(f"Working directory: {os.getcwd()}")

    # Add current directory to Python path for imports
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    
    checks = [
        ("Python Environment", check_python_environment),
        ("Project Structure", check_project_structure),
        ("Package Dependencies", check_required_packages),
        ("MCP Specific", check_mcp_specific),
        ("Configuration", check_configuration),
        ("Server Import", test_server_import)
    ]
    
    failed_checks = []
    
    for check_name, check_func in checks:
        try:
            if not check_func():
                failed_checks.append(check_name)
        except Exception as e:
            print(f"ERROR: {check_name} check crashed: {str(e)}")
            failed_checks.append(check_name)
    
    print_section("SUMMARY")
    
    if not failed_checks:
        print("SUCCESS: All checks passed! Your environment is ready.")
        print("\nNext steps:")
        print("1. Restart VSCode completely")
        print("2. Try starting the MCP server in VSCode")
        return True
    else:
        print(f"FAILED checks: {', '.join(failed_checks)}")

        if "Package Dependencies" in failed_checks:
            print("\nAttempting to fix dependencies...")
            if run_installation_fix():
                print("SUCCESS: Dependencies fixed. Please run this script again.")
            else:
                print("ERROR: Could not fix dependencies automatically.")

        print("\nTroubleshooting suggestions:")
        print("1. Ensure you're in the correct virtual environment")
        print("2. Run: uv pip install -r requirements.txt")
        print("3. Check your .env file has the required API keys")
        print("4. Verify Python version is 3.10+")

        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
