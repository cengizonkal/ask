#!/usr/bin/env python3
import subprocess
import sys
import os
from pathlib import Path

def check_python_version():
    """Check if Python version is 3.8 or higher."""
    if sys.version_info < (3, 8):
        print("Error: Python 3.8 or higher is required.")
        sys.exit(1)

def install_requirements():
    """Install required Python packages."""
    requirements = [
        "pysqlite3-binary",  # For Chrome compatibility
        "langchain",
        "langchain-community",
        "langchain-chroma",
        "chromadb",
        "rich",  # For console formatting
        "PyYAML",  # For config file handling
        "ollama",  # For LLM integration
    ]
    
    print("Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
        for package in requirements:
            print(f"Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    except subprocess.CalledProcessError as e:
        print(f"Error installing packages: {str(e)}")
        sys.exit(1)

def create_config():
    """Create default config.yaml if it doesn't exist."""
    config_content = """
# Model Configuration
embedding_model: "nomic-embed-text"
llm_model: "llama3.2"

# Database Configuration
chroma_path: "chroma"
default_top_k: 5

# Prompt Template
prompt_template: |
    Previous conversation:
    {history}

    Current context:
    --- content ---
    {context}
    --- content ---

    Please answer the following question using the provided context.
    If referring to previous conversation, still ground your answer in the current context.
    Question: {question}
"""
    
    config_path = Path(__file__).parent / "config.yaml"
    if not config_path.exists():
        print("Creating default config.yaml...")
        try:
            with open(config_path, 'w') as f:
                f.write(config_content.lstrip())
        except Exception as e:
            print(f"Error creating config file: {str(e)}")
            sys.exit(1)

def check_ollama():
    """Check if Ollama is installed and running."""
    print("Checking Ollama installation...")
    try:
        result = subprocess.run(['which', 'ollama'], capture_output=True, text=True)
        if result.returncode != 0:
            print("Warning: Ollama is not installed. Please install Ollama from https://ollama.ai")
            print("After installing Ollama, run: 'ollama pull llama3.2' to download the default model")
            return False
        return True
    except Exception:
        print("Warning: Could not check Ollama installation")
        return False

def main():
    """Main installation function."""
    print("Starting installation...")
    
    # Check Python version
    check_python_version()
    
    # Install required packages
    install_requirements()
    
    # Create config file
    create_config()
    
    # Check Ollama installation
    check_ollama()
    
    print("\nInstallation completed successfully!")
    print("\nTo use the RAG CLI tool:")
    print("1. Make sure Ollama is running ('ollama serve' in a separate terminal)")
    print("2. Pull the required models:")
    print("   - ollama pull llama3.2")
    print("   - ollama pull nomic-embed-text")
    print("3. Populate database:")
    print("   - Populate database: python populate_database.py --data-path=/your/data/path")
    print("4. Run the tool:")
    print("   - Basic usage: python ask.py 'your question'")
    print("   - Interactive mode: python ask.py --interactive")

if __name__ == "__main__":
    main()