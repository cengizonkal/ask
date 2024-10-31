import os
import shutil
from typing import List
from pathlib import Path
from rich.console import Console
from langchain_chroma import Chroma

from langchain.schema.document import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader

from langchain_ollama import OllamaEmbeddings


class DatabaseManager:
    def __init__(self, config):
        self.config = config
        self.console = Console()
        self.embedding_function = OllamaEmbeddings(model=config.embedding_model)

    def setup_database(self) -> Chroma:
        """Initialize and return the Chroma database with embeddings."""
        try:
            return Chroma(
                persist_directory=self.config.chroma_path,
                embedding_function=self.embedding_function
            )
        except Exception as e:
            self.console.print(f"[red]Error setting up database: {str(e)}[/red]")
            raise

    def reset_database(self) -> Chroma:
        """Reset and initialize a new database."""
        self.console.print("[yellow]Resetting database...[/yellow]")
        if os.path.exists(self.config.chroma_path):
            shutil.rmtree(self.config.chroma_path)
        return self.setup_database()

    def load_documents(self, data_path: str) -> List[Document]:
        """Load all markdown files from the specified data directory."""
        documents = []
        for file_path in Path(data_path).glob("**/*.md"):
            try:
                loader = TextLoader(str(file_path))
                documents.extend(loader.load())
            except Exception as e:
                self.console.print(f"[red]Error loading {file_path}: {str(e)}[/red]")
        return documents

    def process_documents(self, documents: List[Document]) -> List[Document]:
        """Process documents by splitting and adding chunk IDs."""
        # Split documents
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=80,
            length_function=len,
            is_separator_regex=False,
        )
        chunks = splitter.split_documents(documents)
        
        # Add chunk IDs
        chunk_mapping = {}
        for chunk in chunks:
            source = chunk.metadata.get("source")
            if source not in chunk_mapping:
                chunk_mapping[source] = 0
            chunk.metadata["id"] = f"{source}:{chunk_mapping[source]}"
            chunk_mapping[source] += 1
        
        return chunks