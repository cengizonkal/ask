import argparse
import os
import shutil
from pathlib import Path
from typing import List

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings.ollama import OllamaEmbeddings
import yaml


CHROMA_PATH = "chroma"

# Ensure pysqlite3 compatibility
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')


def load_documents(data_path: str) -> List[Document]:
    """Load all markdown files from the specified data directory."""
    documents = []
    for file_path in Path(data_path).glob("**/*.md"):
        try:
            loader = TextLoader(str(file_path))
            documents.extend(loader.load())
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    return documents


def split_documents(documents: List[Document]) -> List[Document]:
    """Split documents into chunks with optimal size for embedding."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        length_function=len,
        is_separator_regex=False,
    )
    return splitter.split_documents(documents)


def calculate_chunk_ids(chunks: List[Document]) -> List[Document]:
    """Generate unique IDs for each chunk based on source file and position."""
    chunk_mapping = {}
    
    for chunk in chunks:
        source = chunk.metadata.get("source")
        if source not in chunk_mapping:
            chunk_mapping[source] = 0
        
        # Create a more efficient ID format: filename:chunk_index
        chunk_id = f"{source}:{chunk_mapping[source]}"
        chunk.metadata["id"] = chunk_id
        chunk_mapping[source] += 1
    
    return chunks


def reset_and_initialize_db() -> Chroma:
    """Reset the database and initialize a new one."""
    print("Resetting database...")
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)
    
    return Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=OllamaEmbeddings(model="nomic-embed-text")
    )


def populate_chroma(chunks: List[Document], db: Chroma) -> None:
    """Populate the Chroma database with documents."""
    print(f"Adding {len(chunks)} documents to fresh database")
    chunk_ids = [chunk.metadata["id"] for chunk in chunks]
    db.add_documents(chunks, ids=chunk_ids)


def main():
    parser = argparse.ArgumentParser(description="Populate Chroma database with markdown files")
    parser.add_argument("--data-path", type=str, required=True, help="Path to the data directory containing markdown files")
    args = parser.parse_args()

    if not os.path.exists(args.data_path):
        print(f"Error: Data path '{args.data_path}' does not exist")
        return

    try:
        # Initialize fresh database
        db = reset_and_initialize_db()

        # Process documents
        documents = load_documents(args.data_path)
        if not documents:
            print(f"No markdown files found in the directory: {args.data_path}")
            return

        chunks = split_documents(documents)
        chunks = calculate_chunk_ids(chunks)
        populate_chroma(chunks, db)
        print("Database population completed successfully")

    except Exception as e:
        print(f"Error during database population: {e}")
        raise


if __name__ == "__main__":
    main()