
import argparse
from typing import Optional
from rich.console import Console

class CLI:
    def __init__(self, config):
        self.config = config
        self.console = Console()

    def parse_args(self):
        parser = argparse.ArgumentParser(description="RAG CLI Tool")
        
        # Add database population argument
        parser.add_argument(
            "--populate-database",
            action="store_true",
            help="Populate the database with documents"
        )
        parser.add_argument(
            "--data-path",
            type=str,
            help="Path to the data directory containing markdown files"
        )
        
        # Query-related arguments
        parser.add_argument(
            "query",
            type=str,
            nargs="?",
            help="The question or query to process"
        )
        parser.add_argument(
            "--model",
            type=str,
            default=None,
            help=f"The Ollama model to use (default: {self.config.llm_model})"
        )
        parser.add_argument(
            "--top-k",
            type=int,
            default=None,
            help=f"Number of documents to retrieve (default: {self.config.default_top_k})"
        )
        parser.add_argument(
            "--no-sources",
            action="store_true",
            help="Hide source documents in output"
        )
        parser.add_argument(
            "--interactive",
            action="store_true",
            help="Run in interactive mode with conversation history"
        )

        args = parser.parse_args()

        # Validate arguments
        if args.populate_database and not args.data_path:
            parser.error("--data-path is required when using --populate-database")
        elif not args.populate_database and not args.interactive and not args.query:
            parser.error("query is required when not in interactive mode")

        return args