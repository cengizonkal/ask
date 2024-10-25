import argparse
import sys
from typing import List, Tuple, Dict
import yaml
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich import print as rprint
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain.schema.document import Document

# Load config
def load_config():
    """Load configuration from config.yaml"""
    config_path = Path(__file__).parent / "config.yaml"
    try:
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)
    except Exception as e:
        console.print(f"[red]Error loading config: {str(e)}[/red]")
        console.print("[yellow]Using default configuration...[/yellow]")
        return {
            "embedding_model": "nomic-embed-text",
            "llm_model": "llama2",
            "chroma_path": "chroma",
            "default_top_k": 5,
            "prompt_template": "\n".join([
                "Previous conversation:",
                "{history}",
                "",
                "Current context:",
                "--- content ---",
                "{context}",
                "--- content ---",
                "",
                "Please answer the following question using the provided context.",
                "If referring to previous conversation, still ground your answer in the current context.",
                "Question: {question}"
            ])
        }

# Initialize console and config
console = Console()
config = load_config()

# Ensure pysqlite3 compatibility
__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

class InteractiveSession:
    def __init__(self, model_name: str = None):
        self.model_name = model_name or config["llm_model"]
        self.history: List[Dict] = []
        self.model = Ollama(model=self.model_name)
        
    def add_interaction(self, query: str, response: str, context: str = None):
        """Add an interaction to the session history."""
        self.history.append({
            "query": query,
            "response": response,
            "context": context
        })
    
    def get_history_text(self) -> str:
        """Format history for the prompt."""
        if not self.history:
            return "No previous conversation."
        
        history_text = []
        for i, interaction in enumerate(self.history[-3:], 1):  # Only use last 3 interactions
            history_text.extend([
                f"Q{i}: {interaction['query']}",
                f"A{i}: {interaction['response']}",
                ""
            ])
        return "\n".join(history_text)

def setup_database() -> Chroma:
    """Initialize and return the Chroma database with embeddings."""
    try:
        embedding_function = OllamaEmbeddings(model=config["embedding_model"])
        return Chroma(
            persist_directory=config["chroma_path"],
            embedding_function=embedding_function
        )
    except Exception as e:
        console.print(f"[red]Error setting up database: {str(e)}[/red]")
        sys.exit(1)

def search_documents(db: Chroma, query: str, k: int = None) -> List[Tuple[Document, float]]:
    """Search the database for relevant documents."""
    try:
        k = k or config["default_top_k"]
        return db.similarity_search_with_score(query, k=k)
    except Exception as e:
        console.print(f"[red]Error searching documents: {str(e)}[/red]")
        return []

def generate_response(context: str, query: str, session: InteractiveSession = None) -> str:
    """Generate a response using the LLM."""
    try:
        prompt_template = ChatPromptTemplate.from_template(config["prompt_template"])
        
        # Include history if in a session
        history_text = session.get_history_text() if session else "No previous conversation."
        
        prompt = prompt_template.format(
            context=context,
            question=query,
            history=history_text
        )
        
        model = session.model if session else Ollama(model=config["llm_model"])
        return model.invoke(prompt)
    except Exception as e:
        console.print(f"[red]Error generating response: {str(e)}[/red]")
        return ""

def display_results(response: str, sources: List[str], show_sources: bool = True):
    """Display the results in a formatted way."""
    console.print(Panel(response, title="Response", border_style="green"))
    
    if show_sources:
        source_text = "\n".join([f"- {source}" for source in sources])
        console.print(Panel(source_text, title="Sources", border_style="blue"))

def main():
    parser = argparse.ArgumentParser(description="RAG CLI Tool")
    parser.add_argument(
        "query",
        type=str,
        nargs="?",  # Make query optional when using --interactive
        help="The question or query to process"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help=f"The Ollama model to use (default: {config['llm_model']})"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help=f"Number of documents to retrieve (default: {config['default_top_k']})"
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

    if args.interactive and not args.query:
        args.query = None
    elif not args.query:
        parser.error("query is required when not in interactive mode")

    # Initialize database
    db = setup_database()
    
    # Initialize session if in interactive mode
    session = InteractiveSession(args.model) if args.interactive else None

    def process_query(query: str):
        with console.status("[bold green]Searching documents..."):
            results = search_documents(db, query, k=args.top_k)

        if not results:
            console.print("[yellow]No relevant documents found.[/yellow]")
            return

        context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
        
        with console.status("[bold green]Generating response..."):
            response = generate_response(context_text, query, session)

        sources = [doc.metadata.get("id", "Unknown") for doc, _score in results]
        display_results(response, sources, not args.no_sources)
        
        # Add to session history if in interactive mode
        if session:
            session.add_interaction(query, response, context_text)

    if args.interactive:
        console.print("[bold]RAG Interactive Session[/bold] (Type 'exit' to quit)")
        console.print("[blue italic]Session maintains context of previous questions[/blue italic]")
        
        while True:
            try:
                query = console.input("\n[bold blue]Enter your query:[/bold blue] ")
                if query.lower() in ['exit', 'quit']:
                    break
                process_query(query)
            except KeyboardInterrupt:
                console.print("\n[yellow]Session terminated by user[/yellow]")
                break
            except Exception as e:
                console.print(f"\n[red]Error: {str(e)}[/red]")
                continue
    else:
        process_query(args.query)

if __name__ == "__main__":
    main()