#!/usr/bin/env python3
import sys

# Ensure pysqlite3 compatibility
__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

from src.config import Config
from src.database import DatabaseManager
from src.session import InteractiveSession
from src.rag import RAGSystem
from src.cli import CLI
from rich.console import Console

def main():
    # Initialize components
    config = Config()
    console = Console()
    db_manager = DatabaseManager(config)
    rag_system = RAGSystem(config, db_manager)
    cli = CLI(config)
    
    # Parse arguments
    args = cli.parse_args()

    # Handle database population
    if args.populate_database:
        try:
            db = db_manager.reset_database()
            
            with console.status("[bold green]Loading documents..."):
                documents = db_manager.load_documents(args.data_path)
                if not documents:
                    console.print(f"[yellow]No markdown files found in: {args.data_path}[/yellow]")
                    return

            with console.status("[bold green]Processing documents..."):
                chunks = db_manager.process_documents(documents)
                
            with console.status("[bold green]Populating database..."):
                chunk_ids = [chunk.metadata["id"] for chunk in chunks]
                db.add_documents(chunks, ids=chunk_ids)
                
            console.print("[green]Database population completed successfully[/green]")
            return
        except Exception as e:
            console.print(f"[red]Error during database population: {str(e)}[/red]")
            return

    # Initialize session if in interactive mode
    session = InteractiveSession(args.model, config.llm_model) if args.interactive else None

    def process_query(query: str):
        with console.status("[bold green]Searching documents..."):
            results = rag_system.search_documents(query, k=args.top_k)

        if not results:
            console.print("[yellow]No relevant documents found.[/yellow]")
            return

        context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
        
        with console.status("[bold green]Generating response..."):
            response = rag_system.generate_response(context_text, query, session)

        sources = [doc.metadata.get("id", "Unknown") for doc, _score in results]
        rag_system.display_results(response, sources, not args.no_sources)
        
        if session:
            session.add_interaction(query, response, context_text)

    # Handle interactive mode or single query
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