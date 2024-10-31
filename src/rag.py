from typing import List, Tuple
from rich.console import Console
from rich.panel import Panel
from langchain.prompts import ChatPromptTemplate

from langchain_ollama import OllamaLLM
from langchain.schema.document import Document

class RAGSystem:
    def __init__(self, config, db_manager):
        self.config = config
        self.db_manager = db_manager
        self.console = Console()

    def search_documents(self, query: str, k: int = None) -> List[Tuple[Document, float]]:
        """Search the database for relevant documents."""
        try:
            k = k or self.config.default_top_k
            db = self.db_manager.setup_database()
            return db.similarity_search_with_score(query, k=k)
        except Exception as e:
            self.console.print(f"[red]Error searching documents: {str(e)}[/red]")
            return []

    def generate_response(self, context: str, query: str, session=None) -> str:
        """Generate a response using the LLM."""
        try:
            prompt_template = ChatPromptTemplate.from_template(self.config.prompt_template)
            
            history_text = session.get_history_text() if session else "No previous conversation."
            
            prompt = prompt_template.format(
                context=context,
                question=query,
                history=history_text
            )
            
            model = session.model if session else OllamaLLM(model=self.config.llm_model)
            return model.invoke(prompt)
        except Exception as e:
            self.console.print(f"[red]Error generating response: {str(e)}[/red]")
            return ""

    def display_results(self, response: str, sources: List[str], show_sources: bool = True):
        """Display the results in a formatted way."""
        self.console.print(Panel(response, title="Response", border_style="green"))
        
        if show_sources:
            source_text = "\n".join([f"- {source}" for source in sources])
            self.console.print(Panel(source_text, title="Sources", border_style="blue"))
