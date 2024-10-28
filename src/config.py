import yaml
from pathlib import Path
from rich.console import Console

class Config:
    def __init__(self):
        self.console = Console()
        self._load_config()

    def _load_config(self):
        """Load configuration from config.yaml"""
        config_path = Path(__file__).parent.parent / "config.yaml"
        try:
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)
        except Exception as e:
            self.console.print(f"[red]Error loading config: {str(e)}[/red]")
            self.console.print("[yellow]Using default configuration...[/yellow]")
            config = self._default_config()

        for key, value in config.items():
            setattr(self, key, value)

    def _default_config(self):
        return {
            "embedding_model": "nomic-embed-text",
            "llm_model": "llama3.2",
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