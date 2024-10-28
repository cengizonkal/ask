from typing import List, Dict
from langchain_community.llms.ollama import Ollama

class InteractiveSession:
    def __init__(self, model_name: str = None, config_model: str = "llama2"):
        self.model_name = model_name or config_model
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
        for i, interaction in enumerate(self.history[-3:], 1):
            history_text.extend([
                f"Q{i}: {interaction['query']}",
                f"A{i}: {interaction['response']}",
                ""
            ])
        return "\n".join(history_text)