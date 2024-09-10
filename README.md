# Query Data API

This project is a Flask-based API that utilizes a Chroma vector store and the Ollama language model to answer questions based on a given context. 
The API receives a query, retrieves relevant context from a Chroma database, and generates an answer using the Ollama model.

## Prerequisites

- Python 3.10
- Flask
- `pysqlite3`
- `langchain_community`
- `get_embedding_function` (Ensure you have the appropriate script/module)

## Installation

1. Clone the repository:

```bash
git clone https://github.com/your-repo/query-data-api.git
cd query-data-api
```

2. Install the required packages:
```bash
pip install Flask
pip install pysqlite3
pip install langchain_community
```

## Usage
### Starting the Flask Server
1. Start the Flask server:
```bash
python query_data.py
```

2. Make a GET request to the /rag endpoint with the query parameter:
```bash
curl "http://localhost:5000/rag?query=your_question_here"
```

### Populating the Chroma Database

1. Place your `PDF` documents in the data directory.
2. Populate the Chroma database:

```bash
python populate_database.py
```

Example
```bash
curl "http://localhost:5000/rag?query=What is the capital of France?"
```
Response:
```json
{
  "response": "The capital of France is Paris.",
  "sources": ["source1", "source2", "source3"]
}
```




