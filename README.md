# Query Data API

This project is a Flask-based API that utilizes a Chroma vector store and the Ollama language model to answer questions based on a given context. 
The API receives a query, retrieves relevant context from a Chroma database, and generates an answer using the Ollama model.

## Prerequisites

- Python 3.10
- Flask
- `pysqlite3`
- `langchain_community`
- `get_embedding_function` (Ensure you have the appropriate script/module)

## Yükleme

1. Projeyi kopyala:

```bash
git clone https://github.com/your-repo/query-data-api.git
cd query-data-api
```

2. Gerekli kütüphaneleri yükle:
```bash
pip install Flask
pip install pysqlite3
pip install langchain_community
```

## Kullanım
### Flask Server Başlat
1. Sunucuyu başlat:
```bash
python server.py
```

2. Örnek bir sorgu yap:
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

### Veritabanını Doldur

1. `pdf` dosyalarını `data` klasörüne yerleştir.
2. Veritabanını doldur:

```bash
python populate_database.py
```






