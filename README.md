# AI CLI Tool

This is a Python package intended for use directly from the terminal.
It is designed to read your **Markdown** documents and return information about them.


## Requirements
* Python 3
* [Ollama](https://ollama.com/download) service should be up and running

## Installation
```bash
python install.py
```

## Usage
```bash
#first populate database
python ask.py --populate_database --data-path=/path/to/documents

#then run ask.py to get information about your documents
python ask.py "what is the ip address of my router?"
```