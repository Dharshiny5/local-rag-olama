# Local RAG Implementation Using Ollama

## Description

This project implements a local Retrieval-Augmented Generation (RAG) system using Ollama. Four academic assignments (Artificial Intelligence, Data Mining, Human Resource Analytics and Marketing Analytics) are used as the knowledge base.

## Methodology

- Assignment PDFs are loaded and converted into text.
- The text is split into smaller chunks to preserve context.
- Each chunk is embedded using an Ollama embedding model (`nomic-embed-text`).
- Embeddings are stored in a Chroma vector database.

## Key Features

- Fully local execution using Ollama
- Retrieval-based grounding (reduces hallucination)
- Cross-document reasoning across multiple assignments
- Works offline once models are downloaded

## Tech Stack

| Component | Tool |
|---|---|
| Orchestration | [LangChain](https://www.langchain.com/) (`langchain-community`, `langchain-text-splitters`) |
| Local LLM + embeddings runtime | [Ollama](https://ollama.com/) |
| Embedding model | `nomic-embed-text` |
| Generation model | `llama3.2` |
| Vector store | [Chroma](https://www.trychroma.com/) |
| PDF parsing | `pypdf` (via `PyPDFLoader`) |

## Architecture

```
PDFs (data/)
     ↓
PyPDFLoader → raw document pages
     ↓
RecursiveCharacterTextSplitter → 500-char chunks, 100-char overlap
     ↓
OllamaEmbeddings (nomic-embed-text) → vector embeddings
     ↓
Chroma → persisted to ./chroma_db
     ↓
Retriever (top-k similarity search) ←── user question
     ↓
Prompt (context + question) → OllamaLLM (llama3.2)
     ↓
Answer (printed to terminal)
```

## Setup

### Prerequisites

- Python 3.11
- [Ollama](https://ollama.com/download) installed and running locally

### 1. Clone the repo

```bash
git clone https://github.com/<your-username>/local-rag-ollama.git
cd local-rag-ollama
```

### 2. Create a virtual environment

```bash
py -3.11 -m venv venv
venv\Scripts\Activate      # Windows
# source venv/bin/activate   # macOS/Linux
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Pull the required Ollama models

```bash
ollama pull nomic-embed-text
ollama pull llama3.2
```

### 5. Add your documents

Create a `data/` folder in the project root and place your assignment PDFs inside it. This folder is excluded from version control via `.gitignore`, so your documents stay private even though the repo is public.

### 6. Run it

```bash
python rag.py
```

You'll see the pipeline load and chunk your documents, build the vector store, and then drop into an interactive question loop.

## Sample Questions

1. Which classification or clustering methods are mentioned in the Data Mining assignment?
2. Is blockchain technology discussed in any of the assignments?
3. What type of data is analyzed in the Human Resource assignment?
4. What metrics or KPIs are mentioned in the Marketing Analytics assignment?
5. What are the main business problems discussed in the assignments?


