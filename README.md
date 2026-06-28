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

## Sample Questions

1. Which classification or clustering methods are mentioned in the Data Mining assignment?
2. Is blockchain technology discussed in any of the assignments?
3. What type of data is analyzed in the Human Resource assignment?
4. What metrics or KPIs are mentioned in the Marketing Analytics assignment?
5. What are the main business problems discussed in the assignments?


