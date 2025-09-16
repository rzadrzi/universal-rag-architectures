# Universal RAG Architectures Playground

This project implements and compares multiple Retrieval-Augmented Generation (RAG) architectures, showcasing their strengths and weaknesses for different enterprise use cases.

## Architectures Covered
- **Naive RAG** – simple top-k retrieval
- **Retrieve-and-Rerank RAG** – higher precision with rerankers
- **Graph RAG** – knowledge graph traversal + retrieval
- **Hybrid RAG** – BM25 + vector fusion
- **Multimodal RAG** – text + image retrieval
- **Agentic RAG (Router)** – auto-select best retriever
- **Agentic Multi-Agent RAG** – coordination between multiple agents and tools

## Dataset Plan
We will use:
- **WikiMovies Dataset** (movies, actors, directors, genres) → great for GraphRAG
- **Wikipedia subset** → for Naive/Hybrid RAG
- **MS MARCO QA dataset** → for reranker comparison
- **Images from IMDB posters** → for multimodal RAG

## Tech Stack
- Python, FastAPI
- LangChain, LlamaIndex
- Qdrant (vector DB)
- Neo4j (graph DB)
- Elasticsearch (BM25 hybrid search)
- Streamlit/Gradio (demo UI)
- Docker Compose (infrastructure)

## Goals
- Provide a **playground** to test different RAG strategies
- Demonstrate **which architecture fits which use-case**
- Build a **portfolio project** for showcasing expertise to companies

## Next Steps
1. Prepare datasets (`data/` folder)
2. Implement **Naive RAG pipeline**
3. Expand into Retrieve-and-Rerank, Graph, Hybrid, Multimodal, Agentic
4. Build demo UI for comparison
