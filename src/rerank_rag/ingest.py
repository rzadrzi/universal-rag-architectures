# src/rerank_rag/ingest.py

import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import Qdrant

print("Loading environment variables...")
load_dotenv()

def ingest_for_rerank():
    """
    Loads data and prepares it using semantic chunking for the rerank RAG system.
    """
    # Load Documents
    print("Loading data...")
    loader = TextLoader("data/mars_facts.txt")
    documents = loader.load()

    # Split Documents with Semantic Chunker
    print("Splitting documents with Semantic Chunker...")
    embeddings = OpenAIEmbeddings()
    text_splitter = SemanticChunker(
        embeddings, breakpoint_threshold_type="percentile"
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split document into {len(chunks)} semantic chunks.")

    # Store in a new Qdrant collection
    print("Storing chunks in Qdrant for reranking...")
    qdrant_url = os.getenv("QDRANT_URL")
    if not qdrant_url:
        raise ValueError("QDRANT_URL environment variable not set.")

    Qdrant.from_documents(
        chunks,
        embeddings,
        url=qdrant_url,
        prefer_grpc=False,
        # A dedicated collection for this pipeline
        collection_name="mars_facts_rerank",
    )
    print("--- Ingestion for Rerank RAG Complete ---")

if __name__ == "__main__":
    ingest_for_rerank()