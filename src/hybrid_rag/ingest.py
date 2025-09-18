# src/hybrid_rag/ingest.py

import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings

# Import the vector store clients
from langchain_qdrant import Qdrant
from langchain_community.vectorstores import ElasticsearchStore

print("Loading environment variables...")
load_dotenv()

def ingest_for_hybrid_search():
    """
    Loads data and ingests it into both Qdrant and Elasticsearch.
    """
    # --- 1. Load and Chunk Data ---
    print("Loading and chunking data...")
    loader = TextLoader("data/mars_facts.txt")
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    print(f"Split document into {len(chunks)} chunks.")

    # --- 2. Ingest into Elasticsearch (for Keyword Search) ---
    print("Ingesting into Elasticsearch...")
    es_url = os.getenv("ELASTICSEARCH_URL")
    if not es_url:
        raise ValueError("ELASTICSEARCH_URL not set.")
        
    # ElasticsearchStore handles creating the index and ingesting the documents
    ElasticsearchStore.from_documents(
        chunks,
        OpenAIEmbeddings(), # We need embeddings for the vector part if we want it later
        es_url=es_url,
        index_name="mars_facts_hybrid",
    )
    print("Elasticsearch ingestion complete.")

    # --- 3. Ingest into Qdrant (for Vector Search) ---
    print("Ingesting into Qdrant...")
    qdrant_url = os.getenv("QDRANT_URL")
    if not qdrant_url:
        raise ValueError("QDRANT_URL not set.")

    Qdrant.from_documents(
        chunks,
        OpenAIEmbeddings(),
        url=qdrant_url,
        prefer_grpc=False,
        collection_name="mars_facts_hybrid",
    )
    print("Qdrant ingestion complete.")
    print("\n--- Hybrid Ingestion Finished ---")


if __name__ == "__main__":
    ingest_for_hybrid_search()