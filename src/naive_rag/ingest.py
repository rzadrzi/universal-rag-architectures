# src/naive_rag/ingest.py

import os
from dotenv import load_dotenv

# Correct, non-deprecated import for TextLoader
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Embedding model
from langchain_openai import OpenAIEmbeddings

# Vector store
from langchain_qdrant import Qdrant
import qdrant_client

print("Loading environment variables...")
load_dotenv()


def ingest_data():
    """
    Loads data, splits it into chunks, creates embeddings,
    and stores them in Qdrant.
    """
    # --- 1. Load Documents ---
    print("Loading data...")
    loader = TextLoader("data/mars_facts.txt")
    documents = loader.load()

    # --- 2. Split Documents ---
    print("Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split the document into {len(chunks)} chunks.")

    # --- 3. Create Embeddings & Store in Qdrant ---
    print("Creating embeddings and storing in Qdrant...")
    
    # Get the Qdrant URL from environment variables
    qdrant_url = os.getenv("QDRANT_URL")
    if not qdrant_url:
        raise ValueError("QDRANT_URL environment variable not set.")

    # Create the embedding model instance
    embeddings = OpenAIEmbeddings()

    # Use LangChain's Qdrant helper to create the vector store.
    # This will automatically create embeddings for our chunks and upload them.
    qdrant_vector_store = Qdrant.from_documents(
        chunks,
        embeddings,
        url=qdrant_url,
        prefer_grpc=False, # Set to False for simpler local setup
        collection_name="mars_facts",
    )

    print("--- Ingestion Complete ---")
    print(f"Successfully stored {len(chunks)} chunks in the 'mars_facts' collection in Qdrant.")

# This block ensures the code runs only when the script is executed directly
if __name__ == "__main__":
    ingest_data()