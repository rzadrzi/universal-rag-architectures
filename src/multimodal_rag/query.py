# src/multimodal_rag/query.py

import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from langchain_openai import OpenAIEmbeddings
from PIL import Image

print("Loading environment variables...")
load_dotenv()

# --- 1. Initialize Clients ---
qdrant_url = os.getenv("QDRANT_URL")
if not qdrant_url:
    raise ValueError("QDRANT_URL is not set.")

qdrant_client = QdrantClient(url=qdrant_url)
collection_name = "multimodal_movies"
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

# --- 2. Define the Query Function ---
def query_multimodal_database(query: str):
    """
    Takes a text query, finds the most relevant image, and displays it.
    """
    print(f"Searching for images related to: '{query}'")
    
    # Create an embedding for the text query
    query_vector = embeddings.embed_query(query)
    
    # Perform the search in Qdrant
    search_results = qdrant_client.search(
        collection_name=collection_name,
        query_vector=query_vector,
        limit=1 # We only want the single best match
    )
    
    # The result is a list of ScoredPoint objects
    top_result = search_results[0]
    
    print(f"\nFound match with score: {top_result.score:.4f}")
    
    # Extract the image path and description from the payload
    image_path = top_result.payload.get("image_path")
    description = top_result.payload.get("description")
    
    print(f"Description: '{description}'")
    print(f"Image Path: '{image_path}'")
    
    # --- 3. Display the Image ---
    if image_path and os.path.exists(image_path):
        print("\nDisplaying the retrieved image...")
        img = Image.open(image_path)
        img.show()
    else:
        print(f"\nCould not find or open the image at path: {image_path}")


if __name__ == "__main__":
    # A text query that is conceptually similar to one of our posters
    user_query = "a movie about a man in a black coat in a green digital world"
    query_multimodal_database(user_query)