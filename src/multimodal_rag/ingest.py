# src/multimodal_rag/ingest.py

import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient, models
from langchain_openai import OpenAIEmbeddings

print("Loading environment variables...")
load_dotenv()

# --- 1. Initialize Clients ---
qdrant_url = os.getenv("QDRANT_URL")
if not qdrant_url:
    raise ValueError("QDRANT_URL is not set.")

qdrant_client = QdrantClient(url=qdrant_url)
collection_name = "multimodal_movies"
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

# --- 2. Set up Qdrant Collection ---
print(f"Setting up Qdrant collection: {collection_name}")
vector_size = len(embeddings.embed_query("test"))

# Check if the collection already exists
collection_exists = qdrant_client.collection_exists(collection_name=collection_name)

if collection_exists:
    print("Collection already exists. Deleting and recreating.")
    qdrant_client.delete_collection(collection_name=collection_name)

# Create a new collection
qdrant_client.create_collection(
    collection_name=collection_name,
    vectors_config=models.VectorParams(
        size=vector_size,
        distance=models.Distance.COSINE,
    ),
)
print("Collection created successfully.")


# --- 3. Process and Ingest Data ---
def process_and_ingest_data(image_dir="data/images"):
    print(f"\nProcessing data from directory: {image_dir}")
    image_files = [f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    if not image_files:
        print(f"No image files found in '{image_dir}'. Exiting.")
        return

    points_to_upload = []
    point_id = 0

    for image_file in image_files:
        image_path = os.path.join(image_dir, image_file)
        description = f"Movie poster for {os.path.splitext(image_file)[0]}"

        print(f"  - Processing: {description}")
        vector = embeddings.embed_query(description)

        points_to_upload.append(
            models.PointStruct(
                id=point_id,
                vector=vector,
                payload={
                    "description": description,
                    "image_path": image_path
                },
            )
        )
        point_id += 1

    if points_to_upload:
        print(f"\nUploading {len(points_to_upload)} data points to Qdrant...")
        qdrant_client.upsert(
            collection_name=collection_name,
            points=points_to_upload,
            wait=True,
        )
        print("Upload complete.")

if __name__ == "__main__":
    process_and_ingest_data()