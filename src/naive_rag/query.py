# src/naive_rag/query.py

import os
from dotenv import load_dotenv
import qdrant_client
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_qdrant import Qdrant
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Load environment variables from .env file
print("Loading environment variables...")
load_dotenv()

# --- 1. CONNECT TO THE VECTOR STORE ---
# This function will connect to our existing Qdrant collection
def get_vector_store():
    qdrant_url = os.getenv("QDRANT_URL")
    if not qdrant_url:
        raise ValueError("QDRANT_URL environment variable not set.")
    
    # We are not creating new embeddings, just using them for search
    embeddings = OpenAIEmbeddings()

    # Create a Qdrant client to connect to the server
    client = qdrant_client.QdrantClient(
        url=qdrant_url, 
        prefer_grpc=False
    )

    # Instantiate the LangChain Qdrant object to interact with the existing collection
    vector_store = Qdrant(
        client=client, 
        collection_name="mars_facts", 
        embeddings=embeddings
    )
    return vector_store

# --- 2. CREATE THE RAG CHAIN ---
def setup_rag_chain(vector_store):
    # Define the Large Language Model (LLM) we want to use
    llm = ChatOpenAI()

    # Create a "retriever" from our vector store. A retriever's job is to fetch relevant documents.
    retriever = vector_store.as_retriever()

    # Define the prompt template. This is how we structure our request to the LLM.
    template = """
    You are an assistant for question-answering tasks. 
    Use the following pieces of retrieved context to answer the question. 
    If you don't know the answer, just say that you don't know. 
    Keep the answer concise.

    Context: {context} 

    Question: {question} 

    Helpful Answer:
    """
    prompt = ChatPromptTemplate.from_template(template)

    # This is the RAG chain. It pipelines the retriever, prompt, and LLM together.
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain

# --- 3. EXECUTE THE QUERY ---
if __name__ == "__main__":
    # Our question about Mars
    question = "What is the largest volcano on Mars?"
    
    print(f"Querying with question: '{question}'")

    # Connect to the vector store
    vector_store = get_vector_store()

    # Set up and run the RAG chain
    rag_chain = setup_rag_chain(vector_store)
    answer = rag_chain.invoke(question)

    print("\n--- Answer ---")
    print(answer)
    print("--------------")