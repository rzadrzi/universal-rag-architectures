# src/rerank_rag/query.py

import os
from dotenv import load_dotenv
from operator import itemgetter # <-- ADD THIS IMPORT

# LLM and RAG components
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_qdrant import Qdrant
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Reranker components
from langchain.retrievers import ContextualCompressionRetriever
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain.retrievers.document_compressors import CrossEncoderReranker

# Client for connecting to Qdrant
import qdrant_client

print("Loading environment variables...")
load_dotenv()

# --- 1. SET UP THE RERANKING RETRIEVER ---
def get_reranking_retriever():
    qdrant_url = os.getenv("QDRANT_URL")
    if not qdrant_url:
        raise ValueError("QDRANT_URL environment variable not set.")
    
    embeddings = OpenAIEmbeddings()
    client = qdrant_client.QdrantClient(url=qdrant_url, prefer_grpc=False)
    vector_store = Qdrant(client=client, collection_name="mars_facts_rerank", embeddings=embeddings)
    base_retriever = vector_store.as_retriever(search_kwargs={"k": 10})

    # --- RERANKER SETUP ---
    model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-base")
    compressor = CrossEncoderReranker(model=model, top_n=3)
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=base_retriever
    )
    
    return compression_retriever

# --- 2. SET UP THE RAG CHAIN ---
def setup_rag_chain(retriever):
    llm = ChatOpenAI()
    
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
    
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # --- THIS IS THE CORRECTED CHAIN ---
    chain = (
        # This part extracts the question, passes it to the retriever, and then passes the original question through.
        {"context": itemgetter("question") | retriever | format_docs, "question": itemgetter("question")}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain

# --- 3. EXECUTE THE QUERY ---
if __name__ == "__main__":
    question = "What is the second smallest planet and what is its nickname?"
    
    print(f"Querying with question: '{question}'")

    reranking_retriever = get_reranking_retriever()
    rag_chain = setup_rag_chain(reranking_retriever)
    answer = rag_chain.invoke({"question": question})

    print("\n--- Answer ---")
    print(answer)
    print("--------------")