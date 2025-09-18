# src/hybrid_rag/query.py

import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import ElasticsearchStore
from langchain_qdrant import Qdrant
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from operator import itemgetter
from langchain.retrievers import EnsembleRetriever
import qdrant_client

print("Loading environment variables...")
load_dotenv()

def get_hybrid_retriever():
    embeddings = OpenAIEmbeddings()
    
    # --- 1. Set up Elasticsearch Retriever (Keyword Search) ---
    es_url = os.getenv("ELASTICSEARCH_URL")
    
    # THE FIX: We remove the 'strategy' parameter to use the default keyword search (BM25)
    es_store = ElasticsearchStore(
        es_url=es_url,
        index_name="mars_facts_hybrid",
        embedding=embeddings,
    )
    es_retriever = es_store.as_retriever(search_kwargs={"k": 5})
    print("Elasticsearch retriever configured for default keyword search (BM25).")

    # --- 2. Set up Qdrant Retriever (Vector Search) ---
    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_client_instance = qdrant_client.QdrantClient(url=qdrant_url, prefer_grpc=False)
    qdrant_store = Qdrant(client=qdrant_client_instance, collection_name="mars_facts_hybrid", embeddings=embeddings)
    qdrant_retriever = qdrant_store.as_retriever(search_kwargs={"k": 5})
    print("Qdrant retriever configured for vector search.")

    # --- 3. Initialize the Ensemble Retriever ---
    ensemble_retriever = EnsembleRetriever(
        retrievers=[es_retriever, qdrant_retriever],
        weights=[0.5, 0.5]
    )
    print("Ensemble retriever configured with RRF.")
    
    return ensemble_retriever

def setup_rag_chain(retriever):
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    
    template = """
    Use the following pieces of retrieved context to answer the question. 
    Context: {context} 
    Question: {question} 
    Helpful Answer:
    """
    prompt = ChatPromptTemplate.from_template(template)
    
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    chain = (
        {"context": itemgetter("question") | retriever | format_docs, "question": itemgetter("question")}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain

if __name__ == "__main__":
    question = "What is the name of the largest volcano in the Solar System?"
    
    print(f"\nQuerying with: '{question}'")
    hybrid_retriever = get_hybrid_retriever()
    rag_chain = setup_rag_chain(hybrid_retriever)
    answer = rag_chain.invoke({"question": question})

    print("\n--- Answer ---")
    print(answer)
    print("--------------")