# src/graph_rag/query_graph.py

import os
from dotenv import load_dotenv
from langchain_community.graphs import Neo4jGraph
from langchain.chains import GraphCypherQAChain
from langchain_openai import ChatOpenAI

print("Loading environment variables...")
load_dotenv()

# --- 1. CONNECT TO THE GRAPH ---
# Get credentials from .env file
uri = os.getenv("NEO4J_URI")
user = os.getenv("NEO4J_USERNAME")
password = os.getenv("NEO4J_PASSWORD")

# Use the LangChain Neo4jGraph wrapper
graph = Neo4jGraph(
    url=uri, 
    username=user, 
    password=password
)

# --- 2. CREATE THE GRAPH QA CHAIN ---
# Instantiate the LLM
llm = ChatOpenAI(model="gpt-4o", temperature=0)

# Create the Question-Answering chain
chain = GraphCypherQAChain.from_llm(
    graph=graph, 
    llm=llm, 
    verbose=True, # Set to True to see the generated Cypher query
    allow_dangerous_requests=True #acknowledge the risk
)

# --- 3. EXECUTE A QUERY ---
if __name__ == "__main__":
    # A natural language question
    question = "Which movies are in the Sci-Fi genre?"
    
    print(f"Querying the graph with: '{question}'")
    
    # Invoke the chain with the question
    result = chain.invoke({"query": question})
    
    print("\n--- Answer ---")
    print(result["result"])
    print("--------------")