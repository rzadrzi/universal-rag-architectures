# src/agentic_router/main.py

import os
from dotenv import load_dotenv
from operator import itemgetter

# LangChain imports for agent and tools
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.tools import Tool
from langchain import hub

# LLM and RAG component imports
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_qdrant import Qdrant
from langchain_community.graphs import Neo4jGraph
from langchain.chains import GraphCypherQAChain
from langchain.retrievers import EnsembleRetriever
from langchain_community.vectorstores import ElasticsearchStore
import qdrant_client

print("Loading environment variables...")
load_dotenv()

# --- 1. SET UP THE LLM ---
llm = ChatOpenAI(model="gpt-4o", temperature=0)

# --- 2. DEFINE THE TOOLS ---

# Tool 1: Vector and Keyword Search (Hybrid RAG)
def create_hybrid_retriever():
    embeddings = OpenAIEmbeddings()
    es_url = os.getenv("ELASTICSEARCH_URL")
    es_store = ElasticsearchStore(
        es_url=es_url, index_name="mars_facts_hybrid", embedding=embeddings
    )
    es_retriever = es_store.as_retriever(search_kwargs={"k": 5})

    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_client_instance = qdrant_client.QdrantClient(url=qdrant_url, prefer_grpc=False)
    qdrant_store = Qdrant(client=qdrant_client_instance, collection_name="mars_facts_hybrid", embeddings=embeddings)
    qdrant_retriever = qdrant_store.as_retriever(search_kwargs={"k": 5})

    return EnsembleRetriever(
        retrievers=[es_retriever, qdrant_retriever], weights=[0.5, 0.5]
    )

# Tool 2: Graph Search (Graph RAG)
def create_graph_chain():
    uri = os.getenv("NEO4J_URI")
    user = os.getenv("NEO4J_USERNAME")
    password = os.getenv("NEO4J_PASSWORD")
    graph = Neo4jGraph(url=uri, username=user, password=password)
    return GraphCypherQAChain.from_llm(
        graph=graph, llm=ChatOpenAI(temperature=0, model="gpt-4o"), verbose=True, allow_dangerous_requests=True
    )

# Instantiate the tools
hybrid_retriever = create_hybrid_retriever()
graph_chain = create_graph_chain()

# Create LangChain Tool objects
tools = [
    Tool(
        name="Vector and Keyword Search",
        func=lambda query: hybrid_retriever.invoke(query),
        description="""Use this tool for general knowledge questions about Mars, its features, 
        or any topic covered in the text documents. It is good for semantic and keyword searches."""
    ),
    Tool(
        name="Graph Search",
        func=lambda query: graph_chain.invoke({"query": query})["result"],
        description="""Use this tool when answering questions about movies, genres, and their relationships. 
        It is good for specific queries that involve connections between different entities."""
    ),
]

# --- 3. CREATE THE AGENT ---

# Get the ReAct agent prompt
prompt = hub.pull("hwchase17/react")

# Create the agent by binding the tools to the LLM
agent = create_react_agent(llm, tools, prompt)

# Create an agent executor to run the agent
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)


# --- 4. RUN THE AGENT ---
if __name__ == "__main__":
    print("\n--- Testing with a general knowledge question ---")
    general_question = "What is the largest volcano on Mars?"
    result1 = agent_executor.invoke({"input": general_question})
    print(f"\nFinal Answer: {result1['output']}")

    print("\n\n--- Testing with a graph-based question ---")
    graph_question = "Which movies are in the Action genre?"
    result2 = agent_executor.invoke({"input": graph_question})
    print(f"\nFinal Answer: {result2['output']}")