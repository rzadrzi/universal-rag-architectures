# src/graph_rag/ingest_graph.py

import os
import pandas as pd
from dotenv import load_dotenv
from neo4j import GraphDatabase

print("Loading environment variables...")
load_dotenv()

class Neo4jGraph:
    def __init__(self, uri, user, password):
        # Establish a connection to the Neo4j database
        self._driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        # Close the database connection
        self._driver.close()

    def execute_query(self, query, parameters=None):
        # Execute a Cypher query
        with self._driver.session() as session:
            result = session.run(query, parameters)
            return [record for record in result]

    def build_graph_from_csv(self, file_path):
        """
        Reads a CSV and creates Movie and Genre nodes, and IN_GENRE relationships.
        """
        print(f"Reading data from {file_path}...")
        df = pd.read_csv(file_path)

        # Ensure the database is clean before ingesting
        print("Clearing existing graph data...")
        self.execute_query("MATCH (n) DETACH DELETE n")

        print("Ingesting data and building the graph...")
        # Loop through each movie in the dataframe
        for index, row in df.iterrows():
            movie_id = row['movieId']
            title = row['title']
            release_year = row['releaseYear']
            genres = row['genres'].split('|')

            # Create a Movie node
            movie_query = """
            MERGE (m:Movie {movieId: $id})
            ON CREATE SET m.title = $title, m.releaseYear = $year
            """
            self.execute_query(movie_query, parameters={'id': movie_id, 'title': title, 'year': release_year})

            # For each genre, create a Genre node and a relationship to the movie
            for genre_name in genres:
                genre_query = """
                MERGE (g:Genre {name: $g_name})
                WITH g
                MATCH (m:Movie {movieId: $m_id})
                MERGE (m)-[:IN_GENRE]->(g)
                """
                self.execute_query(genre_query, parameters={'g_name': genre_name, 'm_id': movie_id})
        
        print("Graph ingestion complete!")


if __name__ == "__main__":
    # Get credentials from .env file
    uri = os.getenv("NEO4J_URI")
    user = os.getenv("NEO4J_USERNAME")
    password = os.getenv("NEO4J_PASSWORD")

    # Create the graph instance and build the graph
    graph = Neo4jGraph(uri, user, password)
    graph.build_graph_from_csv("data/movies.csv")
    graph.close()