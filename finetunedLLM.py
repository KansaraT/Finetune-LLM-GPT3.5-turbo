#integrate a Neo4j graph database with an OpenAI language model to query movie data using natural language. 
#Framework used : Langchain
#Dataset imported from Hugging Face.

# Import necessary libraries and modules
from langchain_community.graphs import Neo4jGraph
from datasets import load_dataset
from neo4j import GraphDatabase
from langchain.chains import GraphCypherQAChain
from langchain_openai import ChatOpenAI
import os

# Load OpenAI API key from a text file and set it as an environment variable
with open("openaiAPI.txt", 'r') as file:
    for line in file:
        key, value = line.strip().split('=')
        os.environ["OPENAI_API_KEY"] = value

# Load Neo4j credentials from a text file
config = {}
with open("Neo4jCredentials.txt", 'r') as file:
    for _ in range(3):  # Read only the first three lines
        line = file.readline().strip()
        key, value = line.split('=')
        config[key] = value

# Extract Neo4j connection details from the config dictionary
uri = config["NEO4J_URI"]
username = config["NEO4J_USERNAME"]
password = config["NEO4J_PASSWORD"]

# Initialize a Neo4jGraph object with the connection details
graph = Neo4jGraph(uri, username, password)

'''
# Load the movie dataset from Hugging Face
dataset = load_dataset("SandipPalit/Movie_Dataset")

# Create a driver instance for Neo4j database
driver = GraphDatabase.driver(uri, auth=(username, password))

# Insert the dataset into the Neo4j database
with driver.session() as session:
    for idx, record in enumerate(dataset['train']):
        session.run(
            """
            CREATE (m:Movie {
                release_date: $release_date,
                title: $title,
                overview: $overview,
                genre: $genre,
            })
            """,
            release_date=record['Release Date'],
            title=record['Title'],
            overview=record['Overview'],
            genre=record['Genre'],
        )
        if idx % 1000 == 0:
            print(f"{idx} records inserted.")

            
# Close the Neo4j driver instance
driver.close()
'''

# Initialize a GraphCypherQAChain with OpenAI's Chat model and the Neo4j graph
chain = GraphCypherQAChain.from_llm(
    llm=ChatOpenAI(temperature=0, model="gpt-3.5-turbo"),
    graph=graph,
    verbose=True,
    validate_cypher=True,
)

# Execute a query and print the result
print(chain.invoke({"query": "What is the overview of movie title Dementamania?"}))