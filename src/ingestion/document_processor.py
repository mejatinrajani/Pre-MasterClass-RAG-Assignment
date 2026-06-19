import json
import logging
import os
import chromadb
from neo4j import GraphDatabase
from src.config import Config

logger = logging.getLogger(__name__)

class RealDatabaseProcessor:
    def __init__(self):
        self.raw_json_path = Config.RAW_JSON_PATH
        
        # Initialize ChromaDB
        os.makedirs(Config.VECTOR_DB_PATH, exist_ok=True)
        self.chroma_client = chromadb.PersistentClient(path=Config.VECTOR_DB_PATH)
        self.vector_collection = self.chroma_client.get_or_create_collection(name="bastian_knowledge")
        
        # Initialize Neo4j (Local)
        self.neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.neo4j_user = os.getenv("NEO4J_USERNAME", "neo4j")
        self.neo4j_password = os.getenv("NEO4J_PASSWORD", "password")
        
        try:
            self.neo4j_driver = GraphDatabase.driver(self.neo4j_uri, auth=(self.neo4j_user, self.neo4j_password))
            self.neo4j_driver.verify_connectivity()
            self.neo4j_active = True
            logger.info("Connected to local Neo4j database successfully.")
        except Exception as e:
            logger.error(f"Failed to connect to local Neo4j. Is the database running in Neo4j Desktop? Error: {e}")
            self.neo4j_active = False

    def load_rulebook(self) -> dict:
        with open(self.raw_json_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def ingest_to_chroma(self, rulebook: dict):
        logger.info("Ingesting chunks into ChromaDB...")
        venue = rulebook.get("venue_highlights", {})
        notes = rulebook.get("dynamic_info_notes", {})
        
        documents = [
            f"{venue.get('name')} description: {venue.get('description')}",
            f"{venue.get('name')} Contact details. Phone: {venue.get('contact', {}).get('phone')}, WhatsApp: {venue.get('contact', {}).get('whatsapp')}",
            f"{venue.get('name')} Seating Capacity details. Total daily limit: {venue.get('seating_capacity', {}).get('total_daily')}."
        ]
        
        for key, val in notes.items():
            documents.append(f"Policy Note regarding {key.replace('_', ' ')}: {val}")

        ids = [f"chunk_{i}" for i in range(len(documents))]
        metadatas = [{"source": "venue_json"} for _ in range(len(documents))]
        
        self.vector_collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        logger.info(f"Successfully stored {len(documents)} vectors in ChromaDB.")

    def ingest_to_neo4j(self, rulebook: dict):
        if not self.neo4j_active:
            return

        logger.info("Ingesting relationships into local Neo4j...")
        venue = rulebook.get("venue_highlights", {})
        
        def create_tx(tx, source, relation, target):
            query = (
                "MERGE (a:Entity {name: $source}) "
                "MERGE (b:Entity {name: $target}) "
                "MERGE (a)-[r:RELATION {type: $relation}]->(b)"
            )
            tx.run(query, source=source, target=target, relation=relation)

        with self.neo4j_driver.session() as session:
            # Safe to use because this is an isolated local database
            session.run("MATCH (n) DETACH DELETE n")
            
            for founder in venue.get("founders", []):
                session.execute_write(create_tx, founder, "FOUNDED", venue.get("name"))
            
            if "architect" in venue:
                session.execute_write(create_tx, venue.get("architect"), "DESIGNED", venue.get("name"))
                
            session.execute_write(create_tx, venue.get("name"), "LOCATED_AT", venue.get("address"))
            
        logger.info("Successfully mapped relationships in local Neo4j.")

    def run_full_ingestion(self):
        rulebook = self.load_rulebook()
        self.ingest_to_chroma(rulebook)
        self.ingest_to_neo4j(rulebook)