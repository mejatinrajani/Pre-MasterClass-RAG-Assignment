import logging
import re
import chromadb
from typing import List, Dict, Any
from src.config import Config
from src.ingestion.document_processor import RealDatabaseProcessor

logger = logging.getLogger(__name__)

class HybridRetriever:
    def __init__(self):
        processor = RealDatabaseProcessor()
        processor.run_full_ingestion()
        
        self.chroma_client = chromadb.PersistentClient(path=Config.VECTOR_DB_PATH)
        self.vector_collection = self.chroma_client.get_collection(name="bastian_knowledge")
        
        self.neo4j_active = processor.neo4j_active
        if self.neo4j_active:
            self.neo4j_driver = processor.neo4j_driver

    def extract_entities(self, query: str) -> List[str]:
        clean_query = re.sub(r'[^\w\s]', '', query).lower()
        words = [w for w in clean_query.split() if len(w) > 2]
        return words

    def search_graph_primary(self, entities: List[str]) -> List[Dict[str, Any]]:
        graph_results = []
        if not self.neo4j_active or not entities:
            return graph_results

        query = (
            "MATCH (n:Entity)-[r:RELATION]-(m:Entity) "
            "WHERE toLower(n.name) CONTAINS toLower($entity) "
            "RETURN n.name AS source, r.type AS relation, m.name AS target LIMIT 5"
        )
        
        with self.neo4j_driver.session() as session:
            for entity in entities:
                result = session.run(query, entity=entity)
                for record in result:
                    graph_results.append({
                        "text": f"{record['source']} {record['relation']} {record['target']}",
                        "source": "Neo4j Local Database",
                        "type": "relationship"
                    })
        return graph_results

    def search_vector_fallback(self, query: str) -> List[Dict[str, Any]]:
        results = self.vector_collection.query(
            query_texts=[query],
            n_results=Config.TOP_K_RETRIEVAL
        )
        
        vector_results = []
        if results['documents'] and len(results['documents']) > 0:
            for idx, doc in enumerate(results['documents'][0]):
                vector_results.append({
                    "text": doc,
                    "source": results['metadatas'][0][idx].get('source', 'ChromaDB Local'),
                    "type": "semantic_chunk"
                })
        return vector_results

    def retrieve(self, query: str) -> Dict[str, Any]:
        logger.info(f"Incoming Query: {query}")
        entities = self.extract_entities(query)
        
        context = self.search_graph_primary(entities)
        used_engine = "Neo4j Graph RAG"
        
        if len(context) < Config.MIN_SUPPORTING_CHUNKS:
            logger.info("Graph data insufficient. Fallback to ChromaDB.")
            context = self.search_vector_fallback(query)
            used_engine = "ChromaDB Vector RAG"
            
        return {
            "engine_used": used_engine,
            "context_chunks": context,
            "entities_extracted": entities
        }