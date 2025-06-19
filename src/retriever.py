import os
from typing import List, Dict, Any
import numpy as np

from src import RetrieverConfig, logger, get_chroma_client, get_embedder

class Retriever:
    """
    Retrieves documents from a ChromaDB collection.
    """
    def __init__(self, collection_name: str, config: RetrieverConfig):
        self.collection_name = collection_name
        self.config = config
        self.client = get_chroma_client()
        self.embedder = get_embedder()
        self.collection = self.client.get_or_create_collection(name=self.collection_name)

    def retrieve(self, query: str, top_k: int = None) -> List[Dict[str, Any]]:
        """
        Embeds a query and retrieves the top_k most similar documents from ChromaDB.
        """
        if top_k is None:
            top_k = self.config.TOP_K
        
        if self.collection.count() == 0:
            logger.warning(f"Chroma collection '{self.collection_name}' is empty. Cannot retrieve.")
            return []

        try:
            # 1. Embed the query
            query_embedding = self.embedder.embed([query])[0]
            
            # 2. Query ChromaDB
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                include=["metadatas", "documents"] 
            )
            
            # 3. Format results into chunks
            # Chroma returns lists of lists, so we access the first element.
            if not results or not results.get('ids', [[]])[0]:
                return []

            ids = results['ids'][0]
            documents = results['documents'][0]
            metadatas = results['metadatas'][0]
            
            retrieved_chunks = []
            for i, doc_id in enumerate(ids):
                chunk = {
                    'id': doc_id,
                    'narration': documents[i],
                    **metadatas[i]  # Add all other metadata from Chroma
                }
                retrieved_chunks.append(chunk)

            return retrieved_chunks

        except Exception as e:
            logger.error(f"ChromaDB retrieval failed for collection '{self.collection_name}': {e}")
            return []