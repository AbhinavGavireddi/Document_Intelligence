import os
from typing import List, Dict, Any

from src.config import RetrieverConfig
from src import logger  # Use logger from src/__init__.py

class Retriever:
    """
    Hybrid retriever combining BM25 sparse and dense retrieval (no Redis).
    """
    def __init__(self, chunks: List[Dict[str, Any]], config: RetrieverConfig):
        # Lazy import heavy libraries
        import numpy as np
        import hnswlib
        from sentence_transformers import SentenceTransformer
        from rank_bm25 import BM25Okapi
        self.chunks = chunks
        try:
            if not isinstance(chunks, list) or not all(isinstance(c, dict) for c in chunks):
                logger.error("Chunks must be a list of dicts.")
                raise ValueError("Chunks must be a list of dicts.")
            corpus = [c.get('narration', '').split() for c in chunks]
            self.bm25 = BM25Okapi(corpus)
            self.embedder = SentenceTransformer(config.DENSE_MODEL)
            dim = len(self.embedder.encode(["test"])[0])
            self.ann = hnswlib.Index(space='cosine', dim=dim)
            self.ann.init_index(max_elements=len(chunks))
            embeddings = self.embedder.encode([c.get('narration', '') for c in chunks])
            self.ann.add_items(embeddings, ids=list(range(len(chunks))))
            self.ann.set_ef(config.ANN_TOP)
        except Exception as e:
            logger.error(f"Retriever init failed: {e}")
            self.bm25 = None
            self.embedder = None
            self.ann = None

    def retrieve_sparse(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """
        Retrieve chunks using BM25 sparse retrieval.

        Args:
        query (str): Query string.
        top_k (int): Number of top chunks to return.

        Returns:
        List[Dict[str, Any]]: List of top chunks.
        """
        if not self.bm25:
            logger.error("BM25 not initialized.")
            return []
        tokenized = query.split()
        try:
            import numpy as np  # Ensure np is defined here
            scores = self.bm25.get_scores(tokenized)
            top_indices = np.argsort(scores)[::-1][:top_k]
            return [self.chunks[i] for i in top_indices]
        except Exception as e:
            logger.error(f"Sparse retrieval failed: {e}")
            return []

    def retrieve_dense(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """
        Retrieve chunks using dense retrieval.

        Args:
        query (str): Query string.
        top_k (int): Number of top chunks to return.

        Returns:
        List[Dict[str, Any]]: List of top chunks.
        """
        if not self.ann or not self.embedder:
            logger.error("Dense retriever not initialized.")
            return []
        try:
            q_emb = self.embedder.encode([query])[0]
            labels, distances = self.ann.knn_query(q_emb, k=top_k)
            return [self.chunks[i] for i in labels[0]]
        except Exception as e:
            logger.error(f"Dense retrieval failed: {e}")
            return []

    def retrieve(self, query: str, top_k: int = None) -> List[Dict[str, Any]]:
        """
        Retrieve chunks using hybrid retrieval.

        Args:
        query (str): Query string.
        top_k (int, optional): Number of top chunks to return. Defaults to None.

        Returns:
        List[Dict[str, Any]]: List of top chunks.
        """
        if top_k is None:
            top_k = RetrieverConfig.TOP_K
        sparse = self.retrieve_sparse(query, top_k)
        dense = self.retrieve_dense(query, top_k)
        seen = set()
        combined = []
        for c in sparse + dense:
            cid = id(c)
            if cid not in seen:
                seen.add(cid)
                combined.append(c)
        return combined