import os
import numpy as np
import redis
import hnswlib
from typing import List, Dict, Any

from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

class RetrieverConfig:
    TOP_K = 10  # number of candidates per retrieval path
    DENSE_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'
    REDIS_HOST = os.getenv('REDIS_HOST', 'localhost')
    REDIS_PORT = int(os.getenv('REDIS_PORT', 6379))
    REDIS_DB = int(os.getenv('REDIS_DB', 0))
    REDIS_VECTOR_INDEX = 'gpp_vectors'

class Retriever:
    """
    Hybrid retriever combining BM25 sparse and Redis-based dense retrieval.
    """
    def __init__(self, chunks: List[Dict[str, Any]], config: RetrieverConfig):
        self.chunks = chunks
        # Build BM25 index over chunk narrations
        corpus = [c['narration'].split() for c in chunks]
        self.bm25 = BM25Okapi(corpus)
        # Load dense embedder
        self.embedder = SentenceTransformer(config.DENSE_MODEL)
        # Connect to Redis for vector store
        self.redis = redis.Redis(host=config.REDIS_HOST,
                                 port=config.REDIS_PORT,
                                 db=config.REDIS_DB)
        self.vector_index = config.REDIS_VECTOR_INDEX

        # Build HNSW index
        dim = len(self.embedder.encode(["test"])[0])
        self.ann = hnswlib.Index(space='cosine', dim=dim)
        self.ann.init_index(max_elements=len(chunks), ef_construction=200, M=16)
        embeddings = self.embedder.encode([c['narration'] for c in chunks])
        self.ann.add_items(embeddings, ids=list(range(len(chunks))))
        self.ann.set_ef(50)  # ef should be > top_k for accuracy

    def retrieve_sparse(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """Return top_k chunks by BM25 score."""
        tokenized = query.split()
        scores = self.bm25.get_scores(tokenized)
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [self.chunks[i] for i in top_indices]

    def retrieve_dense(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """Return top_k chunks by dense cosine similarity via Redis vectors."""
        # Embed query
        q_emb = self.embedder.encode([query])[0]
        labels, distances = self.ann.knn_query(q_emb, k=top_k)
        return [self.chunks[i] for i in labels[0]]

    def retrieve(self, query: str, top_k: int = RetrieverConfig.TOP_K) -> List[Dict[str, Any]]:
        """Combine sparse + dense results (unique) into candidate pool."""
        sparse = self.retrieve_sparse(query, top_k)
        dense = self.retrieve_dense(query, top_k)
        # Union while preserving order
        seen = set()
        combined = []
        for c in sparse + dense:
            cid = id(c)
            if cid not in seen:
                seen.add(cid)
                combined.append(c)
        return combined