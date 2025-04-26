"""
Central configuration for the entire Document Intelligence app.
All modules import from here rather than hard-coding values.
"""
import os

class RedisConfig:
    HOST = os.getenv('REDIS_HOST', 'localhost')
    PORT = int(os.getenv('REDIS_PORT', 6379))
    DB = int(os.getenv('REDIS_DB', 0))
    VECTOR_INDEX = os.getenv('REDIS_VECTOR_INDEX', 'gpp_vectors')

class EmbeddingConfig:
    TEXT_MODEL = os.getenv('TEXT_EMBED_MODEL', 'sentence-transformers/all-MiniLM-L6-v2')
    META_MODEL = os.getenv('META_EMBED_MODEL', 'sentence-transformers/all-MiniLM-L6-v2')

class RetrieverConfig:
    TOP_K = int(os.getenv('RETRIEVER_TOP_K', 10))  # number of candidates per retrieval path
    DENSE_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'
    ANN_TOP = int(os.getenv('ANN_TOP', 50))

class RerankerConfig:
    MODEL_NAME = os.getenv('RERANKER_MODEL', 'BAAI/bge-reranker-v2-Gemma')
    DEVICE = os.getenv('RERANKER_DEVICE', 'cuda' if os.getenv('CUDA_VISIBLE_DEVICES') else 'cpu')

class GPPConfig:
    CHUNK_TOKEN_SIZE = int(os.getenv('CHUNK_TOKEN_SIZE', 256))
    DEDUP_SIM_THRESHOLD = float(os.getenv('DEDUP_SIM_THRESHOLD', 0.9))
    EXPANSION_SIM_THRESHOLD = float(os.getenv('EXPANSION_SIM_THRESHOLD', 0.85))
    COREF_CONTEXT_SIZE = int(os.getenv('COREF_CONTEXT_SIZE', 3))

# Add other configs (e.g. Streamlit settings, CI flags) as needed.