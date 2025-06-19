import os
from dotenv import load_dotenv
import bleach

load_dotenv()

def sanitize_html(raw):
    # allow only text and basic tags
    return bleach.clean(raw, tags=[], strip=True)

"""
Central configuration for the entire Document Intelligence app.
All modules import from here rather than hard-coding values.
"""

OPENAI_EMBEDDING_MODEL = os.getenv(
        "OPENAI_EMBEDDING_MODEL", "text-embedding-ada-002"
    )
class EmbeddingConfig:
    PROVIDER = os.getenv("EMBEDDING_PROVIDER",'HF')
    TEXT_MODEL = os.getenv('TEXT_EMBED_MODEL', 'sentence-transformers/all-MiniLM-L6-v2')
    META_MODEL = os.getenv('META_EMBED_MODEL', 'sentence-transformers/all-MiniLM-L6-v2')

class RetrieverConfig:
    PROVIDER = os.getenv("EMBEDDING_PROVIDER",'HF')
    TOP_K = int(os.getenv('RETRIEVER_TOP_K', 10))
    DENSE_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'
    ANN_TOP = int(os.getenv('ANN_TOP', 50))

class RerankerConfig:
    @staticmethod
    def get_device():
        import torch
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    MODEL_NAME = os.getenv('RERANKER_MODEL', 'BAAI/bge-reranker-v2-Gemma')
    DEVICE = get_device()

class GPPConfig:
    CHUNK_TOKEN_SIZE = int(os.getenv('CHUNK_TOKEN_SIZE', 256))
    DEDUP_SIM_THRESHOLD = float(os.getenv('DEDUP_SIM_THRESHOLD', 0.9))
    EXPANSION_SIM_THRESHOLD = float(os.getenv('EXPANSION_SIM_THRESHOLD', 0.85))
    COREF_CONTEXT_SIZE = int(os.getenv('COREF_CONTEXT_SIZE', 3))

class GPPConfig:
    """
    Configuration for GPP pipeline.
    """

    CHUNK_TOKEN_SIZE = 256
    DEDUP_SIM_THRESHOLD = 0.9
    EXPANSION_SIM_THRESHOLD = 0.85
    COREF_CONTEXT_SIZE = 3
    HNSW_EF_CONSTRUCTION = int(os.getenv("HNSW_EF_CONSTRUCTION", "200"))
    HNSW_M = int(os.getenv("HNSW_M", "16"))
    HNSW_EF_SEARCH = int(os.getenv("HNSW_EF_SEARCH", "50"))