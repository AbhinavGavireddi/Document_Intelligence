import os
from dotenv import load_dotenv
import bleach
from loguru import logger
import streamlit as st
from sentence_transformers import SentenceTransformer
import torch
import chromadb
from src.utils import OpenAIEmbedder, LocalEmbedder

load_dotenv()

def sanitize_html(raw):
    # allow only text and basic tags
    return bleach.clean(raw, tags=[], strip=True)

"""
Central configuration for the entire Document Intelligence app.
All modules import from here rather than hard-coding values.
"""

# --- Embedding & ChromaDB Config ---
class EmbeddingConfig:
    PROVIDER = os.getenv("EMBEDDING_PROVIDER", 'local')
    TEXT_MODEL = os.getenv('TEXT_EMBED_MODEL', 'sentence-transformers/all-MiniLM-L6-v2')

# --- Retriever Config for Low Latency ---
class RetrieverConfig:
    # Retrieve more chunks initially, let the final prompt handle trimming.
    TOP_K = int(os.getenv('RETRIEVER_TOP_K', 5)) 

# --- GPP Config ---
class GPPConfig:
    CHUNK_TOKEN_SIZE = int(os.getenv('CHUNK_TOKEN_SIZE', 256))
    DEDUP_SIM_THRESHOLD = float(os.getenv('DEDUP_SIM_THRESHOLD', 0.9))
    EXPANSION_SIM_THRESHOLD = float(os.getenv('EXPANSION_SIM_THRESHOLD', 0.85))
    COREF_CONTEXT_SIZE = int(os.getenv('COREF_CONTEXT_SIZE', 3))

# --- Centralized, Streamlit-cached Clients & Models ---
@st.cache_resource(show_spinner="Connecting to ChromaDB...")
def get_chroma_client():
    """
    Initializes a ChromaDB client.
    Defaults to a serverless, persistent client, which is ideal for local
    development and single-container deployments.
    If CHROMA_HOST is set, it will attempt to connect to a standalone server.
    """
    chroma_host = os.getenv("CHROMA_HOST")
    
    if chroma_host:
        logger.info(f"Connecting to ChromaDB server at {chroma_host}...")
        client = chromadb.HttpClient(
            host=chroma_host, 
            port=int(os.getenv("CHROMA_PORT", "8000"))
        )
    else:
        persist_directory = os.getenv("PERSIST_DIRECTORY", "./parsed/chroma_db")
        logger.info(f"Using persistent ChromaDB at: {persist_directory}")
        client = chromadb.PersistentClient(path=persist_directory)
        
    return client

@st.cache_resource(show_spinner="Loading embedding model...")
def get_embedder():
    if EmbeddingConfig.PROVIDER == "openai":
        logger.info(f"Using OpenAI embedder with model: {EmbeddingConfig.TEXT_MODEL}")
        return OpenAIEmbedder(model_name=EmbeddingConfig.TEXT_MODEL)
    else:
        logger.info(f"Using local embedder with model: {EmbeddingConfig.TEXT_MODEL}")
        return LocalEmbedder(model_name=EmbeddingConfig.TEXT_MODEL)

    