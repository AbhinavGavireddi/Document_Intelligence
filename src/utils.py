"""
Utilities module: LLM client wrapper and shared helpers.
"""
import os
import openai
from typing import List
from openai import AzureOpenAI
from langchain_openai import AzureOpenAIEmbeddings
from sentence_transformers import SentenceTransformer
from src import logger


class LLMClient:
    """
    Simple wrapper around OpenAI (or any other) LLM API.
    Reads API key from environment and exposes `generate(prompt)`.
    """
    @staticmethod
    def generate(prompt: str, model: str = None, max_tokens: int = 512, **kwargs) -> tuple[str, str]:
        azure_api_key = os.getenv('AZURE_API_KEY')
        azure_endpoint = os.getenv('AZURE_ENDPOINT')
        azure_api_version = os.getenv('AZURE_API_VERSION')
        openai_model_name = model or os.getenv('OPENAI_MODEL', 'gpt-4o')

        if not (azure_api_key or azure_endpoint or azure_api_version or openai_model_name):
            logger.error('OPENAI_API_KEY is not set')
            raise EnvironmentError('Missing OPENAI_API_KEY')
        client = AzureOpenAI(
                api_key=azure_api_key,
                azure_endpoint=azure_endpoint,
                api_version=azure_api_version
            )
        try:
            resp = client.chat.completions.create(
                model=openai_model_name,
                messages=[{"role": "system", "content": "You are a helpful assistant."},
                          {"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=0.0,
                **kwargs
            )
            text = resp.choices[0].message.content.strip()
            finish_reason = resp.choices[0].finish_reason
            return text, finish_reason
        except Exception as e:
            logger.error(f'LLM generation failed: {e}')
            raise


class LocalEmbedder:
    """
    Wrapper for a local SentenceTransformer model.
    """
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)
        logger.info(f"Initialized local embedder with model: {model_name}")

    def embed(self, texts: List[str]) -> List[List[float]]:
        """Embeds a list of texts using the local SentenceTransformer model."""
        try:
            embeddings = self.model.encode(texts, show_progress_bar=False)
            return embeddings.tolist()
        except Exception as e:
            logger.error(f"Local embedding failed: {e}")
            raise


class OpenAIEmbedder:
    """
    Wrapper around OpenAI and Azure OpenAI Embeddings.
    Automatically uses Azure credentials if available, otherwise falls back to OpenAI.
    """
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.is_azure = os.getenv('AZURE_API_KEY') and os.getenv('AZURE_ENDPOINT')
        
        if self.is_azure:
            logger.info("Using Azure OpenAI for embeddings.")
            self.embedder = AzureOpenAIEmbeddings(
                model=self.model_name,
                azure_deployment=os.getenv("AZURE_EMBEDDING_DEPLOYMENT"), # Assumes a deployment name is set
                api_version=os.getenv("AZURE_API_VERSION")
            )
        else:
            logger.info("Using standard OpenAI for embeddings.")
            # This part would need OPENAI_API_KEY to be set
            from langchain_openai import OpenAIEmbeddings
            self.embedder = OpenAIEmbeddings(model=self.model_name)

    def embed(self, texts: List[str]) -> List[List[float]]:
        """Embeds a list of texts."""
        try:
            return self.embedder.embed_documents(texts)
        except Exception as e:
            logger.error(f"Embedding failed: {e}")
            raise