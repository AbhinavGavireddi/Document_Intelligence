"""
Utilities module: LLM client wrapper and shared helpers.
"""
import os
import openai
from openai import AzureOpenAI, error

try:
    from src.utils import logger
except ImportError:
    import structlog
    logger = structlog.get_logger()

class LLMClient:
    """
    Simple wrapper around OpenAI (or any other) LLM API.
    Reads API key from environment and exposes `generate(prompt)`.
    """
    @staticmethod
    def generate(prompt: str, model: str = None, max_tokens: int = 512, **kwargs) -> str:
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
            resp = client.ChatCompletion.create(
                model=openai_model_name,
                messages=[{"role": "system", "content": "You are a helpful assistant."},
                          {"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=0.0,
                **kwargs
            )
            text = resp.choices[0].message.content.strip()
            return text
        except openai.error.OpenAIError as oe:
            logger.error(f'OpenAI API error: {oe}')
            raise
        except Exception as e:
            logger.exception('LLM generation failed')
            raise
