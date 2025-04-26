"""
Utilities module: LLM client wrapper and shared helpers.
"""
import os
import openai
import logging
import sys
import structlog


def configure_logging():
    structlog.configure(
        processors=[
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer()
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)


configure_logging()
logger = structlog.get_logger()

class LLMClient:
    """
    Simple wrapper around OpenAI (or any other) LLM API.
    Reads API key from environment and exposes `generate(prompt)`.
    """
    @staticmethod
    def generate(prompt: str, model: str = None, max_tokens: int = 512, **kwargs) -> str:
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            logger.error('OPENAI_API_KEY is not set')
            raise EnvironmentError('Missing OPENAI_API_KEY')
        openai.api_key = api_key
        model_name = model or os.getenv('OPENAI_MODEL', 'gpt-4o')
        try:
            resp = openai.ChatCompletion.create(
                model=model_name,
                messages=[{"role": "system", "content": "You are a helpful assistant."},
                          {"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=0.0,
                **kwargs
            )
            text = resp.choices[0].message.content.strip()
            return text
        except Exception as e:
            logger.exception('LLM generation failed')
            raise

