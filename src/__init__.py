import os
from dotenv import load_dotenv
import bleach

import logging
import sys
import structlog

load_dotenv()

os.system('python src/ghm.py')

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
    if not logging.getLogger().handlers:
        logging.basicConfig(stream=sys.stdout, level=logging.INFO)

def get_env(name):
    val = os.getenv(name)
    if not val:
        raise RuntimeError(f"Missing required secret: {name}")
    return val

def sanitize_html(raw):
    # allow only text and basic tags
    return bleach.clean(raw, tags=[], strip=True)

configure_logging()
logger = structlog.get_logger()
