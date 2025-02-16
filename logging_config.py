#logging_config.py
import logging
from config import LOGGING_LEVEL, LOGGING_FORMAT, LOGGING_HANDLERS

def configure_logging():
    logging.basicConfig(
        level=LOGGING_LEVEL,
        format=LOGGING_FORMAT,
        handlers=LOGGING_HANDLERS
    )

    logger = logging.getLogger('veristream')
    return logger