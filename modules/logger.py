"""
Structured logging helper for CRAG modules.

Usage:
    from modules.logger import get_logger
    log = get_logger(__name__)
    log.debug("message")        # developer details
    log.info("message")         # pipeline events
    log.warning("message")      # recoverable issues
    log.error("message")        # failures
"""

import logging
import json


def get_logger(name: str) -> logging.Logger:
    """
    Returns a named logger with a consistent formatter.
    Calling this multiple times with the same name is safe — handlers are only
    added once, so there's no risk of duplicate log lines.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(
            "%(asctime)s [%(name)s] %(levelname)s %(message)s",
            datefmt="%H:%M:%S",
        ))
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)
    return logger


def json_log(logger: logging.Logger, level: str, **kwargs):
    """
    Emit a structured JSON log line.

    Example:
        json_log(log, "debug", query="what is BERT?", score=0.83)
    """
    log_fn = getattr(logger, level, logger.debug)
    log_fn(json.dumps(kwargs, default=str))
