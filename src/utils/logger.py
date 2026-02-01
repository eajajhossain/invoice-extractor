import logging
import os
from typing import Optional


_LOGGERS = {}


def get_logger(name: str, level: Optional[str] = None) -> logging.Logger:
    """
    Centralized logger factory.

    - Prevents duplicate handlers
    - Supports env-based log level
    - Consistent formatting across the app
    """

    if name in _LOGGERS:
        return _LOGGERS[name]

    logger = logging.getLogger(name)

    # Resolve log level
    log_level = level or os.getenv("LOG_LEVEL", "INFO")
    logger.setLevel(log_level)

    # Avoid duplicate handlers in reload / uvicorn
    if not logger.handlers:
        handler = logging.StreamHandler()

        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        handler.setFormatter(formatter)
        logger.addHandler(handler)

    logger.propagate = False
    _LOGGERS[name] = logger
    return logger
