"""
core/logger.py — Project-wide logger using Loguru.

Usage::

    from core.logger import logger
    logger.info("Starting up…")
"""

import sys
from pathlib import Path

from loguru import logger

from config.settings import settings

# Remove the default Loguru handler and replace with our own config.
logger.remove()

_log_dir = Path(settings.log_dir)
_log_dir.mkdir(parents=True, exist_ok=True)

# Console handler — INFO and above, with colour.
logger.add(
    sys.stderr,
    level="INFO",
    format=(
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> — "
        "<level>{message}</level>"
    ),
    colorize=True,
)

# Rotating file handler — DEBUG and above, one file per day.
logger.add(
    _log_dir / "assistant_{time:YYYY-MM-DD}.log",
    level="DEBUG",
    rotation="00:00",      # new file every midnight
    retention="14 days",
    compression="zip",
    encoding="utf-8",
)
