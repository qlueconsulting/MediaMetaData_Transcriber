"""Application logging configuration."""

import sys
from loguru import logger
from app.config import settings


def setup_logger():
    """Configure loguru logging with structured formatting."""
    logger.remove()

    log_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
        "<level>{message}</level>"
    )

    logger.add(
        sys.stderr,
        format=log_format,
        level=settings.LOG_LEVEL.upper(),
        colorize=True,
    )

    return logger


# Global logger instance
log = setup_logger()
