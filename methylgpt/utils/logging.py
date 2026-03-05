"""Logging utilities for MethylGPT."""

import logging
import sys


def setup_logger(name: str, log_file: str, level: int = logging.INFO) -> logging.Logger:
    """Create and configure a logger with file output.

    Args:
        name: Logger name (typically ``__name__`` of the calling module).
        log_file: Path to the log file.
        level: Logging level. Default: ``logging.INFO``.

    Returns:
        Configured ``logging.Logger`` instance.

    Example:
        >>> logger = setup_logger("methylgpt.train", "training.log")
        >>> logger.info("Training started")
    """
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s: %(message)s")
    handler = logging.FileHandler(log_file)
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    logger.propagate = False

    return logger


def add_console_handler(logger: logging.Logger) -> None:
    """Add a console (stdout) handler to an existing logger.

    Args:
        logger: The logger to add console output to.
    """
    console_handler = logging.StreamHandler(stream=sys.stdout)
    logger.addHandler(console_handler)
