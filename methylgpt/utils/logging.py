
import logging
import sys
from pathlib import Path

def setup_logger(name, log_file, level=logging.INFO):
    """Setup and return a logger with the specified name and log file."""
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s: %(message)s")
    handler = logging.FileHandler(log_file)
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    logger.propagate = False

    return logger

def add_console_handler(logger):
    """Add a console handler to the given logger."""
    console_handler = logging.StreamHandler(stream=sys.stdout)
    logger.addHandler(console_handler)

# Save this as "logging_utilities.py" and import in other files as needed.
