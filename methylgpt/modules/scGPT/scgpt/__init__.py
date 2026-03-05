"""Vendored scGPT components used by MethylGPT.

Only the model, tokenizer, loss, and utility modules are included.
Other scGPT features (scbank, tasks, trainer, etc.) have been removed.
"""

__version__ = "0.2.1"

import logging
import sys

logger = logging.getLogger("scGPT")
if not logger.hasHandlers() or len(logger.handlers) == 0:
    logger.propagate = False
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(name)s - %(levelname)s - %(message)s", datefmt="%H:%M:%S"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

from . import model, tokenizer, utils
