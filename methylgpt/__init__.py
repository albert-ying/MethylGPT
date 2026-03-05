"""MethylGPT: a foundation model for the DNA methylome.

This package provides the core model, vocabulary, datasets, and utilities
for pretraining and finetuning MethylGPT on DNA methylation data.

Example:
    >>> from methylgpt import MethylGPTModel, MethylVocab
    >>> vocab = MethylVocab("data/probe_ids.csv", "<pad>", ["<pad>", "<cls>", "<eoc>"], None)
    >>> model = MethylGPTModel.from_pretrained(config, vocab)
"""

import os
import sys
from pathlib import Path

__version__ = "0.2.0"

# Add vendored scGPT to sys.path so that `import scgpt` resolves
_scgpt_path = str(Path(__file__).resolve().parent / "modules" / "scGPT")
if _scgpt_path not in sys.path:
    sys.path.insert(0, _scgpt_path)

# Public API imports
from methylgpt.model.methyl_datasets import (
    CustomDataset,
    StandardizedCpGDataset,
    create_dataloader,
    split_files,
)
from methylgpt.model.methyl_loss import criterion_neg_log_bernoulli, masked_mse_loss, masked_relative_error
from methylgpt.model.methyl_model import MethylGPTModel
from methylgpt.model.methyl_pretraining import random_mask_value
from methylgpt.model.methyl_vocab import MethylVocab

__all__ = [
    "__version__",
    "MethylGPTModel",
    "MethylVocab",
    "CustomDataset",
    "StandardizedCpGDataset",
    "split_files",
    "create_dataloader",
    "masked_mse_loss",
    "criterion_neg_log_bernoulli",
    "masked_relative_error",
    "random_mask_value",
]
