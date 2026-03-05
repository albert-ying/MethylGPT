"""Vocabulary management for MethylGPT CpG site tokens."""

import json
import os

import numpy as np
import pandas as pd
from torchtext._torchtext import Vocab as VocabPybind
from torchtext.vocab import Vocab


class MethylVocab(Vocab):
    """Vocabulary mapping CpG probe IDs to integer token indices.

    Builds a vocabulary from Illumina probe IDs (e.g., ``cg00000029``)
    loaded from a CSV file. Special tokens (``<pad>``, ``<cls>``, ``<eoc>``)
    are prepended to the vocabulary.

    Args:
        probe_id_dir: Path to a CSV file with an ``illumina_probe_id`` column.
        pad_token: The padding token string (typically ``"<pad>"``).
        special_tokens: List of special token strings.
        save_dir: Directory to save ``vocab.json``. If ``None``, vocab is not saved.

    Attributes:
        CpG_list: List of CpG probe ID strings.
        CpG_ids: Numpy array of integer indices for CpG tokens (excluding special tokens).
        pad_token: The padding token string.

    Example:
        >>> vocab = MethylVocab("data/probe_ids.csv", "<pad>", ["<pad>", "<cls>", "<eoc>"], None)
        >>> vocab["cg00000029"]  # Returns integer index
        3
    """

    def __init__(self, probe_id_dir, pad_token, special_tokens, save_dir=None):
        self.probe_id_dir = probe_id_dir
        self.special_tokens = special_tokens
        self.save_dir = save_dir
        self.pad_token = pad_token

        # Initialize vocab with special tokens and CpG list
        cpG_list = self._load_cpg_list()
        vocab_pybind = VocabPybind(self.special_tokens + cpG_list, None)
        super().__init__(vocab_pybind)

        self.set_default_index(self[pad_token])
        self.CpG_list = cpG_list
        self.CpG_ids = len(self.special_tokens) + np.arange(len(cpG_list))

        # Save the vocab to the specified directoryasdfs
        if self.save_dir is not None:
            self._save_vocab()

    def _load_cpg_list(self):
        """Load the CpG list from the given CSV file."""
        return pd.read_csv(self.probe_id_dir)["illumina_probe_id"].tolist()

    def _save_vocab(self):
        """Save the vocabulary as a JSON file in the specified directory."""
        os.makedirs(self.save_dir, exist_ok=True)
        with open(os.path.join(self.save_dir, "vocab.json"), "w") as f:
            json.dump(self.get_stoi(), f, indent=4)
