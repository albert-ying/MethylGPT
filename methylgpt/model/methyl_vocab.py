import os
import json
import pandas as pd
import numpy as np
from torchtext.vocab import Vocab
from torchtext._torchtext import Vocab as VocabPybind

class MethylVocab(Vocab):
    def __init__(self, probe_id_dir, pad_token, special_tokens, save_dir):
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
