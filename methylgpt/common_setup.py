
import json
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.csv as csv
import pyarrow.parquet as pq
import torch
import torchtext.vocab as torch_vocab
import umap
import wandb
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torchtext._torchtext import Vocab as VocabPybind
from torchtext.vocab import Vocab


import sys
sys.path.append('modules/scGPT/')


n_hvg_predefined = {"1": 21700, "2": 38058, "3": 49156, "4": 65537, "5": 46498}


class Config:
    def __init__(self, config_dict):
        self.config_dict=config_dict
        for key, value in config_dict.items():
            if isinstance(value, dict):
                # If the value is a nested dictionary, create a Config object for it
                setattr(self, key, Config(value))
            else:
                # Set the attribute with the key and value
                setattr(self, key, value)

    def __repr__(self, level=0):
        # Create a string representation of the Config object
        config_str = ""
        for key, value in vars(self).items():
            if isinstance(value, Config):
                # Recursively call __repr__ for nested Config objects
                config_str += f"{'  ' * level}{key}:\n{value.__repr__(level + 1)}"
            else:
                config_str += f"{'  ' * level}{key}={value}\n"
        return config_str
    
    


def set_seed(seed):
    """set random seed."""
    # random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False




def split_files(parquet_dirs, valid_ratio):
    total_num_excluded_last=(len(parquet_dirs)-1)
    threshold=int(np.floor(total_num_excluded_last*(1-valid_ratio)))
    train_files= parquet_dirs[:threshold]
    test_files= parquet_dirs[threshold:]
    return train_files, test_files

import hashlib
import json
from pathlib import Path

def save_config(config, save_dir):
        save_dir.mkdir(parents=True, exist_ok=True)
        print(config)
        print(f"save to {save_dir}")
        # save the whole script to the dir
        # os.system(f"cp {__file__} {save_dir}")

        with open(save_dir / "args.json", "w") as f:
            json.dump(
                {
                    k: v if not isinstance(v, Path) else str(v)
                    for k, v in config.items()
                },
                f,
                indent=4,
            )

def make_hash(config):
    """
    Generate a unique hash based on a dictionary of hyperparameters.
    
    Args:
        hyperparameters (dict): A dictionary of hyperparameters. 
                                Paths are converted to strings for serialization.

    Returns:
        str: A SHA-256 hash of the hyperparameters.
    """
    config_str = json.dumps({
        k: v if not isinstance(v, Path) else str(v) for k, v in config.items()
    }, sort_keys=True)
    config_hash = hashlib.sha256(config_str.encode()).hexdigest()
    return config_hash
