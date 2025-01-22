import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent / "modules" / "scGPT"))
current_directory = Path(__file__).parent.absolute()

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy.stats import spearmanr, pearsonr
from scgpt.model.model import TransformerModel
from scgpt.tokenizer import tokenize_and_pad_batch, random_mask_value
import torch.nn.functional as F
from sklearn import preprocessing
import pandas as pd
from tqdm import tqdm
import json
import numpy as np
import torch
import lightning as pl
from sklearn.linear_model import ElasticNet
import math
import pickle
from torch import nn, Tensor
from typing import Dict, Mapping, Optional, Tuple, Any, Union

class MethylGPTModel(TransformerModel):
    def __init__(self, config, vocab):
        super().__init__(
            len(vocab),
            config["layer_size"],
            config["nhead"],
            config["layer_size"],
            config["nlayers"],
            vocab=vocab,
            dropout=config["dropout"],
            pad_token=vocab.pad_token,
            pad_value=vocab[vocab.pad_token],
            do_mvc=True,
            do_dab=False,
            use_batch_labels=False,
            num_batch_labels=None,
            domain_spec_batchnorm=False,
            n_input_bins=None,
            ecs_threshold=None,
            explicit_zero_prob=False,
            use_fast_transformer=config["fast_transformer"],
            pre_norm=config["pre_norm"])

        self.vocab = vocab
        self.config= config
        self.validation_step_outputs = []
        
    def get_cell_embeddings(self, gene_ids, values):
        # with torch.no_grad():
        #     model.eval()
        src_key_padding_mask = gene_ids.eq(self.vocab[self.vocab.pad_token])
        output_dict = self(
            gene_ids,
            values,
            src_key_padding_mask=src_key_padding_mask,
            batch_labels=None,
        )
        cell_embeddings = output_dict["cell_emb"]
            
        return cell_embeddings
    
    
    @classmethod
    def from_pretrained(self, config, vocab):
        if config["load_model"]:
            try:
                self.load_state_dict(torch.load(config["pretrained_file"], map_location="cpu"))
                print(f'Loading all model params from {config["pretrained_file"]}')
            except:
                # only load params that are in the model and match the size
                model_dict = self.state_dict()
                pretrained_dict = torch.load(config["pretrained_file"], map_location="cpu")
                pretrained_dict = {
                    k: v
                    for k, v in pretrained_dict.items()
                    if k in model_dict and v.shape == model_dict[k].shape
                }
                for k, v in pretrained_dict.items():
                    print(f"Loading params {k} with shape {v.shape}")
                model_dict.update(pretrained_dict)
                self.load_state_dict(model_dict)
        

    def prepare_data(self, batch):
        max_seq_len=self.config['max_seq_len']
        mask_ratio=self.config['mask_ratio']
        mask_value=self.config['mask_value']
        pad_token=self.config['pad_token']
        pad_value=self.config['pad_value']

        #methyl_data = batch["data"].astype(np.float32)
        methyl_data = torch.nan_to_num(methyl_data, nan=pad_value)

        if isinstance(methyl_data, torch.Tensor):
            methyl_data = methyl_data.numpy()

        # Tokenize the data
        tokenized_data = tokenize_and_pad_batch(
            methyl_data,
            self.vocab.CpG_ids,
            max_len=max_seq_len,
            vocab=self.vocab,
            pad_token=pad_token,
            pad_value=pad_value,
            append_cls=True,
            include_zero_gene=True,
        )

        # Apply masking
        masked_values = random_mask_value(
            tokenized_data["values"],
            mask_ratio=mask_ratio,
            mask_value=mask_value,
            pad_value=pad_value,
        )

        return {
            "gene_ids": tokenized_data["genes"],
            "values": masked_values,
            "target_values": tokenized_data["values"],
        }
