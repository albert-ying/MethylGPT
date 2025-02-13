import sys
from pathlib import Path
sys.version
sys.path.append('/mnt/environments/scgpt_env/lib/python3.10/site-packages')
sys.path.append("/home/A.Y/project/methylGPT/modules/scGPT")
current_directory = Path(__file__).parent.absolute()
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy.stats import spearmanr, pearsonr
from scgpt.model.model import AdversarialDiscriminator, TransformerModel
from scgpt.tokenizer import tokenize_and_pad_batch, random_mask_value
import torch.nn.functional as F
from torch import nn, optim
from sklearn import preprocessing
import polars as pls
import os
import pandas as pd
from tqdm import tqdm
import json
import numpy as np
import torch
from torchtext.vocab import Vocab
from torchtext._torchtext import Vocab as VocabPybind
import lightning as pl
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch import seed_everything
import numpy as np


class CollatableVocab(object):
    def __init__(self, model_args):
        self.model_args = model_args
        self.max_seq_len = model_args["n_hvg"] + 1
        self.pad_token = "<pad>"
        self.special_tokens = [self.pad_token, "<cls>", "<eoc>"]
        self.mask_value = -1
        self.pad_value = -2
        self.mask_ratio = model_args["mask_ratio"]
        self.mask_seed = model_args["mask_seed"]
        self.vocab, self.CpG_ids = self.set_vocab()
    
    def set_vocab(self):
        CpG_list = pd.read_csv("methylGPT/"+ self.model_args["probe_id_dir"])["illumina_probe_id"].values.tolist()
        CpG_ids = len(self.special_tokens) + np.arange(len(CpG_list))
        vocab = Vocab(VocabPybind(self.special_tokens + CpG_list, None))
        vocab.set_default_index(vocab["<pad>"])
        return vocab, CpG_ids
    
    

class Age_Dataset(torch.utils.data.Dataset):
    def __init__(self, vocab: CollatableVocab, df, scaler):
        self.vocab = vocab
        self.scaler = scaler
        self.gene_datas = df["data"].to_list()
        self.ages_label_norm = self.label_norm(df["age"].to_numpy())
        self.ages_label = df["age"].to_numpy()
    
    def __getitem__(self, index: int):
        gen_data = self.gene_datas[index]
        ages_label = torch.tensor(self.ages_label[index]).float()
        ages_label_norm = self.ages_label_norm[index]
        return gen_data, ages_label, ages_label_norm
    
    def collater(self, batch):
        gen_datas, ages_labels, ages_label_norms = tuple(zip(*batch))
        gene_ids, masked_values, target_values = self.tokenize(torch.tensor(gen_datas))
        ages_labels = torch.stack(ages_labels)
        ages_label_norms = torch.stack(ages_label_norms)
        return gene_ids, masked_values, target_values, ages_labels, ages_label_norms
        
    def __len__(self):
        return len(self.ages_label)
    
    def tokenize(self, data):
        methyl_data = torch.nan_to_num(data, nan=self.vocab.pad_value)

        if isinstance(methyl_data, torch.Tensor):
            methyl_data = methyl_data.numpy()
            
        tokenized_data = tokenize_and_pad_batch(
            methyl_data,
            self.vocab.CpG_ids,
            max_len=self.vocab.max_seq_len,
            vocab=self.vocab.vocab,
            pad_token=self.vocab.pad_token,
            pad_value=self.vocab.pad_value,
            append_cls=True,
            include_zero_gene=True,
        )

        masked_values = random_mask_value(
            tokenized_data["values"],
            mask_ratio=self.vocab.mask_ratio,
            mask_value=self.vocab.mask_value,
            pad_value=self.vocab.pad_value,
            mask_seed=self.vocab.mask_seed
        )
        
        return tokenized_data["genes"], masked_values, tokenized_data["values"]
    
    def label_norm(self, data):
        return torch.tensor(self.scaler.transform(data.reshape(-1, 1)), dtype=torch.float)