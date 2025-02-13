import sys
from pathlib import Path
sys.version

sys.path.append(str(Path(__file__).resolve().parent.parent / "modules" / "scGPT"))
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
        CpG_list = pd.read_csv("/home/A.Y/project/methylGPT/"+ self.model_args["probe_id_dir"])["illumina_probe_id"].values.tolist()
        CpG_ids = len(self.special_tokens) + np.arange(len(CpG_list))
        vocab = Vocab(VocabPybind(self.special_tokens + CpG_list, None))
        vocab.set_default_index(vocab["<pad>"])
        return vocab, CpG_ids
    
class AltumAgeRawDataset(torch.utils.data.Dataset):
    def __init__(self, vocab: CollatableVocab, df):
        self.vocab = vocab
        self.gene_datas = np.stack(df["scaled_gene"].values)
        self.ages_label = df["age"].to_numpy()

    def __getitem__(self, index: int):
        gen_data = self.gene_datas[index]
        ages_label = torch.tensor(self.ages_label[index]).float()

        return gen_data, ages_label
        
    def __len__(self):
        return len(self.ages_label)
    
class DiseaseRawDataset(torch.utils.data.Dataset):
    def __init__(self, vocab: CollatableVocab, df):
        self.vocab = vocab
        self.gene_datas = np.stack(df["data"].values)
        self.ages_label = df["bi"].to_numpy()

    def __getitem__(self, index: int):
        gen_data = self.gene_datas[index]
        ages_label = torch.tensor(self.ages_label[index]).float()

        return gen_data, ages_label
        
    def __len__(self):
        return len(self.ages_label)
    
class DeadDayRawDataset(torch.utils.data.Dataset):
    def __init__(self, vocab: CollatableVocab, df):
        self.vocab = vocab
        self.gene_datas = np.stack(df["scaled_gene"].values)
        self.ages_label = df["diff"].to_numpy()

    def __getitem__(self, index: int):
        gen_data = self.gene_datas[index]
        ages_label = torch.tensor(self.ages_label[index]).float()

        return gen_data, ages_label
        
    def __len__(self):
        return len(self.ages_label)
    
    
class TokenizedDataset(torch.utils.data.Dataset):
    def __init__(self, vocab: CollatableVocab, df):
        self.vocab = vocab
        self.gene_datas = np.stack(df["data"].values)
        self.ages_label = df["age"].to_numpy()
        
    def __getitem__(self, index: int):
        gen_data = self.gene_datas[index]
        ages_label = torch.tensor(self.ages_label[index]).float()
        
        return gen_data, ages_label

    def collater(self, batch):
        gen_datas, ages_labels = tuple(zip(*batch))
        gene_ids, values = self.tokenize(torch.tensor(np.array(gen_datas)))
        ages_labels = torch.stack(ages_labels)
        
        return gene_ids, values, ages_labels
        
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
        )
        
        return tokenized_data["genes"], masked_values
    
    
    
class AltumAgeDataset(torch.utils.data.Dataset):
    def __init__(self, vocab: CollatableVocab, df):
        self.vocab = vocab
        self.gene_datas = np.stack(df["data"].values)
        self.ages_label = df["age"].to_numpy()
        self.ages_label_norm = df["age_scaled"].to_numpy()
        
    def __getitem__(self, index: int):
        gen_data = self.gene_datas[index]
        ages_label = torch.tensor(self.ages_label[index]).float()
        ages_label_norm = torch.tensor(self.ages_label_norm[index])
        
        return gen_data, ages_label, ages_label_norm

    def collater(self, batch):
        gen_datas, ages_labels, ages_labels_norm = tuple(zip(*batch))
        gene_ids, values = self.tokenize(torch.tensor(np.array(gen_datas)))
        ages_labels = torch.stack(ages_labels)
        ages_labels_norm = torch.stack(ages_labels_norm)
        
        return gene_ids, values, ages_labels, ages_labels_norm
        
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
        )
        
        return tokenized_data["genes"], masked_values
    

class AltumAge_MlM_Dataset(torch.utils.data.Dataset):
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
    
class Blood_DeadDay_MlM_Dataset(torch.utils.data.Dataset):
    def __init__(self, vocab: CollatableVocab, df, scaler):
        self.vocab = vocab
        self.scaler = scaler
        self.gene_datas = df["data"].to_list()
        self.ages_label_norm = self.label_norm(df["diff"].to_numpy())
        self.ages_label = df["diff"].to_numpy()
    
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
    
    def m_to_beta(self, m_values):
        m_values = np.array(m_values, dtype=float)
        return 2 ** m_values / (2 ** m_values + 1)
    
    def tokenize(self, data):
        methyl_data = torch.nan_to_num(data, nan=self.vocab.pad_value)

        if isinstance(methyl_data, torch.Tensor):
            methyl_data = methyl_data.numpy()
            
        methyl_data = self.m_to_beta(methyl_data)
            
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
    

class AgeNormDataset(torch.utils.data.Dataset):
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
        gene_ids, values = self.tokenize(torch.tensor(gen_datas))
        ages_labels = torch.stack(ages_labels)
        ages_label_norms = torch.stack(ages_label_norms)
        return gene_ids, values, ages_labels, ages_label_norms
        
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
        )
        
        return tokenized_data["genes"], masked_values
    
    def label_norm(self, data):
        return torch.tensor(self.scaler.transform(data.reshape(-1, 1)), dtype=torch.float)
        

class Age_MlM_Dataset(torch.utils.data.Dataset):
    def __init__(self, vocab: CollatableVocab, df, scaler):
        self.vocab = vocab
        self.scaler = scaler
        self.gene_datas = df["data"].to_list()
        self.ages_label_norm = self.label_norm(df["Age"].to_numpy())
        self.ages_label = df["Age"].to_numpy()
    
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
        )
        
        return tokenized_data["genes"], masked_values, tokenized_data["values"]
    
    def label_norm(self, data):
        return torch.tensor(self.scaler.transform(data.reshape(-1, 1)), dtype=torch.float)


class AgeDataset(torch.utils.data.Dataset):
    def __init__(self, vocab: CollatableVocab, df, scaler):
        self.vocab = vocab
        self.scaler = scaler
        self.gene_datas = df["data"].to_list()
        self.sexs = self.get_onehot(df["Sex"].to_numpy())
        self.ages_label_norm = self.label_norm(df["Age"].to_numpy())
        self.ages_label = df["Age"].to_numpy()
    
    def __getitem__(self, index: int):
        gen_data = self.gene_datas[index]
        sex = self.sexs[index]
        ages_label = torch.tensor(self.ages_label[index]).float()
        ages_label_norm = self.ages_label_norm[index]
        return gen_data, sex, ages_label, ages_label_norm
    
    def collater(self, batch):
        gen_datas, sexs, ages_labels, ages_label_norms = tuple(zip(*batch))
        gene_ids, values = self.tokenize(torch.tensor(gen_datas))
        sexs = torch.stack(sexs)
        ages_labels = torch.stack(ages_labels)
        ages_label_norms = torch.stack(ages_label_norms)
        return gene_ids, values, sexs, ages_labels, ages_label_norms
        
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
        )
        
        return tokenized_data["genes"], masked_values
    
    def get_onehot(self, data):
        data_to_idx = {label: idx for idx, label in enumerate(sorted(set(data)))}
        indices = [data_to_idx[label] for label in data]
        return F.one_hot(torch.tensor(indices), num_classes=len(data_to_idx))
    
    def label_norm(self, data):
        return torch.tensor(self.scaler.transform(data.reshape(-1, 1)), dtype=torch.float)
    
class AltumAgeElasticNetDataset (object):
    def __init__(self, df, scaler):
        self.scaler = scaler
        self.gene_datas = np.stack(df["scaled_gene"].values)
        self.ages_label = df["age"].to_numpy()
        self.ages_label_norm = self.label_norm(df["age"].to_numpy())
        
    def label_norm(self, data):
        return self.scaler.transform(data.reshape(-1, 1))
    
class DeadDayElasticNetDataset (object):
    def __init__(self, df, scaler):
        self.scaler = scaler
        self.gene_datas = np.stack(df["data"].values)
        self.gene_datas = np.nan_to_num(self.gene_datas, nan=0)
        self.ages_label = df["diff"].to_numpy()
        self.ages_label_norm = self.label_norm(df["diff"].to_numpy())
        
    def label_norm(self, data):
        return self.scaler.transform(data.reshape(-1, 1))

    

class Disease_MlM_Dataset(torch.utils.data.Dataset):
    def __init__(self, vocab: CollatableVocab, df, scaler):
        self.vocab = vocab
        self.scaler = scaler
        self.gene_datas = df["data"].to_list()
        self.ages_label_norm = self.label_norm(df["age"].to_numpy())
        self.ages_label = df["age"].to_numpy()
        self.disease_label = df["onehot"].to_numpy()
        self.sample_ids = df["id"].to_list()
    
    def __getitem__(self, index: int):
        sample_id = self.sample_ids[index]
        gen_data = self.gene_datas[index]
        ages_label = torch.tensor(self.ages_label[index]).float()
        ages_label_norm = self.ages_label_norm[index]
        disease_label = torch.tensor(self.disease_label[index]).float()
        return sample_id, gen_data, ages_label, ages_label_norm, disease_label
    
    def collater(self, batch):
        sample_ids, gen_datas, ages_labels, ages_label_norms, disease_labels = tuple(zip(*batch))
        gene_ids, masked_values, target_values = self.tokenize(torch.tensor(gen_datas))
        ages_labels = torch.stack(ages_labels)
        ages_label_norms = torch.stack(ages_label_norms)
        disease_labels = torch.stack(disease_labels)
        return sample_ids, gene_ids, masked_values, target_values, ages_labels, ages_label_norms, disease_labels
        
    def __len__(self):
        return len(self.ages_label)

    def m_to_beta(self, m_values):
        m_values = np.array(m_values, dtype=float)
        return 2 ** m_values / (2 ** m_values + 1)
    
    def tokenize(self, data):
        methyl_data = torch.nan_to_num(data, nan=self.vocab.pad_value)

        if isinstance(methyl_data, torch.Tensor):
            methyl_data = methyl_data.numpy()
            
        methyl_data = self.m_to_beta(methyl_data)
            
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
        )
        
        return tokenized_data["genes"], masked_values, tokenized_data["values"]
    
    def label_norm(self, data):
        return torch.tensor(self.scaler.transform(data.reshape(-1, 1)), dtype=torch.float)

class Disease_Category_RawDataset(torch.utils.data.Dataset):
    def __init__(self, vocab: CollatableVocab, df):
        self.vocab = vocab
        self.gene_datas = df["scaled_gene"].to_list()
        self.ages_label = df["age"].to_numpy()
        self.disease_label = df["onehot"].to_numpy()
        self.sample_ids = df["dnam_id"].to_list()
    
    def __getitem__(self, index: int):
        sample_id = self.sample_ids[index]
        gen_data = self.gene_datas[index]
        ages_label = torch.tensor(self.ages_label[index]).float()
        disease_label = torch.tensor(self.disease_label[index]).float()
        return sample_id, gen_data, ages_label, disease_label
        
    def __len__(self):
        return len(self.ages_label)  
    
class Disease_Category_Dataset(torch.utils.data.Dataset):
    def __init__(self, vocab: CollatableVocab, df, scaler):
        self.vocab = vocab
        self.scaler = scaler
        self.gene_datas = df["data"].to_list()
        self.ages_label_norm = self.label_norm(df["age"].to_numpy())
        self.ages_label = df["age"].to_numpy()
        self.disease_label = df["onehot"].to_numpy()
        self.sample_ids = df["dnam_id"].to_list()
    
    def __getitem__(self, index: int):
        sample_id = self.sample_ids[index]
        gen_data = self.gene_datas[index]
        ages_label = torch.tensor(self.ages_label[index]).float()
        ages_label_norm = self.ages_label_norm[index]
        disease_label = torch.tensor(self.disease_label[index]).float()
        return sample_id, gen_data, ages_label, ages_label_norm, disease_label
    
    def collater(self, batch):
        sample_ids, gen_datas, ages_labels, ages_label_norms, disease_labels = tuple(zip(*batch))
        gene_ids, masked_values, target_values = self.tokenize(torch.tensor(gen_datas))
        ages_labels = torch.stack(ages_labels)
        ages_label_norms = torch.stack(ages_label_norms)
        disease_labels = torch.stack(disease_labels)
        return sample_ids, gene_ids, masked_values, target_values, ages_labels, ages_label_norms, disease_labels
        
    def __len__(self):
        return len(self.ages_label)

    def m_to_beta(self, m_values):
        m_values = np.array(m_values, dtype=float)
        return 2 ** m_values / (2 ** m_values + 1)
    
    def tokenize(self, data):
        methyl_data = torch.nan_to_num(data, nan=self.vocab.pad_value)

        if isinstance(methyl_data, torch.Tensor):
            methyl_data = methyl_data.numpy()
            
        methyl_data = self.m_to_beta(methyl_data)
            
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
        )
        
        return tokenized_data["genes"], masked_values, tokenized_data["values"]
    
    def label_norm(self, data):
        return torch.tensor(self.scaler.transform(data.reshape(-1, 1)), dtype=torch.float)
    
    
class Disease_BI_MlM_Dataset(torch.utils.data.Dataset):
    def __init__(self, vocab: CollatableVocab, df, scaler):
        self.vocab = vocab
        self.scaler = scaler
        self.gene_datas = df["data"].to_list()
        self.ages_label_norm = self.label_norm(df["age"].to_numpy())
        self.ages_label = df["age"].to_numpy()
        self.disease_label = df["bi"].to_numpy()
    
    def __getitem__(self, index: int):
        gen_data = self.gene_datas[index]
        ages_label = torch.tensor(self.ages_label[index]).float()
        ages_label_norm = self.ages_label_norm[index]
        disease_label = torch.tensor(self.disease_label[index]).float()
        return gen_data, ages_label, ages_label_norm, disease_label
    
    def collater(self, batch):
        gen_datas, ages_labels, ages_label_norms, disease_labels = tuple(zip(*batch))
        gene_ids, masked_values, target_values = self.tokenize(torch.tensor(gen_datas))
        ages_labels = torch.stack(ages_labels)
        ages_label_norms = torch.stack(ages_label_norms)
        disease_labels = torch.stack(disease_labels)
        return gene_ids, masked_values, target_values, ages_labels, ages_label_norms, disease_labels
        
    def __len__(self):
        return len(self.ages_label)
    
    def m_to_beta(self, m_values):
        m_values = np.array(m_values, dtype=float)
        return 2 ** m_values / (2 ** m_values + 1)

    
    def tokenize(self, data):
        methyl_data = torch.nan_to_num(data, nan=self.vocab.pad_value)

        if isinstance(methyl_data, torch.Tensor):
            methyl_data = methyl_data.numpy()
        
        methyl_data = self.m_to_beta(methyl_data)
            
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
        )
        
        return tokenized_data["genes"], masked_values, tokenized_data["values"]
    
    def label_norm(self, data):
        return torch.tensor(self.scaler.transform(data.reshape(-1, 1)), dtype=torch.float)
    
    
class Intervention_Dataset(torch.utils.data.Dataset):
    def __init__(self, vocab: CollatableVocab, df, scaler, data_name):
        self.vocab = vocab
        self.scaler = scaler
        self.gene_datas = df["data"].to_list()
        self.sample_ids = df["ID_REF"].to_list()
        self.data_name = data_name
    
    def __getitem__(self, index: int):
        data_name = self.data_name
        gen_data = self.gene_datas[index]
        sample_id = self.sample_ids[index]
        return gen_data, sample_id, data_name
    
    def collater(self, batch):
        gen_datas, sample_ids, data_name = tuple(zip(*batch))
        gene_ids, masked_values, target_values = self.tokenize(torch.tensor(gen_datas))

        return data_name, sample_ids, gene_ids, masked_values, target_values
        
    def __len__(self):
        return len(self.sample_ids)

    
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
        )
        
        return tokenized_data["genes"], masked_values, tokenized_data["values"]
    
