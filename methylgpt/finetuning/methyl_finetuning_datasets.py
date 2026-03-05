"""Finetuning dataset classes for MethylGPT downstream tasks.

Provides vocabulary wrappers and dataset classes for finetuning MethylGPT
on tasks like age prediction, disease classification, and survival analysis.
"""

import numpy as np
import pandas as pd
import torch
from scgpt.tokenizer import random_mask_value, tokenize_and_pad_batch
from torchtext._torchtext import Vocab as VocabPybind
from torchtext.vocab import Vocab


class CollatableVocab:
    """Vocabulary wrapper for finetuning that handles tokenization config.

    Loads CpG probe IDs from a CSV file and builds a torchtext vocabulary
    with special tokens. Also stores tokenization parameters (masking, padding).

    Args:
        model_args: Dictionary with:
            - ``n_hvg`` (int): Number of highly variable genes/CpGs.
            - ``mask_ratio`` (float): Fraction of values to mask.
            - ``mask_seed`` (int): Random seed for masking.
            - ``probe_id_dir`` (str): Path to CSV with ``illumina_probe_id`` column.

    Attributes:
        vocab: The torchtext ``Vocab`` object.
        CpG_ids: Numpy array of CpG token indices.
        max_seq_len: Maximum sequence length (n_hvg + 1 for CLS token).
    """

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
        """Load CpG probe IDs and build vocabulary.

        Returns:
            Tuple of (Vocab, numpy array of CpG indices).
        """
        CpG_list = pd.read_csv(self.model_args["probe_id_dir"])["illumina_probe_id"].values.tolist()
        CpG_ids = len(self.special_tokens) + np.arange(len(CpG_list))
        vocab = Vocab(VocabPybind(self.special_tokens + CpG_list, None))
        vocab.set_default_index(vocab["<pad>"])
        return vocab, CpG_ids


class TokenizedDataset(torch.utils.data.Dataset):
    """Generic tokenized dataset for finetuning with masked value prediction.

    Wraps a DataFrame containing methylation data and labels. Handles
    tokenization and masking via the ``collater`` method.

    Args:
        vocab: A ``CollatableVocab`` instance.
        df: DataFrame with a ``data`` column (list/array of methylation values)
            and a ``label`` column (target values).
        data_col: Column name for methylation data. Default: ``"data"``.
        label_col: Column name for labels. Default: ``"age"``.
    """

    def __init__(self, vocab, df, data_col="data", label_col="age"):
        self.vocab = vocab
        self.gene_datas = np.stack(df[data_col].values)
        self.labels = df[label_col].to_numpy()

    def __getitem__(self, index):
        gen_data = self.gene_datas[index]
        label = torch.tensor(self.labels[index]).float()
        return gen_data, label

    def __len__(self):
        return len(self.labels)

    def collater(self, batch):
        """Collate a batch with tokenization and masking.

        Args:
            batch: List of (data_array, label) tuples from ``__getitem__``.

        Returns:
            Tuple of (gene_ids, masked_values, labels) tensors.
        """
        gen_datas, labels = tuple(zip(*batch))
        gene_ids, values = self.tokenize(torch.tensor(np.array(gen_datas)))
        labels = torch.stack(labels)
        return gene_ids, values, labels

    def tokenize(self, data):
        """Tokenize and mask methylation data.

        Args:
            data: Tensor of shape ``(batch_size, n_cpgs)`` with raw values.

        Returns:
            Tuple of (gene_ids, masked_values) tensors.
        """
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


class RawDataset(torch.utils.data.Dataset):
    """Simple dataset for finetuning without tokenization in collater.

    Suitable for models that handle tokenization internally or for
    embedding-based finetuning.

    Args:
        vocab: A ``CollatableVocab`` instance.
        df: DataFrame with methylation data and labels.
        data_col: Column name for methylation data. Default: ``"data"``.
        label_col: Column name for labels. Default: ``"age"``.
    """

    def __init__(self, vocab, df, data_col="data", label_col="age"):
        self.vocab = vocab
        self.gene_datas = np.stack(df[data_col].values)
        self.labels = df[label_col].to_numpy()

    def __getitem__(self, index):
        gen_data = self.gene_datas[index]
        label = torch.tensor(self.labels[index]).float()
        return gen_data, label

    def __len__(self):
        return len(self.labels)
