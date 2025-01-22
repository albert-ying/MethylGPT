import os
import json
import numpy as np
import pandas as pd
from datasets import Dataset
from torch.utils.data import IterableDataset, DataLoader
from tqdm import tqdm


class CustomDataset(IterableDataset):
    def __init__(self, parquet_files, input_normalization=False):
        self.datasets = [
            Dataset.from_parquet(file).to_iterable_dataset()
            for file in parquet_files
        ]
        self.input_normalization = input_normalization

        if self.input_normalization:
            raise NotImplementedError("Input normalization has not been implemented")

    def __iter__(self):
        for dataset in self.datasets:
            for sample in dataset:
                if self.input_normalization:
                    raise NotImplementedError("Input normalization has not been implemented")
                else:
                    yield {"id": sample["id"], "data": np.array(sample["data"])}

def split_files(files, valid_ratio):
    num_files = len(files)
    split_index = int(np.floor(num_files * (1 - valid_ratio)))
    return files[:split_index], files[split_index:]

# Create datasets
def create_dataloader(parquet_files, batch_size):
    dataset = CustomDataset(parquet_files)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=len(os.sched_getaffinity(0)),
        pin_memory=True,
        prefetch_factor=4
    )