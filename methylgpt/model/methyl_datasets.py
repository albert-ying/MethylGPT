"""Dataset classes for MethylGPT pretraining and inference."""

import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, IterableDataset

logger = logging.getLogger(__name__)


class CustomDataset(IterableDataset):
    """Iterable dataset that streams methylation data from Parquet chunk files.

    Each Parquet file should contain rows with:
        - ``id``: Sample/cell identifier string.
        - ``data``: List or array of methylation beta values, ordered by
          the vocabulary's CpG probe order.

    Args:
        parquet_chunk_files: List of file paths to Parquet chunk files.

    Yields:
        Dictionary with ``"id"`` (str) and ``"data"`` (1D numpy float32 array).
    """

    def __init__(self, parquet_chunk_files):
        self.parquet_chunk_files = parquet_chunk_files

    def __iter__(self):
        for file_path_str in self.parquet_chunk_files:
            file_path = Path(file_path_str)
            try:
                df_chunk = pd.read_parquet(file_path)
            except Exception as e:
                logger.warning(f"Error reading {file_path}: {e}. Skipping.")
                continue

            if "id" not in df_chunk.columns or "data" not in df_chunk.columns:
                logger.warning(f"Missing 'id' or 'data' column in {file_path}. Skipping.")
                continue

            for _, row in df_chunk.iterrows():
                cell_id = str(row["id"])
                values = row["data"]
                if not isinstance(values, (list, np.ndarray)):
                    continue
                try:
                    data_array = np.array(values, dtype=np.float32)
                    if data_array.ndim != 1:
                        continue
                except (ValueError, TypeError):
                    continue
                yield {"id": cell_id, "data": data_array}


class StandardizedCpGDataset(IterableDataset):
    """Dataset that applies per-CpG z-score normalization while streaming.

    Each CpG site's methylation value is standardized as:
    ``(x - mean) / std`` using precomputed per-CpG statistics.
    This forces the model to learn cross-sample variation rather than
    memorizing per-CpG baseline levels.

    Args:
        parquet_chunk_files: List of Parquet file paths.
        cpg_means: 1D numpy array of per-CpG means, shape ``(n_cpgs,)``.
        cpg_stds: 1D numpy array of per-CpG standard deviations, shape ``(n_cpgs,)``.
            Values below 1e-6 are clipped to avoid division by zero.

    Yields:
        Dictionary with ``"id"`` (str) and ``"data"`` (1D numpy float32 array
        of z-scored methylation values).

    Example:
        >>> means = np.load("cpg_stats/cpg_means.npy")
        >>> stds = np.load("cpg_stats/cpg_stds.npy")
        >>> dataset = StandardizedCpGDataset(parquet_files, means, stds)
        >>> for sample in dataset:
        ...     print(sample["data"].mean())  # approximately 0
    """

    def __init__(self, parquet_chunk_files, cpg_means, cpg_stds):
        self.parquet_chunk_files = parquet_chunk_files
        self.cpg_means = cpg_means.astype(np.float32)
        self.cpg_stds = np.clip(cpg_stds.astype(np.float32), a_min=1e-6, a_max=None)

    def __iter__(self):
        for file_path_str in self.parquet_chunk_files:
            file_path = Path(file_path_str)
            try:
                df_chunk = pd.read_parquet(file_path)
            except Exception as e:
                logger.warning(f"Error reading {file_path}: {e}. Skipping.")
                continue

            if "id" not in df_chunk.columns or "data" not in df_chunk.columns:
                logger.warning(f"Missing 'id' or 'data' column in {file_path}. Skipping.")
                continue

            n_cpgs = len(self.cpg_means)
            for _, row in df_chunk.iterrows():
                cell_id = str(row["id"])
                values = row["data"]
                if not isinstance(values, (list, np.ndarray)):
                    continue
                try:
                    raw = np.array(values, dtype=np.float32)[:n_cpgs]
                    standardized = (raw - self.cpg_means[: len(raw)]) / self.cpg_stds[: len(raw)]
                except (ValueError, TypeError):
                    continue
                yield {"id": cell_id, "data": standardized}


def split_files(files, valid_ratio):
    """Split a list of files into training and validation sets.

    Args:
        files: List of file paths.
        valid_ratio: Fraction of files to use for validation (e.g., 0.1).

    Returns:
        Tuple of (train_files, valid_files).
    """
    num_files = len(files)
    split_index = int(np.floor(num_files * (1 - valid_ratio)))
    return files[:split_index], files[split_index:]


def create_dataloader(parquet_chunk_files, batch_size, num_workers=None, max_workers=8):
    """Create a DataLoader for streaming methylation data from Parquet files.

    Args:
        parquet_chunk_files: List of Parquet file paths.
        batch_size: Number of samples per batch.
        num_workers: Number of data loading workers. If ``None``, auto-detected.
        max_workers: Maximum number of workers. Default: 8.

    Returns:
        A ``torch.utils.data.DataLoader`` instance.
    """
    dataset = CustomDataset(parquet_chunk_files)

    n_files = len(parquet_chunk_files)
    if n_files == 0:
        logger.warning("No Parquet files provided. DataLoader will be empty.")
        num_workers = 0

    if num_workers is None:
        try:
            num_cpus = len(os.sched_getaffinity(0))
        except AttributeError:
            num_cpus = os.cpu_count() or 1
        num_workers = min(num_cpus, max_workers, max(n_files, 1))
        if n_files == 0:
            num_workers = 0

    kwargs = {"batch_size": batch_size, "num_workers": num_workers, "pin_memory": True}
    if num_workers > 0:
        kwargs["prefetch_factor"] = 2

    return DataLoader(dataset, **kwargs)
