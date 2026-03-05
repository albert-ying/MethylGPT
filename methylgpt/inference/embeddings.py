"""Embedding extraction utilities for MethylGPT.

Provides functions to extract both sample-level (cell) embeddings and
CpG-level (token) embeddings from a pretrained MethylGPT model.
"""

import logging
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

logger = logging.getLogger(__name__)


def extract_embeddings(
    model,
    data_loader: DataLoader,
    device: str = "cuda",
    max_batches: Optional[int] = None,
) -> Tuple[np.ndarray, List[str]]:
    """Extract sample-level (cell) embeddings from a pretrained model.

    Runs the model in eval mode on all batches from the data loader and
    collects the ``cell_emb`` output for each sample.

    Args:
        model: A ``MethylGPTModel`` instance (already on device, or will be moved).
        data_loader: DataLoader yielding batches with ``"id"`` and ``"data"`` keys,
            as produced by ``create_dataloader``.
        device: Compute device (``"cuda"`` or ``"cpu"``). Default: ``"cuda"``.
        max_batches: If set, stop after this many batches (useful for testing).
            Default: ``None`` (process all batches).

    Returns:
        Tuple of:
            - ``embeddings``: numpy array of shape ``(n_samples, embedding_dim)``.
            - ``sample_ids``: list of sample ID strings.

    Example:
        >>> model = MethylGPTModel.from_pretrained(config, vocab)
        >>> model.eval().to("cuda")
        >>> loader = create_dataloader(parquet_files, batch_size=32)
        >>> embeddings, ids = extract_embeddings(model, loader, device="cuda")
        >>> embeddings.shape
        (1000, 128)
    """
    model = model.to(device)
    model.eval()

    all_embeddings = []
    all_ids = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(data_loader, desc="Extracting embeddings")):
            if max_batches is not None and batch_idx >= max_batches:
                break

            batch_data = model.prepare_data(batch)
            gene_ids = batch_data["gene_ids"].to(device)
            values = batch_data["values"].to(device)

            # Cast values to model dtype (flash_attn requires float16/bfloat16)
            model_dtype = next(model.parameters()).dtype
            values = values.to(model_dtype)

            cell_embs = model.get_cell_embeddings(gene_ids, values)
            all_embeddings.append(cell_embs.cpu().numpy())
            all_ids.extend(batch["id"])

    embeddings = np.concatenate(all_embeddings, axis=0)
    logger.info(f"Extracted embeddings: shape={embeddings.shape}")
    return embeddings, all_ids


def extract_cpg_embeddings(model) -> np.ndarray:
    """Extract CpG-level (token) embeddings from the model's embedding layer.

    Returns the learned embedding vectors for each CpG site in the vocabulary.
    These can be used for CpG similarity analysis, clustering, and annotation.

    Args:
        model: A ``MethylGPTModel`` instance with loaded weights.

    Returns:
        Numpy array of shape ``(n_tokens, embedding_dim)`` containing the
        embedding matrix. Token indices correspond to the vocabulary order
        (special tokens first, then CpG sites).

    Example:
        >>> model = MethylGPTModel.from_pretrained(config, vocab)
        >>> cpg_embs = extract_cpg_embeddings(model)
        >>> cpg_embs.shape  # (n_special_tokens + n_cpgs, embedding_dim)
        (49159, 128)
    """
    model.eval()
    with torch.no_grad():
        # The encoder's embedding layer holds the token embeddings
        embedding_layer = model.encoder.embedding
        token_embeddings = embedding_layer.weight.detach().cpu().numpy()

    logger.info(f"Extracted CpG embeddings: shape={token_embeddings.shape}")
    return token_embeddings
