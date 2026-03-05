"""Pretraining utilities for MethylGPT masked value prediction."""

from typing import Union

import numpy as np
import torch


def random_mask_value(
    values: Union[torch.Tensor, np.ndarray],
    mask_ratio: float = 0.15,
    mask_value: int = -1,
    pad_value: int = 0,
) -> torch.Tensor:
    """Randomly mask a fraction of non-padding values in each sample.

    Used during pretraining to create the masked value prediction task.
    For each row, a random subset of non-padding positions is replaced
    with ``mask_value``.

    Args:
        values: Input values of shape ``(batch_size, seq_len)``. Can be a
            torch Tensor or numpy array.
        mask_ratio: Fraction of non-padding values to mask. Default: 0.15.
        mask_value: Value to assign to masked positions. Default: -1.
        pad_value: Value that indicates padding (excluded from masking).
            Default: 0.

    Returns:
        Float tensor of shape ``(batch_size, seq_len)`` with masked values.

    Example:
        >>> values = torch.tensor([[0.5, 0.3, 0.8, -2.0], [0.1, 0.9, -2.0, -2.0]])
        >>> masked = random_mask_value(values, mask_ratio=0.5, pad_value=-2.0)
        >>> masked.shape
        torch.Size([2, 4])
    """
    if isinstance(values, torch.Tensor):
        # Clone to avoid modifying the original tensor
        values = values.clone().detach().numpy()
    else:
        values = values.copy()

    for i in range(len(values)):
        row = values[i]
        non_padding_idx = np.nonzero(row - pad_value)[0]
        n_mask = int(len(non_padding_idx) * mask_ratio)
        mask_idx = np.random.choice(non_padding_idx, n_mask, replace=False)
        row[mask_idx] = mask_value

    return torch.from_numpy(values).float()
