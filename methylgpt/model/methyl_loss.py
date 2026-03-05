"""Loss functions for MethylGPT pretraining and finetuning."""

import torch
import torch.nn.functional as F


def masked_mse_loss(input: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Compute masked mean squared error loss.

    Only positions where ``mask`` is nonzero contribute to the loss.
    This is used for the masked value prediction pretraining objective.

    Args:
        input: Predicted values of shape ``(batch_size, seq_len)``.
        target: Ground truth values of shape ``(batch_size, seq_len)``.
        mask: Binary mask of shape ``(batch_size, seq_len)`` where 1 indicates
            positions to include in the loss computation.

    Returns:
        Scalar tensor with the mean squared error over masked positions.
    """
    mask = mask.float()
    loss = F.mse_loss(input * mask, target * mask, reduction="sum")
    return loss / mask.sum()


def criterion_neg_log_bernoulli(input: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Compute masked negative log-likelihood of Bernoulli distribution.

    Used for binary methylation state prediction where values are
    thresholded to 0/1.

    Args:
        input: Predicted probabilities of shape ``(batch_size, seq_len)``.
        target: Ground truth values of shape ``(batch_size, seq_len)``.
        mask: Binary mask of shape ``(batch_size, seq_len)``.

    Returns:
        Scalar tensor with the negative log-likelihood over masked positions.
    """
    mask = mask.float()
    bernoulli = torch.distributions.Bernoulli(probs=input)
    masked_log_probs = bernoulli.log_prob((target > 0).float()) * mask
    return -masked_log_probs.sum() / mask.sum()


def masked_relative_error(input: torch.Tensor, target: torch.Tensor, mask: torch.LongTensor) -> torch.Tensor:
    """Compute masked mean relative error between input and target.

    Args:
        input: Predicted values of shape ``(batch_size, seq_len)``.
        target: Ground truth values of shape ``(batch_size, seq_len)``.
        mask: Boolean mask of shape ``(batch_size, seq_len)``.

    Returns:
        Scalar tensor with the mean relative error over masked positions.

    Raises:
        AssertionError: If no positions are masked (``mask.any()`` is False).
    """
    assert mask.any()
    loss = torch.abs(input[mask] - target[mask]) / (target[mask] + 1e-6)
    return loss.mean()
