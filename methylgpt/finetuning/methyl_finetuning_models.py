"""Finetuning model heads and Lightning modules for MethylGPT downstream tasks.

Provides prediction heads (age regression, classification) and Lightning
modules that combine a pretrained MethylGPT encoder with task-specific heads.
"""

import logging

import lightning as pl
import torch
import torch.nn.functional as F
import torch.optim as optim
from scgpt.model.model import TransformerModel
from torch import nn

from methylgpt.finetuning.finetuning_metrics import regression_metric

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prediction Heads
# ---------------------------------------------------------------------------


class RegressionHead(nn.Module):
    """MLP head for regression tasks (e.g., age prediction).

    A 3-layer MLP with batch normalization, dropout, and ReLU activations.
    Final output passes through sigmoid to constrain to [0, 1] range,
    suitable for normalized age targets.

    Args:
        input_dim: Input feature dimension (embedding size).
        hidden_dim: Hidden layer dimension.
        output_dim: Output dimension (1 for scalar regression).
        dropout: Dropout rate. Default: 0.1.
    """

    def __init__(self, input_dim, hidden_dim, output_dim=1, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim),
            nn.BatchNorm1d(input_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim, output_dim),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


class ClassificationHead(nn.Module):
    """MLP head for binary or multi-label classification.

    A deep MLP with SELU activations and Kaiming initialization,
    suitable for disease classification tasks.

    Args:
        input_dim: Input feature dimension.
        hidden_dim: Hidden layer dimension.
        output_dim: Number of output classes.
        dropout: Dropout rate. Default: 0.1.
    """

    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.BatchNorm1d(input_dim),
            nn.Dropout(dropout),
            nn.Linear(input_dim, hidden_dim),
            nn.SELU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SELU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="leaky_relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.net(x)


class EmbeddingReductionHead(nn.Module):
    """Head that reduces per-token embeddings to a single vector, then predicts.

    First reduces the embedding dimension via a linear layer (squeeze),
    then passes through a classification/regression head.

    Args:
        emb_dim: Per-token embedding dimension.
        input_dim: Sequence length or number of tokens.
        hidden_dim: Hidden layer dimension.
        output_dim: Output dimension.
        dropout: Dropout rate. Default: 0.1.
    """

    def __init__(self, emb_dim, input_dim, hidden_dim, output_dim, dropout=0.1):
        super().__init__()
        self.reduction = nn.Linear(emb_dim, 1)
        self.head = ClassificationHead(input_dim, hidden_dim, output_dim, dropout)

    def forward(self, x):
        x = self.reduction(x).squeeze(-1)
        return self.head(x)


# ---------------------------------------------------------------------------
# Lightning Module
# ---------------------------------------------------------------------------


class FintuneModel(pl.LightningModule):
    """General-purpose finetuning module for MethylGPT.

    Combines a pretrained MethylGPT encoder with a task-specific prediction
    head. Supports age regression with normalized targets and inverse scaling.

    Args:
        model_args: Configuration dictionary with:
            - ``layer_size``, ``nhead``, ``nlayers``, ``dropout``: Architecture.
            - ``fast_transformer``, ``pre_norm``: Transformer options.
            - ``pretrained_file`` (str): Path to pretrained checkpoint.
            - ``pretrained_lr`` (float): Learning rate for encoder.
            - ``head_lr`` (float): Learning rate for prediction head.
            - ``weight_decay`` (float): Weight decay for encoder.
            - ``ecs_thres`` (float): ECS threshold.
        vocab: A ``CollatableVocab`` instance.
        head: A ``nn.Module`` prediction head.
        scaler: A fitted sklearn scaler for inverse-transforming predictions.
            If ``None``, no inverse scaling is applied.
    """

    def __init__(self, model_args, vocab, head, scaler=None):
        super().__init__()
        self.vocab = vocab
        self.model_args = model_args
        self.scaler = scaler
        self.head = head
        self.validation_step_outputs = []

        self.encoder = self._load_pretrained(model_args, vocab)

    def forward(self, gene_ids, values):
        """Forward pass: encode + predict.

        Args:
            gene_ids: Tensor of CpG token IDs ``(batch, seq_len)``.
            values: Tensor of methylation values ``(batch, seq_len)``.

        Returns:
            Prediction tensor from the head.
        """
        cell_embs = self._get_cell_embeddings(gene_ids, values)
        return self.head(cell_embs)

    def training_step(self, batch, batch_idx):
        gene_id, value, label = batch[:3]
        pred = self(gene_id, value).squeeze()
        loss = F.mse_loss(pred, label.squeeze())
        self.log("train_loss", loss, prog_bar=True, sync_dist=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        gene_id, value, label = batch[:3]
        pred = self(gene_id, value).squeeze()
        loss = F.mse_loss(pred, label.squeeze())
        self.log("val_loss", loss, prog_bar=True, sync_dist=True, on_epoch=True)

        # Inverse scale if scaler is provided
        if self.scaler is not None:
            pred_orig = torch.tensor(
                self.scaler.inverse_transform(pred.detach().cpu().numpy().reshape(-1, 1))
            ).squeeze()
        else:
            pred_orig = pred.detach().cpu()

        self.validation_step_outputs.append(
            {
                "pred_age": pred_orig,
                "label": label.detach().cpu(),
            }
        )

    def on_validation_epoch_end(self):
        metrics = regression_metric(self.validation_step_outputs)
        for key, value in metrics.items():
            self.log(f"val_{key}", value, prog_bar=True, on_epoch=True, sync_dist=True)
        self.validation_step_outputs.clear()

    def configure_optimizers(self):
        params = [
            {
                "params": self.encoder.parameters(),
                "lr": self.model_args["pretrained_lr"],
                "weight_decay": self.model_args.get("weight_decay", 0.0),
            },
            {
                "params": self.head.parameters(),
                "lr": self.model_args["head_lr"],
            },
        ]
        return optim.Adam(params)

    def _get_cell_embeddings(self, gene_ids, values):
        """Extract cell embeddings from encoder.

        Args:
            gene_ids: Token ID tensor.
            values: Value tensor.

        Returns:
            Cell embedding tensor.
        """
        src_key_padding_mask = gene_ids.eq(self.vocab.vocab[self.vocab.pad_token])
        output_dict = self.encoder(
            gene_ids,
            values,
            src_key_padding_mask=src_key_padding_mask,
            batch_labels=None,
        )
        return output_dict["cell_emb"]

    @staticmethod
    def _load_pretrained(model_args, vocab):
        """Load a pretrained TransformerModel encoder.

        Args:
            model_args: Configuration dictionary.
            vocab: CollatableVocab instance.

        Returns:
            A ``TransformerModel`` with pretrained weights.
        """
        model = TransformerModel(
            len(vocab.vocab),
            model_args["layer_size"],
            model_args["nhead"],
            model_args["layer_size"],
            model_args["nlayers"],
            vocab=vocab.vocab,
            dropout=model_args["dropout"],
            pad_token=vocab.pad_token,
            pad_value=vocab.pad_value,
            do_mvc=True,
            do_dab=False,
            use_batch_labels=False,
            num_batch_labels=None,
            domain_spec_batchnorm=False,
            n_input_bins=None,
            ecs_threshold=model_args.get("ecs_thres", None),
            explicit_zero_prob=False,
            use_fast_transformer=model_args.get("fast_transformer", False),
            pre_norm=model_args.get("pre_norm", False),
        )

        pretrained_file = model_args.get("pretrained_file")
        if pretrained_file:
            state_dict = torch.load(pretrained_file, map_location="cpu", weights_only=True)
            try:
                model.load_state_dict(state_dict)
                logger.info(f"Loaded all model params from {pretrained_file}")
            except RuntimeError:
                model_dict = model.state_dict()
                compatible = {k: v for k, v in state_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
                skipped = set(state_dict.keys()) - set(compatible.keys())
                if skipped:
                    logger.warning(f"Skipped {len(skipped)} incompatible params: {skipped}")
                model_dict.update(compatible)
                model.load_state_dict(model_dict)

        return model
