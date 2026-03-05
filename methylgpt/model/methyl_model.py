"""Core MethylGPT model implementation."""

import torch
from scgpt.model.model import TransformerModel
from scgpt.tokenizer import random_mask_value, tokenize_and_pad_batch


class MethylGPTModel(TransformerModel):
    """Transformer model for DNA methylation analysis.

    Extends the scGPT TransformerModel for methylation-specific tasks
    including pretraining with masked value prediction, cell-level
    embedding extraction, and downstream finetuning.

    Args:
        config: Model configuration dictionary containing:
            - ``layer_size`` (int): Dimension of transformer layers.
            - ``nhead`` (int): Number of attention heads.
            - ``nlayers`` (int): Number of transformer layers.
            - ``dropout`` (float): Dropout rate.
            - ``fast_transformer`` (bool): Whether to use flash attention.
            - ``pre_norm`` (bool): Whether to use pre-layer normalization.
        vocab: A ``MethylVocab`` instance with CpG site vocabulary.

    Example:
        >>> config = {"layer_size": 128, "nhead": 4, "nlayers": 4,
        ...           "dropout": 0.1, "fast_transformer": False, "pre_norm": False}
        >>> model = MethylGPTModel(config, vocab)
    """

    def __init__(self, config, vocab):
        # Ensure required config keys have sensible defaults
        config.setdefault("max_seq_len", config.get("n_hvg", 49156) + 1)
        config.setdefault("mask_ratio", 0.0)
        config.setdefault("mask_value", -1)
        config.setdefault("pad_token", "<pad>")
        config.setdefault("pad_value", -2)

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
            pre_norm=config["pre_norm"],
        )

        self.vocab = vocab
        self.config = config
        self.validation_step_outputs = []

    def get_cell_embeddings(self, gene_ids, values):
        """Extract cell-level embeddings from tokenized methylation data.

        Args:
            gene_ids: Tensor of shape ``(batch_size, seq_len)`` with CpG token IDs.
            values: Tensor of shape ``(batch_size, seq_len)`` with methylation values.

        Returns:
            Tensor of shape ``(batch_size, embedding_dim)`` with cell embeddings.
        """
        # Ensure values match model dtype (flash_attn requires float16/bfloat16)
        model_dtype = next(self.parameters()).dtype
        values = values.to(model_dtype)

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
    def from_pretrained(cls, config, vocab):
        """Load a pretrained MethylGPT model from a checkpoint file.

        Creates a new model instance and loads pretrained weights. If some
        parameters in the checkpoint do not match the model architecture
        (e.g., different layer sizes), only the compatible parameters are
        loaded and a warning is printed.

        Args:
            config: Model configuration dictionary. Must contain:
                - All keys required by ``__init__`` (layer_size, nhead, etc.)
                - ``load_model`` (bool): Whether to load pretrained weights.
                - ``pretrained_file`` (str): Path to the checkpoint ``.pt`` file.
            vocab: A ``MethylVocab`` instance.

        Returns:
            A ``MethylGPTModel`` instance with pretrained weights loaded.

        Raises:
            FileNotFoundError: If ``config["pretrained_file"]`` does not exist.
            RuntimeError: If the checkpoint cannot be loaded at all.
        """
        # Config defaults are set in __init__, so just create the model
        model = cls(config, vocab)
        if config.get("load_model", False):
            pretrained_file = config["pretrained_file"]
            state_dict = torch.load(pretrained_file, map_location="cpu", weights_only=True)
            try:
                model.load_state_dict(state_dict)
                print(f"Loading all model params from {pretrained_file}")
            except RuntimeError:
                # Only load params that are in the model and match the size
                model_dict = model.state_dict()
                compatible = {k: v for k, v in state_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
                skipped = set(state_dict.keys()) - set(compatible.keys())
                if skipped:
                    print(f"Warning: Skipped loading {len(skipped)} incompatible params: {skipped}")
                for k, v in compatible.items():
                    print(f"Loading params {k} with shape {v.shape}")
                model_dict.update(compatible)
                model.load_state_dict(model_dict)
        return model

    def prepare_data(self, batch):
        """Tokenize and mask a batch of methylation data for pretraining.

        Converts raw methylation values into tokenized, padded, and masked
        inputs suitable for the masked value prediction pretraining objective.

        Args:
            batch: Dictionary with key ``"data"`` containing a tensor of
                shape ``(batch_size, n_cpgs)`` with raw methylation beta values.

        Returns:
            Dictionary with:
                - ``"gene_ids"``: Tensor of CpG token IDs.
                - ``"values"``: Masked input values.
                - ``"target_values"``: Original (unmasked) target values.
        """
        # Retrieve config values
        max_seq_len = self.config["max_seq_len"]
        mask_ratio = self.config["mask_ratio"]
        mask_value = self.config["mask_value"]
        pad_token = self.config["pad_token"]
        pad_value = self.config["pad_value"]
        # Use config for these flags, with defaults matching common usage or your snippet
        append_cls = self.config.get("append_cls", True)
        include_zero_gene = self.config.get("include_zero_gene", True)  # From your snippet; scGPT often False

        # batch["data"] is a PyTorch Tensor [batch_size, num_features]
        methyl_data_tensor = batch["data"]

        # Ensure float type and handle NaNs using PyTorch operations
        methyl_data_tensor = methyl_data_tensor.float()
        methyl_data_tensor = torch.nan_to_num(methyl_data_tensor, nan=float(pad_value))

        # Convert to NumPy array for the tokenizer
        # Ensure tensor is on CPU before converting to NumPy
        methyl_data_numpy = methyl_data_tensor.cpu().numpy()

        # Tokenize the data
        # The second argument should be the list of all CpG IDs from the vocabulary.
        # Assuming self.vocab.CpG_ids holds this list, as per your provided snippet.
        if not hasattr(self.vocab, "CpG_ids"):
            # Fallback or error if CpG_ids is not the correct attribute name
            # This might be self.vocab.CpG_list or another name depending on MethylVocab implementation
            raise AttributeError(
                "MethylVocab instance does not have 'CpG_ids' attribute. "
                "Please verify attribute name for the list of all CpG sites."
            )

        tokenized_data = tokenize_and_pad_batch(
            methyl_data_numpy,  # First argument: batch data as 2D NumPy array
            self.vocab.CpG_ids,  # Second argument: list of all CpG IDs in the vocab space
            max_len=max_seq_len,
            vocab=self.vocab,
            pad_token=pad_token,
            pad_value=pad_value,
            append_cls=append_cls,
            include_zero_gene=include_zero_gene,
        )

        # Apply masking to the padded values
        masked_input_values = random_mask_value(
            tokenized_data["values"],
            mask_ratio=mask_ratio,
            mask_value=mask_value,
            pad_value=pad_value,
        )

        return {
            "gene_ids": tokenized_data["genes"],
            "values": masked_input_values,
            "target_values": tokenized_data["values"],
        }
