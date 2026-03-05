import json
import logging
import os
import pickle
import warnings
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from methylgpt.model.methyl_datasets import create_dataloader
from methylgpt.model.methyl_model import MethylGPTModel
from methylgpt.model.methyl_vocab import MethylVocab

try:
    from flash_attn.flash_attention import FlashMHA
    flash_attn_available = True
except ImportError:
    import warnings
    warnings.warn("flash_attn is not installed")
    flash_attn_available = False

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ['TORCH_USE_CUDA_DSA'] = "1"


def generate_cell_embeddings(model, data_loader, device, vocab, max_seq_len, config, mask_value, pad_value, pad_token):
    """
    Generate cell embeddings using the provided model and data loader.

    Args:
        model (torch.nn.Module): The model to generate embeddings.
        data_loader (torch.utils.data.DataLoader): DataLoader containing the dataset.
        device (torch.device): The device to run the model on.
        vocab (dict): Vocabulary dictionary.
        max_seq_len (int): Maximum sequence length.
        config (object): Configuration object containing mask_ratio.
        mask_value (float): Value used for masking.
        pad_value (float): Value used for padding.
        pad_token (str): Token used for padding.

    Returns:
        np.ndarray: Array of cell embeddings.

    Raises:
        RuntimeError: If there's an error during embedding generation.
    """
    logger = logging.getLogger(__name__)
    cell_embs = []
    cell_ids = []
    
    logger.info("Generating embedding...")
    
    with torch.no_grad():
        for i, batch in enumerate(tqdm(data_loader, desc="Processing batches")):
            # Prepare data
            if i == 100:
                break
            batch_data = model.prepare_data(batch)
            
            input_gene_ids = batch_data["gene_ids"].to(device)
            input_values = batch_data["values"].to(device).half()
            
            src_key_padding_mask = input_gene_ids.eq(vocab[pad_token]).to(device)
            output_dict = model(
                input_gene_ids,
                input_values,
                src_key_padding_mask=src_key_padding_mask,
                MVC=config["GEPC"],
                ECS=config["ecs_thres"] > 0,
            )
            output_values = output_dict["cell_emb"].cpu().numpy()
            cell_embs.append(output_values)
            cell_ids.append(batch["id"])
            
            logger.debug(f"Batch embedding shape: {output_values.shape}")
    
    cell_emb = np.concatenate(cell_embs, axis=0)
    cell_list = np.concatenate(cell_ids, axis=0)
    logger.info(f"Validset embedding shape: {cell_emb.shape}")
    return cell_emb, cell_list


def main():
    # Configuration and paths
    SAVE_DIR = Path('Embeddings')
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"save to {SAVE_DIR}")

    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    PARQUET_DIR = "data/processed_type3_parquet_shuffled"
    MODEL_PATH_DIR = "pretrained_models/methylgpt-medium"
    # Find the model checkpoint file (name varies by model version)
    import glob as _glob
    model_files = _glob.glob(MODEL_PATH_DIR + "/*.pt")
    assert model_files, f"No .pt files found in {MODEL_PATH_DIR}"
    MODEL_DIR = model_files[0]
    CPG_LIST_DIR = "probe_ids_type3.csv"

    # Load from config file
    with open(Path(MODEL_PATH_DIR + "/args.json"), "r") as f:
        config = json.load(f)

    print(config)

    # Update config dict
    config["load_model"] = True
    config["batch_size"] = 32
    config["model_file"] = MODEL_DIR
    config["mask_ratio"] = 0
    config["probe_id_dir"] = CPG_LIST_DIR

    pad_token = "<pad>"
    special_tokens = [pad_token, "<cls>", "<eoc>"]
    
    # Add tokens to config
    config["pad_token"] = pad_token
    config["special_tokens"] = special_tokens

    mask_ratio = config["mask_ratio"]
    mask_value = -1
    pad_value = -2
    
    # Add mask and pad values to config
    config["mask_value"] = mask_value
    config["pad_value"] = pad_value

    # Number of highly variable CpG sites
    n_hvg = config["n_hvg"]  
    max_seq_len = n_hvg + 1
    config["max_seq_len"] = max_seq_len

    per_seq_batch_sample = False
    DSBN = True  # Domain-spec batchnorm
    explicit_zero_prob = False  # Whether explicit bernoulli for zeros

    # Create data loader
    parquet_dirs = [
        os.path.join(PARQUET_DIR, f) for f in os.listdir(PARQUET_DIR)
    ]
    valid_dataloader = create_dataloader([parquet_dirs[0]], config["batch_size"])

    # Initialize vocabulary and model
    methyl_vocab = MethylVocab(config["probe_id_dir"], config["pad_token"], config["special_tokens"], save_dir=None)
    model = MethylGPTModel(config, methyl_vocab)

    # Load model weights
    try:
        model.load_state_dict(torch.load(MODEL_DIR, map_location="cpu"))
        print(f"Loading all model params from {MODEL_DIR}")
    except:
        # Only load params that are in the model and match the size
        model_dict = model.state_dict()
        pretrained_dict = torch.load(MODEL_DIR, map_location="cpu")
        pretrained_dict = {
            k: v
            for k, v in pretrained_dict.items()
            if k in model_dict and v.shape == model_dict[k].shape
        }
        for k, v in pretrained_dict.items():
            print(f"Loading params {k} with shape {v.shape}")
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    model.eval()  # Switch to evaluation mode (turns off dropout, etc.)
    model.to(DEVICE)
    model.half()

    for name, param in model.named_parameters():
        print(name, param.dtype)

    # Generate embeddings
    valid_cell_emb, valid_cell_list = generate_cell_embeddings(
        model, 
        valid_dataloader, 
        DEVICE,
        methyl_vocab,
        max_seq_len,
        config,
        mask_value,
        pad_value,
        pad_token
    )

    # Save embeddings
    valid_emb_path = SAVE_DIR / "cell_emb.pt"
    with open(valid_emb_path, "wb") as file:
        pickle.dump({"cell_emb": valid_cell_emb, "cell_list": valid_cell_list}, file)

    print(f"Embeddings saved to {valid_emb_path}")


if __name__ == "__main__":
    main()
