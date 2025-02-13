import argparse
import ast
import copy
import gc
import hashlib
import json
import logging
import os
import pickle
import sys
import time
import warnings
from collections import Counter, OrderedDict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Union

from utils.plot_embeddings import plot_umap_categorical, plot_umap_numerical
from utils.logging import *

from tqdm import tqdm

from model.methyl_datasets import create_dataloader
from model.methyl_model import MethylGPTModel
from model.methyl_vocab import MethylVocab
from model.methyl_loss import masked_mse_loss


from common_setup import *

try:
    from flash_attn.flash_attention import FlashMHA

    flash_attn_available = True
except ImportError:
    import warnings

    warnings.warn("flash_attn is not installed")
    flash_attn_available = False


os.environ['CUDA_LAUNCH_BLOCKING']="1"
os.environ['TORCH_USE_CUDA_DSA'] = "1"

# Manually set default values
device = "cuda:0"
data_type = "3"
input_normalization = False
savename = "pretraining_test"

# Create parser for optional command-line overrides
parser = argparse.ArgumentParser()
parser.add_argument("-device", "--device", help="device", default=device)
parser.add_argument("-data_type", "--data_type", default=data_type)
parser.add_argument(
    "-savename",
    "--savename",
    type=str,
    default=savename,
    help="The name to save the output",
)

try:
    args = parser.parse_args()
except:
    args = argparse.Namespace()
    args.device = device
    args.data_type = data_type
    args.savename = savename


config= dict(
    # Important thing to control
    seed=42,
    input_type=f"CpGs_type{args.data_type}",
    parquet_dir=Path(f"../data/pretraining/processed_type{args.data_type}_parquet_shuffled"),
    probe_id_dir=Path(f"../data/pretraining/probe_ids_type{args.data_type}.csv"),
    qced_data_table=Path(f"../data/pretraining/QCed_samples_type{args.data_type}.csv"),
    compiled_data_dir=Path(f"/home/A.Y/project/MethylGPT_clean/data/pretraining/compiled_metadata.csv"),
    valid_ratio=0.1,
    n_hvg=n_hvg_predefined[args.data_type],  
    max_fi=500000,  # To use full dataset, Just set >500000
    do_train=True,
    pretrained_file=None,  # None for pretraining from scratch
    mask_ratio=0.3,
    GEPC=True,  # Masked value prediction for cell embedding
    dab_weight=1.0,
    pretraining_dataset_name=f"CpGs_type{args.data_type}",

    # Model and training
    epochs=100,
    ecs_thres=0.0,  # Elastic cell similarity objective, 0.0 to 1.0, 0.0 to disable
    lr=1e-3,
    batch_size=32,  #4,  
    layer_size=64, #16,
    nlayers=6, #4,
    nhead=4,
    dropout=0.1,
    schedule_ratio=0.9,  # ratio of epochs for learning rate schedule
    save_eval_interval=10,
    log_interval=1000,
    fast_transformer=True,
    pre_norm=False,
    amp=True,  # Automatic Mixed Precision


   # Additional tokens and values
    pad_token="<pad>",
    special_tokens=["<pad>", "<cls>", "<eoc>"],
    mask_value=-1,
    pad_value=-2,
    explicit_zero_prob=False,  # Flag for explicit zero probability
    max_seq_len=n_hvg_predefined[args.data_type] + 1,  # Calculated max sequence length
    per_seq_batch_sample=False,  # Flag for per-sequence batch sampling
)

config_hash = make_hash(config)

checkpoint_dir = './checkpoints'
os.makedirs(checkpoint_dir, exist_ok=True)
set_seed(config["seed"])


dataset_name=config["pretraining_dataset_name"]
save_dir = Path(f"../save/dev_{args.savename}-dataset_{dataset_name}-{time.strftime('%b%d-%H-%M')}/")
save_config(config, save_dir)

logger = setup_logger("logger", save_dir / "run.log")
add_console_handler(logger)
train_logger = setup_logger("train_logger", save_dir / "train.log")
add_console_handler(train_logger)
test_logger = setup_logger("test_logger", save_dir / "test.log")
add_console_handler(test_logger)


parquet_dirs = [
    os.path.join(config["parquet_dir"], f) for f in os.listdir(config["parquet_dir"]) if f.endswith(".parquet")
]
print(f"Number of parquet files: {len(parquet_dirs)}")

train_files, valid_files = split_files(parquet_dirs, valid_ratio=config["valid_ratio"])
train_dataloader = create_dataloader(train_files, config["batch_size"])
valid_dataloader = create_dataloader(valid_files, config["batch_size"])


methyl_vocab = MethylVocab(config["probe_id_dir"], config["pad_token"], config["special_tokens"], save_dir)


device = torch.device(args.device if torch.cuda.is_available() else "cpu")
model = MethylGPTModel(config, methyl_vocab)

if config["pretrained_file"] is not None:
    model = model.from_pretrained(config, methyl_vocab)
model.to(device)


criterion = masked_mse_loss
criterion_dab = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    model.parameters(), lr=config["lr"], eps=1e-4 if config["amp"] else 1e-8
)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=config["schedule_ratio"])
scaler = torch.cuda.amp.GradScaler(enabled=config["amp"])


def train(model, dataloader, optimizer, criterion, device, config, train_logger):

    pad_value=config['pad_value']
    mask_value=config['mask_value']

    model.train()
    total_loss = 0.0
    total_mse = 0.0
    total_gepc = 0.0
    total_num =0
    log_interval = config["log_interval"]
    start_time = time.time()


    for index_batch, batch in enumerate(dataloader):
        
        #if index_batch >= 4:  # Stop after processing 4 batches
        #        break
        optimizer.zero_grad()

        # Prepare data
        batch_data = model.prepare_data(batch)
        input_gene_ids = batch_data["gene_ids"].to(device)
        input_values = batch_data["values"].to(device)
        target_values = batch_data["target_values"].to(device)
        src_key_padding_mask = target_values.eq(pad_value)

        with torch.cuda.amp.autocast(enabled=config["amp"]):
            output_dict = model(
                input_gene_ids,
                input_values,
                src_key_padding_mask=src_key_padding_mask,
                MVC=config["GEPC"],
                ECS=config["ecs_thres"] > 0,
            )
            # Calculate loss 
            imputed_positions = target_values.eq(pad_value).to(device)
            masked_positions =  input_values.eq(mask_value).to(device)
            loss_positions = torch.logical_and(~imputed_positions,masked_positions)

            loss = loss_mse = criterion(output_dict["mlm_output"], target_values, loss_positions)

            if config["GEPC"]:
                loss_gepc = criterion(
                    output_dict["mvc_output"], target_values, loss_positions
                )
                loss = loss + loss_gepc

        # Backward pass and optimization
        
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        total_loss += loss.item()
        
        total_loss += loss.item() * len(input_gene_ids)
        total_mse += loss_mse.item() * len(input_gene_ids)
        total_gepc += loss_gepc.item() * len(input_gene_ids) if config["GEPC"] else 0.0
        total_num += len(input_gene_ids)
        elapsed_time = time.time() - start_time
    
        if index_batch % log_interval == 0:
            train_logger.info(
                f"Total Loss: {total_loss / total_num:.5f} | "
                f"Total MSE: {total_mse / total_num :.5f} | "
                f"Total GEPC: {total_gepc / total_num:.5f} | "
                f"Elapsed Time: {elapsed_time:5.2f}s"
            )
            total_loss = 0
            total_mse = 0
            total_gepc = 0
            total_num = 0
            start_time = time.time()
    
    return total_loss / total_num, total_mse / total_num, total_gepc / total_num

compiled_data = pd.read_csv(config["compiled_data_dir"])

def evaluate(model, dataloader, optimizer, criterion, device, config, test_logger,  epoch=0):

    pad_value=config['pad_value']
    mask_value=config['mask_value']

    model.eval()
    total_loss = 0.0
    total_mse = 0.0
    total_gepc = 0.0
    total_num =0
    start_time = time.time()

    all_data = dataloader.dataset.dataset
    all_ids = [d["id"] for d in all_data]
    all_embs = []

    with torch.no_grad():
        for index_batch, batch in enumerate(dataloader):

            # Prepare data
            batch_data = model.prepare_data(batch)
            input_gene_ids = batch_data["gene_ids"].to(device)
            input_values = batch_data["values"].to(device)
            target_values = batch_data["target_values"].to(device)

            # Forward pass
            src_key_padding_mask = target_values.eq(pad_value)
            with torch.cuda.amp.autocast(enabled=config["amp"]):
                output_dict = model(input_gene_ids, input_values, src_key_padding_mask= src_key_padding_mask)
                all_embs.append(output_dict["cell_emb"].cpu().numpy())

                # Calculate loss            
                imputed_positions = target_values.eq(pad_value).to(device)
                masked_positions =  input_values.eq(mask_value).to(device)
                loss_positions = torch.logical_and(~imputed_positions,masked_positions)                
                loss = loss_mse = criterion(output_dict["mlm_output"], target_values, loss_positions)

                if config["GEPC"]:
                    loss_gepc = criterion(
                        output_dict["mvc_output"], target_values, loss_positions
                    )
                    loss = loss + loss_gepc

            total_loss += loss.item()
            
            total_loss += loss.item() * len(input_gene_ids)
            total_mse += loss_mse.item() * len(input_gene_ids)
            total_gepc += loss_gepc.item() * len(input_gene_ids) if config["GEPC"] else 0.0
            total_num += len(input_gene_ids)
            elapsed_time = time.time() - start_time
        
        test_logger.info(
            f"Total Loss: {total_loss / total_num:.5f} | "
            f"Total MSE: {total_mse / total_num :.5f} | "
            f"Total GEPC: {total_gepc / total_num:.5f} | "
            f"Elapsed Time: {elapsed_time:5.2f}s"
        )

        # making the umap and plot
        all_embs = np.concatenate(all_embs, axis=0)
        umap_model = umap.UMAP(n_components=2, random_state=42)
        umap_emb = umap_model.fit_transform(all_embs)
        cell_list = all_ids

        cell_emb_df = pd.DataFrame(umap_emb, index=cell_list)
        cell_emb_df.reset_index(inplace=True)
        cell_emb_df.columns = ["GSM_ID", "UMAP1", "UMAP2"]

        merged_data = pd.merge(compiled_data, cell_emb_df, on="GSM_ID")

        plot_umap_categorical(
            "tissue",
            merged_data,
            save_as=f"{save_dir}/embeding_tissue_e{epoch}.png",
        )


    return total_loss / total_num

##
def get_latest_checkpoint(checkpoint_dir, config_hash):
    # Get all checkpoint files for the given config hash
    checkpoint_files = Path(checkpoint_dir).rglob(f'checkpoint_{config_hash}_*.pth')

    # Get the paths and epochs from the file names
    checkpoints = [(path, int(path.stem.split('_')[-1])) for path in checkpoint_files]

    # If there are no checkpoints, return None
    if not checkpoints:
        return None

    # Sort the checkpoints by epoch and return the path of the latest one
    latest_checkpoint_path, _ = max(checkpoints, key=lambda x: x[1])
    return str(latest_checkpoint_path)

# Check if a checkpoint exists and load it
latest_checkpoint_path = get_latest_checkpoint(checkpoint_dir, config_hash)  # you need to implement this function
if latest_checkpoint_path is not None:
    checkpoint = torch.load(latest_checkpoint_path)
    start_epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])  # load scheduler state
    # load more state here if necessary
else:
    start_epoch = 1


# Main training loop
best_val_loss = float("inf")
best_model = None

for epoch in range(start_epoch, config["epochs"] + 1):
    print(f"Epoch {epoch} started")
    epoch_start_time = time.time()

    train_loss = train(model, train_dataloader, optimizer, criterion, device, config, train_logger)
    val_loss = evaluate(model, valid_dataloader, optimizer,  criterion, device, config, test_logger, epoch=epoch)

    elapsed = time.time() - epoch_start_time

    logger.info(
        f"Epoch {epoch}/{config['epochs']} - Train Loss: {train_loss:.4f} - Time: {elapsed:.2f}s"
    )

    # After each epoch, save the model state
    checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_{config_hash}_{epoch}.pth')
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),  # save scheduler state
        # add more state here if necessary
    }, checkpoint_path)

    # Save the model
    if epoch % config["save_eval_interval"] == 0 or epoch == config["epochs"]:
        torch.save(model.state_dict(), save_dir / f"model_epoch{epoch}.pt")
        
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model = copy.deepcopy(model)
        best_model_epoch = epoch
        logger.info(f"Best model with score {best_val_loss:5.4f}")
    
    torch.save(best_model.state_dict(), save_dir / f"best_model_epoch{epoch}.pt")
    
    scheduler.step()