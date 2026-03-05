import os
import sys
from pathlib import Path
import methylgpt.modules.scGPT.scgpt as scgpt
# Get the directory of the current script
script_dir = Path(__file__).parent.absolute()
from sklearn import preprocessing
import pandas as pd
import argparse
import json
import yaml
import torch
import lightning as pl
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch import seed_everything
from finetuning_age_datasets import CollatableVocab, Age_Dataset
from finetuning_age_models import methyGPT_Age_Model

seed_everything(42, workers=True)

# Debug: Print CUDA_VISIBLE_DEVICES and GPU information
print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")
print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"torch.cuda.device_count(): {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

# Helper function to resolve paths relative to the script directory
def resolve_path(path_str):
    if not path_str:
        return None
    path_obj = Path(path_str)
    if path_obj.is_absolute():
        return str(path_obj)
    return str(script_dir / path_obj)

def train(cli_args, config_from_yaml): # Changed signature to accept the whole YAML config

    # Resolve paths for config files and directories
    # Get pretrained_model_args_path directly from the loaded YAML config
    pretrained_model_args_path_str = config_from_yaml.get("pretrained_model_args_path")
    pretrained_model_args_path = resolve_path(pretrained_model_args_path_str)

    if not pretrained_model_args_path or not Path(pretrained_model_args_path).exists():
        raise FileNotFoundError(f"Pretrained model args file not found: {pretrained_model_args_path} (from YAML key 'pretrained_model_args_path': '{pretrained_model_args_path_str}')")
    with open(pretrained_model_args_path, 'r') as f:
        pretrain_model_config_from_json = json.load(f)
    
    # Combine configurations: 
    # 1. Start with pretrained model's args.json (for model architecture)
    # 2. Override with YAML config (for training parameters)
    # 3. Override with CLI arguments (highest priority)
    combined_args = pretrain_model_config_from_json.copy() # Start with pretrained model config
    
    # Override with YAML config, giving priority to training-specific parameters
    combined_args.update(config_from_yaml)
    
    # Specifically preserve batch size from YAML config if present
    if 'train_batch_size' in config_from_yaml:
        combined_args['batch_size'] = config_from_yaml['train_batch_size']
    if 'valid_batch_size' in config_from_yaml:
        combined_args['valid_batch_size'] = config_from_yaml['valid_batch_size']

    # Override with CLI arguments
    for key, value in vars(cli_args).items():
        if value is not None:  # Only override if CLI arg was explicitly provided
            combined_args[key] = value

    # Override with CLI arguments
    if cli_args.gpus is not None:
        combined_args['gpus'] = cli_args.gpus
    if cli_args.output_dir is not None:
        combined_args['output_dir'] = cli_args.output_dir
    else: # Ensure output_dir has a default if not in CLI or YAML
        combined_args['output_dir'] = combined_args.get('output_dir', "./output_finetune")
        
    if cli_args.probe_id_file is not None:
        combined_args['probe_id_file'] = cli_args.probe_id_file
    # Ensure probe_id_file has a default if not in CLI or YAML
    elif 'probe_id_file' not in combined_args:
        combined_args['probe_id_file'] = "./probe_ids_type3.csv" # Default relative to script

    if cli_args.data_dir is not None:
        combined_args['data_dir'] = cli_args.data_dir
    # Ensure data_dir has a default if not in CLI or YAML
    elif 'data_dir' not in combined_args:
        combined_args['data_dir'] = "./data" # Default relative to script

    # Mask ratio and seed: CLI > YAML > Default
    if cli_args.mask_ratio is not None:
        combined_args['mask_ratio'] = cli_args.mask_ratio
    elif 'mask_ratio' not in combined_args: # If not in CLI or YAML
        combined_args['mask_ratio'] = 0.15 # Default mask_ratio
    # Ensure it's float
    combined_args['mask_ratio'] = float(combined_args['mask_ratio'])

    if cli_args.mask_seed is not None:
        combined_args['mask_seed'] = cli_args.mask_seed
    elif 'mask_seed' not in combined_args: # If not in CLI or YAML
        combined_args['mask_seed'] = 42   # Default mask_seed
    # Ensure it's int
    combined_args['mask_seed'] = int(combined_args['mask_seed'])

    # Resolve data_dir 
    data_dir = resolve_path(combined_args["data_dir"])
    if not data_dir or not Path(data_dir).exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir} (from combined_args['data_dir'] '{combined_args['data_dir']}')")

    # Resolve output_dir
    output_dir = resolve_path(combined_args['output_dir'])
    Path(output_dir).mkdir(parents=True, exist_ok=True) # Ensure output directory exists

    # Resolve probe_id_file path
    probe_id_file_str = combined_args.get('probe_id_file')
    probe_id_file = resolve_path(probe_id_file_str)
    if not probe_id_file or not Path(probe_id_file).exists():
        # Try to default to a file in the script_dir if not found
        default_probe_file = script_dir / "probe_ids_type3.csv"
        if default_probe_file.exists():
            print(f"Warning: Probe ID file '{probe_id_file}' (from '{probe_id_file_str}') not found. Using default: {default_probe_file}")
            probe_id_file = str(default_probe_file)
        else:
            raise FileNotFoundError(f"Probe ID file not found: {probe_id_file} (from '{probe_id_file_str}') and default {default_probe_file} also not found.")
    combined_args['probe_id_file'] = probe_id_file # Store resolved path

    # Update version and weights_name using combined_args
    dataset_name = combined_args.get("dataset", "dataset") # Default dataset name
    mask_ratio_for_name = combined_args['mask_ratio']
    mask_seed_for_name = combined_args['mask_seed']

    combined_args["version"] = f'Finetune-methylGPT-{dataset_name}-mask{mask_ratio_for_name*100:.0f}percent-seed{mask_seed_for_name}'
    combined_args["weights_name"] = combined_args["version"] + '_{epoch:02d}-{step:02d}-{valid_medae:.4f}-{valid_mae:.4f}-{valid_s_r:.4f}-{test_medae:.4f}-{test_mae:.4f}-{test_s_r:.4f}'
    combined_args["dropout"] = combined_args.get("dropout", 0) # Default dropout if not set

    # Prepare data
    methyGPT_vocab = CollatableVocab(combined_args)
    
    train_df = pd.read_parquet(Path(data_dir) / combined_args["train_file_name"]) # Use combined_args directly
    valid_df = pd.read_parquet(Path(data_dir) / combined_args["valid_file_name"]) # Use combined_args directly
    test_df = pd.read_parquet(Path(data_dir) / combined_args["test_file_name"])   # Use combined_args directly


    scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
    scaler.fit(train_df["age"].to_numpy().reshape(-1, 1))
    train_dataset = Age_Dataset(methyGPT_vocab, train_df, scaler)
    valid_dataset = Age_Dataset(methyGPT_vocab, valid_df, scaler)
    test_dataset = Age_Dataset(methyGPT_vocab, test_df, scaler)
        
    train_loader: torch.utils.data.DataLoader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=combined_args["train_batch_size"],
        collate_fn=train_dataset.collater,
        shuffle=True,
        drop_last=True,
        num_workers=combined_args.get("num_workers", 4), # Default num_workers
    )
    
    valid_loader: torch.utils.data.DataLoader = torch.utils.data.DataLoader(
        valid_dataset,
        collate_fn=valid_dataset.collater,
        batch_size=combined_args["valid_batch_size"],
        num_workers=combined_args.get("num_workers", 4),
    )    
    
    test_loader: torch.utils.data.DataLoader = torch.utils.data.DataLoader(
        test_dataset,
        collate_fn=test_dataset.collater,
        batch_size=combined_args["valid_batch_size"],
        num_workers=combined_args.get("num_workers", 4),
    )
    
    # Init model
    model = methyGPT_Age_Model(
                model_args=combined_args, 
                vocab=methyGPT_vocab,
                scaler=scaler,
                )
    
    # DEBUG: Check model device after instantiation
    print(f"DEBUG: Model created on device: {next(model.parameters()).device}")
    print(f"DEBUG: Model is on CUDA: {next(model.parameters()).is_cuda}")
            
    if combined_args.get("mode", "train") == "train":
        checkpoint_callback = pl.pytorch.callbacks.ModelCheckpoint(
            dirpath=resolve_path(combined_args.get("weights_save_path", str(Path(output_dir) / "checkpoints"))),
            filename=combined_args["weights_name"],
            monitor="valid_medae",
            mode="min",
            save_top_k=combined_args.get("save_top_k", 1),
        )
        
        lr_logger = pl.pytorch.callbacks.LearningRateMonitor()
        
        if combined_args.get("wandb", False):
            wandb_save_path = str(Path(output_dir) / "wandb" / combined_args["version"])
            Path(wandb_save_path).mkdir(parents=True, exist_ok=True)
            wandb_logger = WandbLogger(project=combined_args.get("project", "methylgpt_finetune"),
                                    name=combined_args["version"],
                                    save_dir=wandb_save_path,
                                    config=combined_args # Log all combined args to wandb
                                    )
        else:
            wandb_logger = None
            
        device_str = combined_args.get("gpus", "cuda:1")
        
        # Extract device number from "cuda:X" format for PyTorch Lightning
        if device_str.startswith("cuda:"):
            device_num = int(device_str.split(":")[1])
        else:
            device_num = 1  # Default to GPU 1
        
        print(f"Debug: Using resolved output_dir: {output_dir}")
        print(f"Debug: Using device: {device_str} -> device_num: {device_num} for pl.Trainer")
        
        # Debug: Check device placement before training
        print(f"Debug: torch.cuda.current_device(): {torch.cuda.current_device()}")
        if torch.cuda.is_available():
            test_tensor = torch.tensor([1.0]).cuda()
            print(f"Debug: Default CUDA tensor device: {test_tensor.device}")
            del test_tensor
            torch.cuda.empty_cache()

        # Memory optimization settings
        torch.backends.cuda.matmul.allow_tf32 = True  # Enable TF32 for faster training
        torch.backends.cudnn.allow_tf32 = True
        
        # Set memory efficient attention backend if available
        try:
            torch.backends.cuda.enable_flash_sdp(True)
            print("DEBUG: Flash Attention (SDPA) enabled")
        except:
            print("DEBUG: Flash Attention (SDPA) not available, using default")
        
        trainer = pl.Trainer(
            default_root_dir=str(output_dir),
            logger=wandb_logger,
            devices=[device_num],
            accelerator="gpu",
            callbacks=[lr_logger, checkpoint_callback],
            gradient_clip_val=combined_args.get("gradient_clip_val", 1.0),
            max_epochs=combined_args.get("max_epochs", 50),
            strategy="auto",  # Let PyTorch Lightning choose the best strategy for single GPU
            log_every_n_steps=combined_args.get("log_every_n_steps", 10),  
            precision=combined_args.get("precision", "16-mixed"),  # Use mixed precision for memory savings
            enable_checkpointing=True,
            enable_progress_bar=True,
            enable_model_summary=True,
        )
        
        # DEBUG: Check model and trainer device setup before training
        print(f"DEBUG: Model device before training: {next(model.parameters()).device}")
        print(f"DEBUG: Trainer devices: {trainer.device_ids}")
        print(f"DEBUG: Current CUDA device: {torch.cuda.current_device()}")
        
        trainer.fit(model, train_loader, [valid_loader, test_loader])
            
    elif combined_args.get("mode") == "valid":
        gpus_input_value_valid = combined_args.get("gpus", "0")
        processed_gpus_for_valid_trainer = []
        if isinstance(gpus_input_value_valid, str):
            if "," in gpus_input_value_valid:
                try: processed_gpus_for_valid_trainer = [int(g.strip()) for g in gpus_input_value_valid.split(',')]
                except ValueError: processed_gpus_for_valid_trainer = [0]
            else:
                try: processed_gpus_for_valid_trainer = [int(gpus_input_value_valid.strip())]
                except ValueError: processed_gpus_for_valid_trainer = [0]
        elif isinstance(gpus_input_value_valid, int): processed_gpus_for_valid_trainer = [gpus_input_value_valid]
        elif isinstance(gpus_input_value_valid, list) and all(isinstance(item, int) for item in gpus_input_value_valid):
            processed_gpus_for_valid_trainer = gpus_input_value_valid
        else: processed_gpus_for_valid_trainer = [0]
        if not processed_gpus_for_valid_trainer: processed_gpus_for_valid_trainer = [0]

        print(f"Debug: Using processed_gpus_for_valid_trainer: {processed_gpus_for_valid_trainer} for pl.Trainer (validation)")
        trainer = pl.Trainer(
            default_root_dir=str(output_dir),
            devices=processed_gpus_for_valid_trainer,
            accelerator="gpu",
            precision=combined_args.get("precision", "bf16-true"),
        )
        # Resolve valid_ckpt_path if provided
        valid_ckpt_path_str = combined_args.get("valid_ckpt_path")
        valid_ckpt_path_resolved = resolve_path(valid_ckpt_path_str) if valid_ckpt_path_str and valid_ckpt_path_str.lower() != "none" else None
        if valid_ckpt_path_resolved and not Path(valid_ckpt_path_resolved).exists():
            print(f"Warning: Validation checkpoint path {valid_ckpt_path_resolved} (from '{valid_ckpt_path_str}') not found. Model will be validated without loading specific weights unless already loaded.")
            valid_ckpt_path_resolved = None

        trainer.validate(model, valid_loader, ckpt_path=valid_ckpt_path_resolved)
        trainer.test(model, test_loader, ckpt_path=valid_ckpt_path_resolved) # Also test with the same checkpoint

if __name__ == "__main__":
    print(f"Raw sys.argv: {sys.argv}")
    parser = argparse.ArgumentParser(description='Fine-tuning for age prediction task')
    parser.add_argument('--config_file', type=str, required=True, help='Path to the YAML configuration file')
    parser.add_argument('--gpus', type=str, help='GPU IDs to use (e.g., "0" or "0,1"). Overrides YAML.')
    parser.add_argument('--data_dir', type=str, help='Path to the data directory. Overrides YAML.')
    parser.add_argument('--output_dir', type=str, help='Path to the output directory. Overrides YAML.')
    parser.add_argument('--probe_id_file', type=str, help='Path to the probe ID file. Overrides YAML.')
    parser.add_argument('--mask_ratio', type=float, help='Mask ratio (e.g., 0.15 for 15%). Overrides YAML.')
    parser.add_argument('--mask_seed', type=int, help='Seed for masking. Overrides YAML.')

    cli_args = parser.parse_args()

    resolved_config_file = resolve_path(cli_args.config_file)
    if not resolved_config_file or not Path(resolved_config_file).exists():
        raise FileNotFoundError(f"Main configuration file not found: {resolved_config_file} (from CLI arg '{cli_args.config_file}')")

    with open(resolved_config_file, 'r') as f:
        config_from_yaml = yaml.safe_load(f)
    
    # Call train function with resolved CLI args and the loaded YAML config dictionary
    try:
        train(cli_args, config_from_yaml) # Pass the whole YAML config
    except Exception as e:
        print(f"Finetuning script failed: {e}")
        import traceback
        traceback.print_exc() # Print full traceback for debugging
        sys.exit(1)

