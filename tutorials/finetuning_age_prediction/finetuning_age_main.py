import os
import sys
from pathlib import Path
import methylgpt.modules.scGPT.scgpt as scgpt
current_directory = Path(__file__).parent.absolute()
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


def train (args):

    # Define model args
    with open("/home/A.Y/project/MethylGPT_clean/pretrained_models/args.json", 'r') as file:
        pretrain_args = json.load(file)
    # Define training args
    with open("tutorials_age_prediction/train_methyGPT.yml", 'r') as add_file:
        add_args = yaml.safe_load(add_file)
    
    model_args = {**pretrain_args, **add_args}
    model_args["version"]= f'Finetune-methylGPT-AltumAgeMLMPrediction-mask{model_args["mask_ratio"]}-dataset-{model_args["dataset"]}-basedon-Nov29-12-01'
    model_args["weights_name"] = model_args["version"] + '_{epoch:02d}-{step:02d}-{valid_medae:.4f}-{valid_mae:.4f}-{valid_s_r:.4f}-{test_medae:.4f}-{test_mae:.4f}-{test_s_r:.4f}'
    
    model_args["mask_ratio"] = args.mask_ratio*0.01
    model_args["mask_seed"] = args.mask_seed
    model_args["dropout"] = 0
    
    
    # Prepare data
    methyGPT_vocab = CollatableVocab(model_args)
    
    train_file = model_args["train_file"]
    valid_flie = model_args["valid_file"]
    test_file = model_args["test_file"]
    train_df = pd.read_parquet(train_file)
    valid_df = pd.read_parquet(valid_flie)
    test_df = pd.read_parquet(test_file)


    scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
    scaler.fit(train_df["age"].to_numpy().reshape(-1, 1))
    train_dataset = Age_Dataset(methyGPT_vocab, train_df, scaler)
    valid_dataset = Age_Dataset(methyGPT_vocab, valid_df, scaler)
    test_dataset = Age_Dataset(methyGPT_vocab, test_df, scaler)
        
    train_loader: torch.utils.data.DataLoader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=model_args["train_batch_size"],
        collate_fn=train_dataset.collater,
        shuffle=True,
        drop_last=True,
        num_workers=4,
    )
    
    valid_loader: torch.utils.data.DataLoader = torch.utils.data.DataLoader(
        valid_dataset,
        collate_fn=valid_dataset.collater,
        batch_size=model_args["valid_batch_size"],
        num_workers=4,
    )    
    
    test_loader: torch.utils.data.DataLoader = torch.utils.data.DataLoader(
        test_dataset,
        collate_fn=test_dataset.collater,
        batch_size=model_args["valid_batch_size"],
        num_workers=4,
    )
    
        
    # Init model
    model = methyGPT_Age_Model(
                model_args=model_args,
                vocab=methyGPT_vocab,
                scaler=scaler,
                )
            
    
    if model_args["mode"] == "train":

        checkpoint_callback = pl.pytorch.callbacks.ModelCheckpoint(
            dirpath=model_args["weights_save_path"],
            filename=model_args["weights_name"],
            monitor="valid_medae",
            mode="min",
            save_top_k=1,
        )
        
        lr_logger = pl.pytorch.callbacks.LearningRateMonitor()
        
        if model_args["wandb"]:
            wandb_save_path = os.path.join(str(current_directory) + "/wandb",  model_args["version"])
            os.makedirs(wandb_save_path, exist_ok=True)
            wandb_logger = WandbLogger(project=model_args["project"],
                                    name=model_args["version"],
                                    save_dir=wandb_save_path,
                                    )
        else:
            wandb_logger = None
            
        # train model
        trainer = pl.Trainer(
            default_root_dir=current_directory,
            logger=wandb_logger,
            devices=model_args["gpus"],
            accelerator="gpu",
            callbacks=[lr_logger, checkpoint_callback],
            gradient_clip_val=model_args["gradient_clip_val"],
            max_epochs=model_args["max_epochs"],
            strategy="ddp_find_unused_parameters_true", 
            log_every_n_steps=model_args["log_every_n_steps"],  
            precision="bf16-true",
        )

        trainer.fit(model, train_loader, [valid_loader, test_loader])
            
    elif model_args["mode"] == "valid":
        model.load_state_dict(torch.load(model_args["valid_ckpt_path"], map_location="cpu")['state_dict'], strict=True)
        model.eval()
        # validate model
        trainer = pl.Trainer(
            default_root_dir=current_directory,
            devices=1,
            accelerator="gpu",
            strategy="ddp_find_unused_parameters_true", 
            precision="bf16-true",
        )
        
        trainer.validate(model, [valid_loader, test_loader])
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--mask_ratio", type=float, default=0)
    parser.add_argument("--mask_seed", type=int, default=42)
    args = parser.parse_args()
    train(args)

