import sys
from pathlib import Path
sys.version

sys.path.append(str(Path(__file__).resolve().parent.parent / "modules" / "scGPT"))
current_directory = Path(__file__).parent.absolute()

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy.stats import spearmanr, pearsonr
from scgpt.model.model import AdversarialDiscriminator, TransformerModel
from scgpt.tokenizer import tokenize_and_pad_batch, random_mask_value
from scgpt.loss import masked_mse_loss
import torch.nn.functional as F
from torch import nn, optim
from sklearn import preprocessing
import os
import pandas as pd
from tqdm import tqdm
import json
import numpy as np
import torch
from torchtext.vocab import Vocab
from torchtext._torchtext import Vocab as VocabPybind
import lightning as pl
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch import seed_everything
from fintuning_age_metrics import regression_metric, EvalMultiLabel, disease_metric, disease_age_metric, disease_new_metric, disease_category_metric
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.linear_model import ElasticNet
import math
from einops import rearrange
import pickle


class FintuneAgeHead_Altumage_EMB(nn.Module):
    def __init__(
        self,
        emb_dim,
        input_dim,
        inner_dim,
        out_dim
    ):
        super().__init__()
        self.input_size = input_dim
        self.hidden_size = inner_dim
        self.reduction = nn.Linear(emb_dim, 1)
        self.layers = nn.Sequential(
            nn.BatchNorm1d(self.input_size),
            nn.Dropout(p=0.1),
            nn.Linear(self.input_size, self.hidden_size),
            nn.SELU(),
            nn.BatchNorm1d(self.hidden_size),
            nn.Dropout(p=0.1),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.SELU(),
            nn.BatchNorm1d(self.hidden_size),
            nn.Dropout(p=0.1),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.SELU(),
            nn.BatchNorm1d(self.hidden_size),
            nn.Dropout(p=0.1),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.SELU(),
            nn.BatchNorm1d(self.hidden_size),
            nn.Dropout(p=0.1),
            nn.Linear(self.hidden_size, out_dim),
        )
        
        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.reduction(x)
        x = x.squeeze(-1)
        return self.layers(x)

class FintuneAgeHead_Altumage(nn.Module):
    def __init__(
        self,
        input_dim,
        inner_dim,
        out_dim
    ):
        super().__init__()
        self.input_size = input_dim
        self.hidden_size = inner_dim
        
        self.layers = nn.Sequential(
            nn.BatchNorm1d(self.input_size),
            nn.Dropout(p=0.1),
            nn.Linear(self.input_size, self.hidden_size),
            nn.SELU(),
            nn.BatchNorm1d(self.hidden_size),
            nn.Dropout(p=0.1),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.SELU(),
            nn.BatchNorm1d(self.hidden_size),
            nn.Dropout(p=0.1),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.SELU(),
            nn.BatchNorm1d(self.hidden_size),
            nn.Dropout(p=0.1),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.SELU(),
            nn.BatchNorm1d(self.hidden_size),
            nn.Dropout(p=0.1),
            nn.Linear(self.hidden_size, out_dim),
        )
        
        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
    
        return self.layers(x)
    
    
class FintuneDeadDay_Head(nn.Module):
    def __init__(
        self,
        input_dim,
        inner_dim,
        out_dim
    ):
        super().__init__()
        self.input_size = input_dim
        self.hidden_size = inner_dim
        
        self.layers = nn.Sequential(
            nn.BatchNorm1d(self.input_size),
  
            nn.Linear(self.input_size, self.hidden_size),
            nn.Sigmoid(),
            nn.BatchNorm1d(self.hidden_size),

            nn.Linear(self.hidden_size, self.hidden_size),
            nn.Sigmoid(),
            nn.BatchNorm1d(self.hidden_size),

            nn.Linear(self.hidden_size, self.hidden_size),
            nn.Sigmoid(),
            nn.BatchNorm1d(self.hidden_size),
 
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.Sigmoid(),
            nn.BatchNorm1d(self.hidden_size),
 
            nn.Linear(self.hidden_size, out_dim),
            nn.Sigmoid()
        )
        
        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.layers(x)
    
    
    
class FintuneAgeHead(nn.Module):
    """Head for Age Prediction tasks."""
    def __init__(
        self,
        input_dim,
        inner_dim,
        out_dim
    ):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, inner_dim)
        self.bn1 = nn.BatchNorm1d(inner_dim)  
        self.dropout1 = nn.Dropout(0.1)
        self.fc2 = nn.Linear(inner_dim, inner_dim)
        self.bn2 = nn.BatchNorm1d(inner_dim)  
        self.dropout2 = nn.Dropout(0.1)
        self.fc3 = nn.Linear(inner_dim, input_dim)
        self.bn3 = nn.BatchNorm1d(input_dim)  
        self.dropout3 = nn.Dropout(0.1)
        self.pred = nn.Linear(input_dim, out_dim)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)  
        x = F.relu(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.bn2(x)  
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        x = self.bn3(x)  
        x = F.relu(x)
        x = self.dropout3(x)
        x = self.pred(x)
        x = self.sigmoid(x)
        return x

class FintuneAgeHead_Downsample(nn.Module):
    """Head for Age Prediction tasks."""
    def __init__(
        self,
        input_dim,
        # inner_dim,
        out_dim
    ):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, input_dim//2)
        self.bn1 = nn.BatchNorm1d(input_dim//2)  
        self.dropout1 = nn.Dropout(0.1)
        
        self.fc2 = nn.Linear(input_dim//2, input_dim//4)
        self.bn2 = nn.BatchNorm1d(input_dim//4)  
        self.dropout2 = nn.Dropout(0.1)
        
        self.fc3 = nn.Linear(input_dim//4, input_dim//2)
        self.bn3 = nn.BatchNorm1d(input_dim//2)  
        self.dropout3 = nn.Dropout(0.1)
        
        self.fc4 = nn.Linear(input_dim//2, input_dim)
        self.bn4 = nn.BatchNorm1d(input_dim)  
        self.dropout4 = nn.Dropout(0.1)
        
        self.pred = nn.Linear(input_dim, out_dim)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)  
        x = F.relu(x)
        x = self.dropout1(x)
        
        x = self.fc2(x)
        x = self.bn2(x)  
        x = F.relu(x)
        x = self.dropout2(x)
        
        x = self.fc3(x)
        x = self.bn3(x)  
        x = F.relu(x)
        x = self.dropout3(x)
        
        x = self.fc4(x)
        x = self.bn4(x)  
        x = F.relu(x)
        x = self.dropout4(x)
        
        x = self.pred(x)
        return x
    
class FintuneAgeHead_Upsample(nn.Module):
    """Head for Age Prediction tasks."""
    def __init__(
        self,
        input_dim,
        inner_dim,
        out_dim
    ):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, input_dim*2)
        self.bn1 = nn.BatchNorm1d(input_dim*2)  
        self.dropout1 = nn.Dropout(0.1)
        
        self.fc2 = nn.Linear(input_dim*2, input_dim*4)
        self.bn2 = nn.BatchNorm1d(input_dim*4)  
        self.dropout2 = nn.Dropout(0.1)
        
        self.fc3 = nn.Linear(input_dim*4, input_dim*2)
        self.bn3 = nn.BatchNorm1d(input_dim*2)  
        self.dropout3 = nn.Dropout(0.1)
        
        self.fc4 = nn.Linear(input_dim*2, input_dim)
        self.bn4 = nn.BatchNorm1d(input_dim)  
        self.dropout4 = nn.Dropout(0.1)
        
        self.pred = nn.Linear(input_dim, out_dim)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)  
        x = F.relu(x)
        x = self.dropout1(x)
        
        x = self.fc2(x)
        x = self.bn2(x)  
        x = F.relu(x)
        x = self.dropout2(x)
        
        x = self.fc3(x)
        x = self.bn3(x)  
        x = F.relu(x)
        x = self.dropout3(x)
        
        x = self.fc4(x)
        x = self.bn4(x)  
        x = F.relu(x)
        x = self.dropout4(x)
        
        x = self.pred(x)    
        x = self.sigmoid(x)
        return x
    
class FintuneAgeHead_Simple(nn.Module):
    """Head for Age Prediction tasks."""
    def __init__(
        self,
        input_dim,
        inner_dim,
        out_dim
    ):
        super().__init__()
        self.pred = nn.Linear(input_dim, out_dim)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        
        x = self.pred(x)
        x = self.sigmoid(x)
        return x
    
class MLP(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels
    ):
        super().__init__()
        self.fc = nn.Linear(in_channels, out_channels)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        # self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        x = self.fc(x)
        x = self.bn(x)
        x = self.relu(x)
        # x = self.dropout(x)
        return x  
        
class FintuneAgeHead_Unet(nn.Module):
    """Head for Age Prediction tasks."""
    def __init__(
        self,
        input_dim,
        inner_dim,
        out_dim
    ):
        super().__init__()
        self.down1 = MLP(input_dim, input_dim*2)
        self.down2 = MLP(input_dim*2, input_dim*4)
        self.down3 = MLP(input_dim*4, input_dim*8)
        self.down4 = MLP(input_dim*8, input_dim*8)
        
        self.up1 = MLP(input_dim*8, input_dim*8)
        self.up2 = MLP(input_dim*8, input_dim*4)
        self.up3 = MLP(input_dim*4, input_dim*2)
        self.up4 = MLP(input_dim*2, input_dim)
        
        self.pred = nn.Linear(input_dim, out_dim)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        
        x = self.up1(x4)
        _x = torch.mean(torch.stack([x, x3], dim=1), dim=1)
    
        x = self.up2(_x)
        _x =  torch.mean(torch.stack([x, x2], dim=1), dim=1)
        
        x = self.up3(_x)
        _x =  torch.mean(torch.stack([x, x1], dim=1), dim=1)
        
        x = self.up4(_x)
        x = self.pred(x)
        x = self.sigmoid(x)
        return x
    
class FintuneAgeHead_Unet_Pure(nn.Module):
    """Head for Age Prediction tasks."""
    def __init__(
        self,
        input_dim,
        inner_dim,
        out_dim
    ):
        super().__init__()
        self.down1 = nn.Linear(input_dim, input_dim*2)
        self.down2 = nn.Linear(input_dim*2, input_dim*4)
        self.down3 = nn.Linear(input_dim*4, input_dim*8)
        self.down4 = nn.Linear(input_dim*8, input_dim*8)
        
        self.up1 = nn.Linear(input_dim*8, input_dim*8)
        self.up2 = nn.Linear(input_dim*8, input_dim*4)
        self.up3 = nn.Linear(input_dim*4, input_dim*2)
        self.up4 = nn.Linear(input_dim*2, input_dim)
        
        self.pred = nn.Linear(input_dim, out_dim)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        
        x = self.up1(x4)
        _x = torch.mean(torch.stack([x, x3], dim=1), dim=1)
    
        x = self.up2(_x)
        _x =  torch.mean(torch.stack([x, x2], dim=1), dim=1)
        
        x = self.up3(_x)
        _x =  torch.mean(torch.stack([x, x1], dim=1), dim=1)
        
        x = self.up4(_x)
        x = self.pred(x)
        x = self.sigmoid(x)
        return x
    
def conv1d_3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv1d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

def conv1d_1x1(in_planes, out_planes, stride=1, padding=1):
    """1x1 convolution"""
    return nn.Conv1d(in_planes, out_planes, kernel_size=1, stride=stride, padding=padding, bias=False)
    
class ResBlock1D(nn.Module):
    def __init__(
            self,
            in_planes,
            out_planes,
            stride=1,
    ):
        super(ResBlock1D, self).__init__()
        self.bn1 = nn.BatchNorm1d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = conv1d_3x3(in_planes, out_planes, stride)
        self.bn2 = nn.BatchNorm1d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = conv1d_3x3(out_planes, out_planes)

        if stride > 1 or out_planes != in_planes:
            self.downsample = nn.Sequential(
                conv1d_1x1(in_planes, out_planes, stride=stride, padding=0),
                nn.BatchNorm1d(out_planes),
            )
        else:
            self.downsample = None

        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.bn1(x)
        out = self.relu1(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.conv2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return out
    
    
class ResBlock1D_Disease(nn.Module):
    def __init__(
            self,
            in_planes,
            out_planes,
            stride=1,
    ):
        super(ResBlock1D_Disease, self).__init__()
        self.bn1 = nn.BatchNorm1d(in_planes)
        self.relu1 = nn.ELU(inplace=True)
        self.conv1 = conv1d_3x3(in_planes, out_planes, stride)
        self.bn2 = nn.BatchNorm1d(out_planes)
        self.relu2 = nn.ELU(inplace=True)
        self.conv2 = conv1d_3x3(out_planes, out_planes)

        if stride > 1 or out_planes != in_planes:
            self.downsample = nn.Sequential(
                conv1d_1x1(in_planes, out_planes, stride=stride, padding=0),
                nn.BatchNorm1d(out_planes),
            )
        else:
            self.downsample = None

        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.bn1(x)
        out = self.relu1(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.conv2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return out


class ResNet1D(nn.Module):
    def __init__(
            self,
            in_planes,
            main_planes,
            out_planes,
            dropout=0.2,
    ):
        super(ResNet1D, self).__init__()
        self.net = nn.Sequential(
            conv1d_3x3(in_planes, main_planes),
            ResBlock1D(main_planes * 1, main_planes * 1, stride=2),
            ResBlock1D(main_planes * 1, main_planes * 1, stride=1),
            ResBlock1D(main_planes * 1, main_planes * 1, stride=2),
            ResBlock1D(main_planes * 1, main_planes * 1, stride=1),
            ResBlock1D(main_planes * 1, main_planes * 1, stride=2),
            ResBlock1D(main_planes * 1, main_planes * 1, stride=1),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(main_planes * 1, out_planes),
        )

        for m in self.modules():
            classname = m.__class__.__name__
            if classname.find('Linear') != -1:
                nn.init.normal_(m.weight, std=0.001)
                if isinstance(m.bias, nn.Parameter):
                    nn.init.constant_(m.bias, 0.0)
            elif classname.find('Conv') != -1:
                nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif classname.find('BatchNorm1d') != -1:
                if m.affine:
                    nn.init.constant_(m.weight, 1.0)
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        x = self.net(x)
        x = torch.sigmoid(x)
        return x
    
class FintuneAgeModel(pl.LightningModule):
    def __init__(self, model_args, vocab, scaler):
        super().__init__()
        self.vocab = vocab
        self.model_args= model_args
        self.scaler = scaler
        self.pretrained_model = self.from_pretrained(
            model_args,
            vocab,
        )
        self.age_head = FintuneAgeHead_Unet(
            input_dim=64,
            inner_dim=32,
            out_dim=1
        )
        self.validation_step_outputs = []
        # self.pretrained_model.eval()
        
    def forward(self, gene_ids, values, sex):
        # with torch.no_grad():
        cell_embs = self.get_cell_embeddings(self.pretrained_model, gene_ids, values)

        pred_age = self.age_head(cell_embs)
        
        return pred_age
    
    def training_step(self, batch, batch_idx):
        gene_id, value, sex, ages_label, ages_label_norm = batch
        pred_age_norm = self(gene_id, value, sex)
        pred_age_norm= pred_age_norm.squeeze()
        
        mse_loss_norm = nn.MSELoss()(pred_age_norm, ages_label_norm.squeeze())
        mse_loss_norm = mse_loss_norm.mean()
        mae_loss_norm = nn.L1Loss()(pred_age_norm, ages_label_norm.squeeze())
        mae_loss_norm = mae_loss_norm.mean()
        
        pred_age = torch.tensor(self.scaler.inverse_transform(pred_age_norm.detach().to(torch.float).cpu().numpy().reshape(-1, 1))).to(device=self.device)
        
        mse_loss = nn.MSELoss()(pred_age.squeeze(), ages_label)
        mse_loss = mse_loss.mean()
        mae_loss= nn.L1Loss()(pred_age.squeeze(), ages_label)
        mae_loss = mae_loss.mean()
        
        
        self.log("train_mse_loss_norm", mse_loss_norm, prog_bar=True, sync_dist=True, on_epoch=True)
        self.log("train_mae_loss_norm", mae_loss_norm, prog_bar=True, sync_dist=True, on_epoch=True)
        
        self.log("train_mse_loss", mse_loss, prog_bar=True, sync_dist=True, on_epoch=True)
        self.log("train_mae_loss", mae_loss, prog_bar=True, sync_dist=True, on_epoch=True)
        
        return mse_loss_norm
        
    def validation_step(self, batch, batch_idx):
        result={}
        gene_id, value, sex, ages_label, ages_label_norm = batch
        pred_age_norm = self(gene_id, value, sex)
        pred_age_norm= pred_age_norm.squeeze()
        
        mse_loss_norm = nn.MSELoss()(pred_age_norm, ages_label_norm.squeeze())
        mse_loss_norm = mse_loss_norm.mean()
        mae_loss_norm = nn.L1Loss()(pred_age_norm, ages_label_norm.squeeze())
        mae_loss_norm = mae_loss_norm.mean()
        
        pred_age = torch.tensor(self.scaler.inverse_transform(pred_age_norm.detach().to(torch.float).cpu().numpy().reshape(-1, 1))).to(device=self.device).squeeze()
        
        mse_loss = nn.MSELoss()(pred_age, ages_label)
        mse_loss = mse_loss.mean()
        mae_loss= nn.L1Loss()(pred_age, ages_label)
        mae_loss = mae_loss.mean()
        
        
        self.log("valid_mse_loss_norm", mse_loss_norm, prog_bar=True, sync_dist=True, on_epoch=True)
        self.log("valid_mae_loss_norm", mae_loss_norm, prog_bar=True, sync_dist=True, on_epoch=True)
        
        self.log("valid_mse_loss", mse_loss, prog_bar=True, sync_dist=True, on_epoch=True)
        self.log("valid_mae_loss", mae_loss, prog_bar=True, sync_dist=True, on_epoch=True)
        
        
        result = {
            'pred_age': pred_age.detach().cpu(),
            'label': ages_label.detach().cpu(),
        }
        
        self.validation_step_outputs.append(result)
        
        return result
    
    def on_validation_epoch_end(self):  
        metrics = regression_metric(self.validation_step_outputs)
        for key, value in metrics.items():
            key = f"valid_{key}"
            self.log(key, value, prog_bar=True, on_epoch=True, sync_dist=True)
            
        self.validation_step_outputs.clear()

    def configure_optimizers(self):
        params = [
            {
                "params": self.pretrained_model.parameters(),
                "lr": self.model_args["pretrained_lr"],
                "weight_decay": self.model_args["weight_decay"],
            },
            {
                "params": self.age_head.parameters(),
                "lr": self.model_args["head_lr"],
            },
        ]
        optimizer = optim.Adam(
            params,
            # betas=self.model_args["adam_betas"],
        )

        return [optimizer]
        
    def get_embeddings(self, model, gene_ids, values):
        src_key_padding_mask = gene_ids.eq(self.vocab.vocab[self.vocab.pad_token])
        transformer_output = model._encode(
            gene_ids,
            values,
            src_key_padding_mask=src_key_padding_mask,
            batch_labels=None,
        )
            
        return transformer_output
    
    def get_cell_embeddings(self, model, gene_ids, values):
        # with torch.no_grad():
        #     model.eval()
        src_key_padding_mask = gene_ids.eq(self.vocab.vocab[self.vocab.pad_token])
        output_dict = model(
            gene_ids,
            values,
            src_key_padding_mask=src_key_padding_mask,
            batch_labels=None,
        )
        cell_embeddings = output_dict["cell_emb"]
            
        return cell_embeddings
    
    @classmethod
    def from_pretrained(cls, model_args, vocab):
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
            ecs_threshold=model_args["ecs_thres"],
            explicit_zero_prob=False,
            use_fast_transformer=model_args["fast_transformer"],
            # use_fast_transformer=False,
            pre_norm=model_args["pre_norm"],
        )

        try:
            model.load_state_dict(torch.load(model_args["pretrained_file"], map_location="cpu"))
            print(f'Loading all model params from {model_args["pretrained_file"]}')
        except:
            # only load params that are in the model and match the size
            model_dict = model.state_dict()
            pretrained_dict = torch.load(model_args["pretrained_file"], map_location="cpu")
            pretrained_dict = {
                k: v
                for k, v in pretrained_dict.items()
                if k in model_dict and v.shape == model_dict[k].shape
            }
            for k, v in pretrained_dict.items():
                print(f"Loading params {k} with shape {v.shape}")
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)
            
        return model


class FintuneAgeModel_EMB(pl.LightningModule):
    def __init__(self, model_args, vocab, scaler):
        super().__init__()
        self.vocab = vocab
        self.model_args= model_args
        self.scaler = scaler
        self.pretrained_model = self.from_pretrained(
            model_args,
            vocab,
        )
        self.age_head = ResNet1D(
            in_planes=64,
            main_planes=32,
            out_planes=1
        )
        self.validation_step_outputs = []
        
    def forward(self, gene_ids, values, sex):

        embs = self.get_embeddings(self.pretrained_model, gene_ids, values)[:, 1:, :]
        embs = embs.permute(0, 2, 1)
        pred_age = self.age_head(embs)
        
        return pred_age
    
    def training_step(self, batch, batch_idx):
        gene_id, value, sex, ages_label, ages_label_norm = batch
        pred_age_norm = self(gene_id, value, sex)
        pred_age_norm= pred_age_norm.squeeze(-1)
        
        mse_loss_norm = nn.MSELoss()(pred_age_norm, ages_label_norm.squeeze())
        mse_loss_norm = mse_loss_norm.mean()
        mae_loss_norm = nn.L1Loss()(pred_age_norm, ages_label_norm.squeeze())
        mae_loss_norm = mae_loss_norm.mean()
        
        pred_age = torch.tensor(self.scaler.inverse_transform(pred_age_norm.detach().to(torch.float).cpu().numpy().reshape(-1, 1))).to(device=self.device)
        
        mse_loss = nn.MSELoss()(pred_age.squeeze(), ages_label)
        mse_loss = mse_loss.mean()
        mae_loss= nn.L1Loss()(pred_age.squeeze(), ages_label)
        mae_loss = mae_loss.mean()
        
        
        self.log("train_mse_loss_norm", mse_loss_norm, prog_bar=True, sync_dist=True, on_epoch=True)
        self.log("train_mae_loss_norm", mae_loss_norm, prog_bar=True, sync_dist=True, on_epoch=True)
        
        self.log("train_mse_loss", mse_loss, prog_bar=True, sync_dist=True, on_epoch=True)
        self.log("train_mae_loss", mae_loss, prog_bar=True, sync_dist=True, on_epoch=True)
        
        return mse_loss_norm
        
    def validation_step(self, batch, batch_idx):
        result={}
        gene_id, value, sex, ages_label, ages_label_norm = batch
        pred_age_norm = self(gene_id, value, sex)
        pred_age_norm= pred_age_norm.squeeze(-1)
        
        mse_loss_norm = nn.MSELoss()(pred_age_norm, ages_label_norm.squeeze())
        mse_loss_norm = mse_loss_norm.mean()
        mae_loss_norm = nn.L1Loss()(pred_age_norm, ages_label_norm.squeeze())
        mae_loss_norm = mae_loss_norm.mean()
        
        pred_age = torch.tensor(self.scaler.inverse_transform(pred_age_norm.detach().to(torch.float).cpu().numpy().reshape(-1, 1))).to(device=self.device).squeeze(-1)
        
        mse_loss = nn.MSELoss()(pred_age, ages_label)
        mse_loss = mse_loss.mean()
        mae_loss= nn.L1Loss()(pred_age, ages_label)
        mae_loss = mae_loss.mean()
        
        
        self.log("valid_mse_loss_norm", mse_loss_norm, prog_bar=True, sync_dist=True, on_epoch=True)
        self.log("valid_mae_loss_norm", mae_loss_norm, prog_bar=True, sync_dist=True, on_epoch=True)
        
        self.log("valid_mse_loss", mse_loss, prog_bar=True, sync_dist=True, on_epoch=True)
        self.log("valid_mae_loss", mae_loss, prog_bar=True, sync_dist=True, on_epoch=True)
        
        
        result = {
            'pred_age': pred_age.detach().cpu(),
            'label': ages_label.detach().cpu(),
        }
        
        self.validation_step_outputs.append(result)
        
        return result
    
    def on_validation_epoch_end(self):  
        metrics = regression_metric(self.validation_step_outputs)
        for key, value in metrics.items():
            key = f"valid_{key}"
            self.log(key, value, prog_bar=True, on_epoch=True, sync_dist=True)
            
        self.validation_step_outputs.clear()

    def configure_optimizers(self):
        params = [
            {
                "params": self.pretrained_model.parameters(),
                "lr": self.model_args["pretrained_lr"],
                "weight_decay": self.model_args["weight_decay"],
            },
            {
                "params": self.age_head.parameters(),
                "lr": self.model_args["head_lr"],
            },
        ]
        optimizer = optim.Adam(
            params,
            # betas=self.model_args["adam_betas"],
        )

        return [optimizer]
        
    
    def get_embeddings(self, model, gene_ids, values):
        src_key_padding_mask = gene_ids.eq(self.vocab.vocab[self.vocab.pad_token])
        transformer_output = model._encode(
            gene_ids,
            values,
            src_key_padding_mask=src_key_padding_mask,
            batch_labels=None,
        )
            
        return transformer_output
    
    def get_cell_embeddings(self, model, gene_ids, values):
        # with torch.no_grad():
        #     model.eval()
        src_key_padding_mask = gene_ids.eq(self.vocab.vocab[self.vocab.pad_token])
        output_dict = model(
            gene_ids,
            values,
            src_key_padding_mask=src_key_padding_mask,
            batch_labels=None,
        )
        cell_embeddings = output_dict["cell_emb"]
            
        return cell_embeddings
    
    @classmethod
    def from_pretrained(cls, model_args, vocab):
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
            ecs_threshold=model_args["ecs_thres"],
            explicit_zero_prob=False,
            use_fast_transformer=model_args["fast_transformer"],
            # use_fast_transformer=False,
            pre_norm=model_args["pre_norm"],
        )

        try:
            model.load_state_dict(torch.load(model_args["pretrained_file"], map_location="cpu"))
            print(f'Loading all model params from {model_args["pretrained_file"]}')
        except:
            # only load params that are in the model and match the size
            model_dict = model.state_dict()
            pretrained_dict = torch.load(model_args["pretrained_file"], map_location="cpu")
            pretrained_dict = {
                k: v
                for k, v in pretrained_dict.items()
                if k in model_dict and v.shape == model_dict[k].shape
            }
            for k, v in pretrained_dict.items():
                print(f"Loading params {k} with shape {v.shape}")
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)
            
        return model
    
class FintuneAltumAgeModel_EMB(pl.LightningModule):
    def __init__(self, model_args, vocab, scaler):
        super().__init__()
        self.vocab = vocab
        self.model_args= model_args
        self.scaler = scaler
        self.pretrained_model = self.from_pretrained(
            model_args,
            vocab,
        )
        self.age_head = ResNet1D(
            in_planes=64,
            main_planes=32,
            out_planes=1
        )
        self.validation_step_outputs = []
        self.test_step_outputs = []
        
    def forward(self, gene_ids, values):

        embs = self.get_embeddings(self.pretrained_model, gene_ids, values)[:, 1:, :]
        embs = embs.permute(0, 2, 1)
        pred_age = self.age_head(embs)
        
        return pred_age
    
    def training_step(self, batch, batch_idx):
        gene_id, value, ages_label, ages_label_norm = batch
        pred_age_norm = self(gene_id, value)
        pred_age_norm= pred_age_norm.squeeze()
        
        mse_loss_norm = nn.MSELoss()(pred_age_norm, ages_label_norm.squeeze())
        mse_loss_norm = mse_loss_norm.mean()
        mae_loss_norm = nn.L1Loss()(pred_age_norm, ages_label_norm.squeeze())
        mae_loss_norm = mae_loss_norm.mean()
        
        pred_age = torch.tensor(self.scaler.inverse_transform(pred_age_norm.detach().to(torch.float).cpu().numpy().reshape(-1, 1))).to(device=self.device)
        
        mse_loss = nn.MSELoss()(pred_age.squeeze(), ages_label)
        mse_loss = mse_loss.mean()
        mae_loss= nn.L1Loss()(pred_age.squeeze(), ages_label)
        mae_loss = mae_loss.mean()
        
        
        self.log("train_mse_loss_norm", mse_loss_norm, prog_bar=True, sync_dist=True, on_epoch=True)
        self.log("train_mae_loss_norm", mae_loss_norm, prog_bar=True, sync_dist=True, on_epoch=True)
        
        self.log("train_mse_loss", mse_loss, prog_bar=True, sync_dist=True, on_epoch=True)
        self.log("train_mae_loss", mae_loss, prog_bar=True, sync_dist=True, on_epoch=True)
        
        return mse_loss_norm
        
    def validation_step(self, batch, batch_idx, dataloader_idx):
        if dataloader_idx == 0: # for valid set
            gene_id, value, ages_label, ages_label_norm = batch
            pred_age_norm = self(gene_id, value)
            pred_age_norm= pred_age_norm.squeeze()
            
            mse_loss_norm = nn.MSELoss()(pred_age_norm, ages_label_norm.squeeze())
            mse_loss_norm = mse_loss_norm.mean()
            mae_loss_norm = nn.L1Loss()(pred_age_norm, ages_label_norm.squeeze())
            mae_loss_norm = mae_loss_norm.mean()
            
            pred_age = torch.tensor(self.scaler.inverse_transform(pred_age_norm.detach().to(torch.float).cpu().numpy().reshape(-1, 1))).to(device=self.device).squeeze()
            
            mse_loss = nn.MSELoss()(pred_age, ages_label)
            mse_loss = mse_loss.mean()
            mae_loss= nn.L1Loss()(pred_age, ages_label)
            mae_loss = mae_loss.mean()
            
            
            self.log("valid_mse_loss_norm", mse_loss_norm, prog_bar=True, sync_dist=True, on_epoch=True)
            self.log("valid_mae_loss_norm", mae_loss_norm, prog_bar=True, sync_dist=True, on_epoch=True)
            
            self.log("valid_mse_loss", mse_loss, prog_bar=True, sync_dist=True, on_epoch=True)
            self.log("valid_mae_loss", mae_loss, prog_bar=True, sync_dist=True, on_epoch=True)
            
            result={}
            result = {
                'pred_age': pred_age.detach().cpu(),
                'label': ages_label.detach().cpu(),
            }
            
            self.validation_step_outputs.append(result)
            
        elif dataloader_idx == 1: # for test set
            gene_id, value, ages_label, ages_label_norm = batch
            pred_age_norm = self(gene_id, value)
            pred_age_norm= pred_age_norm.squeeze()
            
            mse_loss_norm = nn.MSELoss()(pred_age_norm, ages_label_norm.squeeze())
            mse_loss_norm = mse_loss_norm.mean()
            mae_loss_norm = nn.L1Loss()(pred_age_norm, ages_label_norm.squeeze())
            mae_loss_norm = mae_loss_norm.mean()
            
            pred_age = torch.tensor(self.scaler.inverse_transform(pred_age_norm.detach().to(torch.float).cpu().numpy().reshape(-1, 1))).to(device=self.device).squeeze()
            
            mse_loss = nn.MSELoss()(pred_age, ages_label)
            mse_loss = mse_loss.mean()
            mae_loss= nn.L1Loss()(pred_age, ages_label)
            mae_loss = mae_loss.mean()
            
            
            self.log("test_mse_loss_norm", mse_loss_norm, prog_bar=True, sync_dist=True, on_epoch=True)
            self.log("test_mae_loss_norm", mae_loss_norm, prog_bar=True, sync_dist=True, on_epoch=True)
            
            self.log("test_mse_loss", mse_loss, prog_bar=True, sync_dist=True, on_epoch=True)
            self.log("test_mae_loss", mae_loss, prog_bar=True, sync_dist=True, on_epoch=True)
            
            result={}
            result = {
                'pred_age': pred_age.detach().cpu(),
                'label': ages_label.detach().cpu(),
            }
            
            self.test_step_outputs.append(result)
            
        
        return result
    
    def on_validation_epoch_end(self):  
        valid_metrics = regression_metric(self.validation_step_outputs)
        for key, value in valid_metrics.items():
            key = f"valid_{key}"
            self.log(key, value, prog_bar=True, on_epoch=True, sync_dist=True)
        
        test_metrics = regression_metric(self.test_step_outputs)
        for key, value in test_metrics.items():
            key = f"test_{key}"
            self.log(key, value, prog_bar=True, on_epoch=True, sync_dist=True)
            
        self.validation_step_outputs.clear()
        self.test_step_outputs.clear()

    def configure_optimizers(self):
        params = [
            {
                "params": self.pretrained_model.parameters(),
                "lr": float(self.model_args["pretrained_lr"]),
                "weight_decay": float(self.model_args["weight_decay"]),
            },
            {
                "params": self.age_head.parameters(),
                "lr": float(self.model_args["head_lr"]),
            },
        ]
        optimizer = optim.Adam(
            params,
            # betas=self.model_args["adam_betas"],
        )

        return [optimizer]
        
    
    def get_embeddings(self, model, gene_ids, values):
        src_key_padding_mask = gene_ids.eq(self.vocab.vocab[self.vocab.pad_token])
        transformer_output = model._encode(
            gene_ids,
            values,
            src_key_padding_mask=src_key_padding_mask,
            batch_labels=None,
        )
            
        return transformer_output
    
    def get_cell_embeddings(self, model, gene_ids, values):
        # with torch.no_grad():
        #     model.eval()
        src_key_padding_mask = gene_ids.eq(self.vocab.vocab[self.vocab.pad_token])
        output_dict = model(
            gene_ids,
            values,
            src_key_padding_mask=src_key_padding_mask,
            batch_labels=None,
        )
        cell_embeddings = output_dict["cell_emb"]
            
        return cell_embeddings
    
    @classmethod
    def from_pretrained(cls, model_args, vocab):
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
            ecs_threshold=model_args["ecs_thres"],
            explicit_zero_prob=False,
            use_fast_transformer=model_args["fast_transformer"],
            # use_fast_transformer=False,
            pre_norm=model_args["pre_norm"],
        )

        try:
            model.load_state_dict(torch.load(model_args["pretrained_file"], map_location="cpu"))
            print(f'Loading all model params from {model_args["pretrained_file"]}')
        except:
            # only load params that are in the model and match the size
            model_dict = model.state_dict()
            pretrained_dict = torch.load(model_args["pretrained_file"], map_location="cpu")
            pretrained_dict = {
                k: v
                for k, v in pretrained_dict.items()
                if k in model_dict and v.shape == model_dict[k].shape
            }
            for k, v in pretrained_dict.items():
                print(f"Loading params {k} with shape {v.shape}")
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)
            
        return model
    

class FintuneAltumAgeModel_CellEmbData(pl.LightningModule):
    def __init__(self, model_args, vocab):
        super().__init__()
        self.vocab = vocab
        self.model_args= model_args
        self.age_head = FintuneAgeHead_Altumage(
            input_dim=64,
            inner_dim=32,
            out_dim=1
        )
        self.pretrained_model = self.from_pretrained(
            model_args,
            vocab,
        )
        self.validation_step_outputs = []
        self.test_step_outputs = []
        
    def forward(self, gene_ids, values):

        cell_embs = self.get_cell_embeddings(self.pretrained_model, gene_ids, values)

        pred_age = self.age_head(cell_embs)
        
        
        return pred_age
        
    
    def training_step(self, batch, batch_idx):
        gene_id, value, ages_label = batch
        pred_age = self(gene_id, value)
        
        mse_loss = nn.MSELoss()(pred_age.squeeze(-1), ages_label)
        mse_loss = mse_loss.mean()
        mae_loss= nn.L1Loss()(pred_age.squeeze(-1), ages_label)
        mae_loss = mae_loss.mean()
        
        self.log("train_mse_loss", mse_loss, prog_bar=True, sync_dist=True, on_epoch=True)
        self.log("train_mae_loss", mae_loss, prog_bar=True, sync_dist=True, on_epoch=True)
        
        return mse_loss
        
    def validation_step(self, batch, batch_idx, dataloader_idx):
        if dataloader_idx == 0: # for valid set
            gene_id, value, ages_label = batch
            pred_age = self(gene_id, value)
            
            mse_loss = nn.MSELoss()(pred_age.squeeze(-1), ages_label)
            mse_loss = mse_loss.mean()
            mae_loss= nn.L1Loss()(pred_age.squeeze(-1), ages_label)
            mae_loss = mae_loss.mean()
            
            self.log("valid_mse_loss", mse_loss, prog_bar=True, sync_dist=True, on_epoch=True)
            self.log("valid_mae_loss", mae_loss, prog_bar=True, sync_dist=True, on_epoch=True)
            
            result={}
            result = {
                'pred_age': pred_age.detach().cpu(),
                'label': ages_label.detach().cpu(),
            }
            
            self.validation_step_outputs.append(result)
            
        elif dataloader_idx == 1: # for test set
            gene_id, value, ages_label = batch
            pred_age = self(gene_id, value)
            
            mse_loss = nn.MSELoss()(pred_age.squeeze(-1), ages_label)
            mse_loss = mse_loss.mean()
            mae_loss= nn.L1Loss()(pred_age.squeeze(-1), ages_label)
            mae_loss = mae_loss.mean()
            
            self.log("test_mse_loss", mse_loss, prog_bar=True, sync_dist=True, on_epoch=True)
            self.log("test_mae_loss", mae_loss, prog_bar=True, sync_dist=True, on_epoch=True)
            
            result={}
            result = {
                'pred_age': pred_age.detach().cpu(),
                'label': ages_label.detach().cpu(),
            }
            
            self.test_step_outputs.append(result)
            
        
        return result
    
    def on_validation_epoch_end(self):  
        valid_metrics = regression_metric(self.validation_step_outputs)
        for key, value in valid_metrics.items():
            key = f"valid_{key}"
            self.log(key, value, prog_bar=True, on_epoch=True, sync_dist=True)
        
        test_metrics = regression_metric(self.test_step_outputs)
        for key, value in test_metrics.items():
            key = f"test_{key}"
            self.log(key, value, prog_bar=True, on_epoch=True, sync_dist=True)
            
        self.validation_step_outputs.clear()
        self.test_step_outputs.clear()

    def configure_optimizers(self):
        params = [
            {
                "params": self.pretrained_model.parameters(),
                "lr": float(self.model_args["pretrained_lr"]),
            },
            {
                "params": self.age_head.parameters(),
                "lr": float(self.model_args["head_lr"]),
            },
        ]
        optimizer = optim.Adam(
            params,
            betas=(0.9, 0.999),
            eps=1e-7,
        )
        
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=30, min_lr=0.00001)
        
        scheduler_dict = {"scheduler": scheduler, "interval": "epoch", "monitor": "train_mse_loss"}
        
        return [optimizer], [scheduler_dict]

    def get_embeddings(self, model, gene_ids, values):
        with torch.no_grad():
            src_key_padding_mask = gene_ids.eq(self.vocab.vocab[self.vocab.pad_token])
            transformer_output = model._encode(
                gene_ids,
                values,
                src_key_padding_mask=src_key_padding_mask,
                batch_labels=None,
            )
            
        return transformer_output
    
    def get_cell_embeddings(self, model, gene_ids, values):
        # with torch.no_grad():
        #     model.eval()
        src_key_padding_mask = gene_ids.eq(self.vocab.vocab[self.vocab.pad_token])
        output_dict = model(
            gene_ids,
            values,
            src_key_padding_mask=src_key_padding_mask,
            batch_labels=None,
        )
        cell_embeddings = output_dict["cell_emb"]
            
        return cell_embeddings
    
    @classmethod
    def from_pretrained(cls, model_args, vocab):
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
            ecs_threshold=model_args["ecs_thres"],
            explicit_zero_prob=False,
            use_fast_transformer=model_args["fast_transformer"],
            # use_fast_transformer=False,
            pre_norm=model_args["pre_norm"],
        )

        try:
            model.load_state_dict(torch.load(model_args["pretrained_file"], map_location="cpu"))
            print(f'Loading all model params from {model_args["pretrained_file"]}')
        except:
            # only load params that are in the model and match the size
            model_dict = model.state_dict()
            pretrained_dict = torch.load(model_args["pretrained_file"], map_location="cpu")
            pretrained_dict = {
                k: v
                for k, v in pretrained_dict.items()
                if k in model_dict and v.shape == model_dict[k].shape
            }
            for k, v in pretrained_dict.items():
                print(f"Loading params {k} with shape {v.shape}")
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)
            
        return model


class FintuneAgeModel_EMB_MLM(pl.LightningModule):
    def __init__(self, model_args, vocab, scaler):
        super().__init__()
        self.vocab = vocab
        self.model_args= model_args
        self.scaler = scaler
        self.pretrained_model = self.from_pretrained(
            model_args,
            vocab,
        )
        self.age_head = ResNet1D(
            in_planes=model_args["layer_size"],
            main_planes=32,
            out_planes=1
        )
        self.validation_step_outputs = []
        self.test_step_outputs = []
        
    def forward(self, gene_ids, values):
        pred_mlm = self.mlm_predict(self.pretrained_model, gene_ids, values)
        embs = self.get_embeddings(self.pretrained_model, gene_ids, values)[:, 1:, :]
        embs = embs.permute(0, 2, 1)
        pred_age = self.age_head(embs)
        
        return pred_mlm, pred_age
    
    def training_step(self, batch, batch_idx):
        gene_id, masked_value, target_value, ages_label, ages_label_norm = batch
        pred_mlm, pred_age_norm = self(gene_id, masked_value)
        
        # age prediction
        pred_age_norm= pred_age_norm.squeeze()
        
        mse_loss_norm = nn.MSELoss()(pred_age_norm, ages_label_norm.squeeze())
        mse_loss_norm = mse_loss_norm.mean()
        mae_loss_norm = nn.L1Loss()(pred_age_norm, ages_label_norm.squeeze())
        mae_loss_norm = mae_loss_norm.mean()
        
        pred_age = torch.tensor(self.scaler.inverse_transform(pred_age_norm.detach().to(torch.float).cpu().numpy().reshape(-1, 1))).to(device=self.device)
        
        mse_loss = nn.MSELoss()(pred_age.squeeze(), ages_label)
        mse_loss = mse_loss.mean()
        mae_loss= nn.L1Loss()(pred_age.squeeze(), ages_label)
        mae_loss = mae_loss.mean()
        
        # mlm predition
        masked_positions = masked_value.eq(self.vocab.mask_value)
        mse_loss_mlm = masked_mse_loss(pred_mlm, target_value, masked_positions)
        
        loss = mse_loss_norm + self.model_args["mlm_wt"]*mse_loss_mlm
        
        
        self.log("train_mse_loss_norm", mse_loss_norm, prog_bar=True, sync_dist=True, on_epoch=True)
        self.log("train_mae_loss_norm", mae_loss_norm, prog_bar=True, sync_dist=True, on_epoch=True)
        
        self.log("train_mse_loss", mse_loss, prog_bar=True, sync_dist=True, on_epoch=True)
        self.log("train_mae_loss", mae_loss, prog_bar=True, sync_dist=True, on_epoch=True)
        
        self.log("train_mse_mlm", mse_loss_mlm, prog_bar=True, sync_dist=True, on_epoch=True)
        self.log("train_loss", loss, prog_bar=True, sync_dist=True, on_epoch=True)
        
        return loss
        
    def validation_step(self, batch, batch_idx):

        gene_id, masked_value, target_value, ages_label, ages_label_norm = batch
        pred_mlm, pred_age_norm = self(gene_id, masked_value)
        
        # age prediction
        pred_age_norm= pred_age_norm.squeeze()
        
        mse_loss_norm = nn.MSELoss()(pred_age_norm, ages_label_norm.squeeze())
        mse_loss_norm = mse_loss_norm.mean()
        mae_loss_norm = nn.L1Loss()(pred_age_norm, ages_label_norm.squeeze())
        mae_loss_norm = mae_loss_norm.mean()
        
        pred_age = torch.tensor(self.scaler.inverse_transform(pred_age_norm.detach().to(torch.float).cpu().numpy().reshape(-1, 1))).to(device=self.device)
        
        mse_loss = nn.MSELoss()(pred_age.squeeze(), ages_label)
        mse_loss = mse_loss.mean()
        mae_loss= nn.L1Loss()(pred_age.squeeze(), ages_label)
        mae_loss = mae_loss.mean()
        
        # mlm predition
        masked_positions = masked_value.eq(self.vocab.mask_value)
        mse_loss_mlm = masked_mse_loss(pred_mlm, target_value, masked_positions)
        
        loss = mse_loss_norm + self.model_args["mlm_wt"]*mse_loss_mlm
        
        
        self.log("valid_mse_loss_norm", mse_loss_norm, prog_bar=True, sync_dist=True, on_epoch=True)
        self.log("valid_mae_loss_norm", mae_loss_norm, prog_bar=True, sync_dist=True, on_epoch=True)
        
        self.log("valid_mse_loss", mse_loss, prog_bar=True, sync_dist=True, on_epoch=True)
        self.log("valid_mae_loss", mae_loss, prog_bar=True, sync_dist=True, on_epoch=True)
        
        self.log("valid_mse_mlm", mse_loss_mlm, prog_bar=True, sync_dist=True, on_epoch=True)
        self.log("valid_loss", loss, prog_bar=True, sync_dist=True, on_epoch=True)
        
        result={}
        result = {
            'pred_age': pred_age.detach().cpu(),
            'label': ages_label.detach().cpu(),
        }
        
        self.validation_step_outputs.append(result)
        
        return result
    
    def on_validation_epoch_end(self):  
        valid_metrics = regression_metric(self.validation_step_outputs)
        for key, value in valid_metrics.items():
            key = f"valid_{key}"
            self.log(key, value, prog_bar=True, on_epoch=True, sync_dist=True)
            
        self.validation_step_outputs.clear()


    def configure_optimizers(self):
        params = [
            {
                "params": self.pretrained_model.parameters(),
                "lr": float(self.model_args["pretrained_lr"]),
                "weight_decay": float(self.model_args["weight_decay"]),
            },
            {
                "params": self.age_head.parameters(),
                "lr": float(self.model_args["head_lr"]),
            },
        ]
        optimizer = optim.Adam(
            params,
            # betas=self.model_args["adam_betas"],
        )

        return [optimizer]
        
    def mlm_predict(self, model, gene_ids, values):
        src_key_padding_mask = gene_ids.eq(self.vocab.vocab[self.vocab.pad_token])
        output_dict = model(
            gene_ids,
            values,
            src_key_padding_mask=src_key_padding_mask,
            batch_labels=None,
        )
        mlm_preds = output_dict["pred_values"]#output_dict["mlm_output"]
        
        return mlm_preds
    
    def get_embeddings(self, model, gene_ids, values):
        src_key_padding_mask = gene_ids.eq(self.vocab.vocab[self.vocab.pad_token])
        transformer_output = model._encode(
            gene_ids,
            values,
            src_key_padding_mask=src_key_padding_mask,
            batch_labels=None,
        )
            
        return transformer_output
    
    def get_cell_embeddings(self, model, gene_ids, values):
        # with torch.no_grad():
        #     model.eval()
        src_key_padding_mask = gene_ids.eq(self.vocab.vocab[self.vocab.pad_token])
        output_dict = model(
            gene_ids,
            values,
            src_key_padding_mask=src_key_padding_mask,
            batch_labels=None,
        )
        cell_embeddings = output_dict["cell_emb"]
            
        return cell_embeddings
    

    def get_attention_map(self, model, gene_ids, values, h=4, layer=None):
        num_attn_layers = self.model_args["nlayers"] - 1
        # Use inplace operations where possible
        src_key_padding_mask = gene_ids.eq_(self.vocab.vocab[self.vocab.pad_token])
        
        # Process embeddings in a memory-efficient way
        with torch.no_grad():  # Disable gradient tracking if not needed
            src_embs = model.encoder(gene_ids)
            val_embs = model.value_encoder(values)
            total_embs = src_embs.add_(val_embs)  # Inplace addition
            del src_embs, val_embs  # Explicitly free memory
            
            if self.model_args["domain_spec_batchnorm"]:
                total_embs = model.bn(total_embs.permute(0, 2, 1)).permute(0, 2, 1)
            
            # Process through layers
            for layer in model.transformer_encoder.layers[:num_attn_layers]:
                total_embs = layer(total_embs, src_key_padding_mask=src_key_padding_mask)
                
            # Get QKV more efficiently
            qkv = model.transformer_encoder.layers[num_attn_layers].self_attn.Wqkv(total_embs)
            del total_embs  # Free memory
            
            # Calculate attention scores in chunks if sequence length is large
            b, s, _ = qkv.shape
            d = qkv.size(-1) // (3 * h)  # Calculate d based on input size
            
            # Reshape more memory efficiently
            qkv = qkv.view(b, s, 3, h, d)
            for i in range(5):
                print(f"d is {b, s, 3, h, d}")
            
            # Extract only Q and K, immediately delete qkv
            q = qkv[:, :, 0, :, :].contiguous()
            k = qkv[:, :, 1, :, :].contiguous()
            del qkv  # Explicitly free memory
            
            # Compute attention scores with reduced precision if possible
            q = q.permute(0, 2, 1, 3)
            k = k.permute(0, 2, 3, 1)
            
            # Normalize by sqrt(d_k) during the matrix multiplication
            attn_scores = (q @ k)
            del q, k  # Clean up
            
            # Optional: If memory is still an issue, you can process in chunks:
            """
            chunk_size = 1024  # Adjust based on your memory constraints
            attn_scores = []
            for i in range(0, s, chunk_size):
                chunk_q = q[:, :, i:i+chunk_size, :]
                chunk_scores = (chunk_q @ k) / math.sqrt(d)
                attn_scores.append(chunk_scores)
            attn_scores = torch.cat(attn_scores, dim=2)
            """
            
            return attn_scores
        
    def from_pretrained(self, model_args, vocab):
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
            ecs_threshold=model_args["ecs_thres"],
            explicit_zero_prob=False,
            use_fast_transformer=model_args["fast_transformer"],
            # use_fast_transformer=False,
            pre_norm=model_args["pre_norm"],
        )
        
        if self.model_args["pretrained_file"]!="None":
            try:
                model.load_state_dict(torch.load(model_args["pretrained_file"], map_location="cpu"))
                print(f'Loading all model params from {model_args["pretrained_file"]}')
            except:
                # only load params that are in the model and match the size
                model_dict = model.state_dict()
                pretrained_dict = torch.load(model_args["pretrained_file"], map_location="cpu")
                pretrained_dict = {
                    k: v
                    for k, v in pretrained_dict.items()
                    if k in model_dict and v.shape == model_dict[k].shape
                }
                for k, v in pretrained_dict.items():
                    print(f"Loading params {k} with shape {v.shape}")
                model_dict.update(pretrained_dict)
                model.load_state_dict(model_dict)
            
        return model
    
class FintuneAltumAgeModel_EMB_MLM(pl.LightningModule):
    def __init__(self, model_args, vocab, scaler):
        super().__init__()
        self.vocab = vocab
        self.model_args= model_args
        self.scaler = scaler
        self.pretrained_model = self.from_pretrained(
            model_args,
            vocab,
        )
        self.age_head = ResNet1D(
            in_planes=model_args["layer_size"],
            main_planes=32,
            out_planes=1
        )
        self.valid_step_outputs = []
        self.test_step_outputs = []
        self.infer_step_outputs = []
        
    def forward(self, gene_ids, values):
        # pred_mlm = self.mlm_predict(self.pretrained_model, gene_ids, values)
        embs = self.get_embeddings(self.pretrained_model, gene_ids, values)[:, 1:, :]
        embs = embs.permute(0, 2, 1)
        pred_age = self.age_head(embs)
        
        return pred_age
    
    def training_step(self, batch, batch_idx):
        gene_id, masked_value, target_value, ages_label, ages_label_norm = batch
        if self.model_args["mask_ratio"]==0:
            pred_age_norm = self(gene_id, target_value)
        else:
            pred_age_norm = self(gene_id, masked_value)
        
        # age prediction
        pred_age_norm= pred_age_norm.squeeze()
        mse_loss_norm = nn.MSELoss()(pred_age_norm, ages_label_norm.squeeze())
        mse_loss_norm = mse_loss_norm.mean()
        mae_loss_norm = nn.L1Loss()(pred_age_norm, ages_label_norm.squeeze())
        mae_loss_norm = mae_loss_norm.mean()
        
        pred_age = torch.tensor(self.scaler.inverse_transform(pred_age_norm.detach().to(torch.float).cpu().numpy().reshape(-1, 1))).to(device=self.device)
        
        mse_loss = nn.MSELoss()(pred_age.squeeze(), ages_label)
        mse_loss = mse_loss.mean()
        mae_loss= nn.L1Loss()(pred_age.squeeze(), ages_label)
        mae_loss = mae_loss.mean()
        
        loss = mse_loss_norm 
        
        
        self.log("train_mse_loss_norm", mse_loss_norm, prog_bar=True, sync_dist=True, on_epoch=True)
        self.log("train_mae_loss_norm", mae_loss_norm, prog_bar=True, sync_dist=True, on_epoch=True)
        
        self.log("train_mse_loss", mse_loss, prog_bar=True, sync_dist=True, on_epoch=True)
        self.log("train_mae_loss", mae_loss, prog_bar=True, sync_dist=True, on_epoch=True)
        
        return loss
        
    def validation_step(self, batch, batch_idx, dataloader_idx):
        if dataloader_idx == 0: # for valid set
            split = "valid"
            gene_id, masked_value, target_value, ages_label, ages_label_norm = batch
            if self.model_args["mask_ratio"]==0:
                pred_age_norm = self(gene_id, target_value)
            else:
                pred_age_norm = self(gene_id, masked_value)
            
            # age prediction
            pred_age_norm= pred_age_norm.squeeze()
            mse_loss_norm = nn.MSELoss()(pred_age_norm, ages_label_norm.squeeze())
            mse_loss_norm = mse_loss_norm.mean()
            mae_loss_norm = nn.L1Loss()(pred_age_norm, ages_label_norm.squeeze())
            mae_loss_norm = mae_loss_norm.mean()
            
            pred_age = torch.tensor(self.scaler.inverse_transform(pred_age_norm.detach().to(torch.float).cpu().numpy().reshape(-1, 1))).to(device=self.device)
            
            mse_loss = nn.MSELoss()(pred_age.squeeze(), ages_label)
            mse_loss = mse_loss.mean()
            mae_loss= nn.L1Loss()(pred_age.squeeze(), ages_label)
            mae_loss = mae_loss.mean()
            
            loss = mse_loss_norm 
            
            
            self.log(f"{split}_mse_loss_norm", mse_loss_norm, prog_bar=True, sync_dist=True, on_epoch=True)
            self.log(f"{split}_loss_norm", mae_loss_norm, prog_bar=True, sync_dist=True, on_epoch=True)
            
            self.log(f"{split}_mse_loss", mse_loss, prog_bar=True, sync_dist=True, on_epoch=True)
            self.log(f"{split}_mae_loss", mae_loss, prog_bar=True, sync_dist=True, on_epoch=True)
        
            
            result={}
            result = {
                'pred_age': pred_age.detach().cpu(),
                'label': ages_label.detach().cpu(),
            }
            
            self.valid_step_outputs.append(result)
            
        elif dataloader_idx == 1: # for test set
            split = "test"
            gene_id, masked_value, target_value, ages_label, ages_label_norm = batch
            if self.model_args["mask_ratio"]==0:
                pred_age_norm = self(gene_id, target_value)
            else:
                pred_age_norm = self(gene_id, masked_value)
            
            # age prediction
            pred_age_norm= pred_age_norm.squeeze()
            mse_loss_norm = nn.MSELoss()(pred_age_norm, ages_label_norm.squeeze())
            mse_loss_norm = mse_loss_norm.mean()
            mae_loss_norm = nn.L1Loss()(pred_age_norm, ages_label_norm.squeeze())
            mae_loss_norm = mae_loss_norm.mean()
            
            pred_age = torch.tensor(self.scaler.inverse_transform(pred_age_norm.detach().to(torch.float).cpu().numpy().reshape(-1, 1))).to(device=self.device)
            
            mse_loss = nn.MSELoss()(pred_age.squeeze(), ages_label)
            mse_loss = mse_loss.mean()
            mae_loss= nn.L1Loss()(pred_age.squeeze(), ages_label)
            mae_loss = mae_loss.mean()
            
            loss = mse_loss_norm 
            
            
            self.log(f"{split}_mse_loss_norm", mse_loss_norm, prog_bar=True, sync_dist=True, on_epoch=True)
            self.log(f"{split}_loss_norm", mae_loss_norm, prog_bar=True, sync_dist=True, on_epoch=True)
            
            self.log(f"{split}_mse_loss", mse_loss, prog_bar=True, sync_dist=True, on_epoch=True)
            self.log(f"{split}_mae_loss", mae_loss, prog_bar=True, sync_dist=True, on_epoch=True)
            
            result={}
            result = {
                'pred_age': pred_age.detach().cpu(),
                'label': ages_label.detach().cpu(),
            }
            
            self.test_step_outputs.append(result)
        
        return result
    
    def on_validation_epoch_end(self):  
        valid_metrics = regression_metric(self.valid_step_outputs)
        for key, value in valid_metrics.items():
            key = f"valid_{key}"
            self.log(key, value, prog_bar=True, on_epoch=True, sync_dist=True)
        
        test_metrics = regression_metric(self.test_step_outputs)
        for key, value in test_metrics.items():
            key = f"test_{key}"
            self.log(key, value, prog_bar=True, on_epoch=True, sync_dist=True)
            
        self.valid_step_outputs.clear()
        self.test_step_outputs.clear()
        
    def test_step(self, batch, batch_idx):
        dnaids, gene_ids, masked_values, target_values, ages_labels = batch
        pred_age_norm = self(gene_ids, target_values)
        pred_age = torch.tensor(self.scaler.inverse_transform(pred_age_norm.detach().to(torch.float).cpu().numpy().reshape(-1, 1))).to(device=self.device)
        
        result={}
        result = {
            'sample_id': dnaids,
            'pred_age': pred_age.detach().cpu(),
            'label': ages_labels.detach().cpu(),
        }
        self.infer_step_outputs.append(result)
        
        return result
    
    def on_test_epoch_end(self):  
        with open(f'{self.model_args["output_dir"]}/{self.model_args["ckpt_type"]}-prediction_age_results.pkl', 'wb') as f:
            pickle.dump(self.infer_step_outputs, f)
        print(f"Prediction age results saved to {self.model_args['output_dir']}/prediction_age_results_HumanMethylationEPIC v2.0-Sample Methylation Profile.pkl")


    def configure_optimizers(self):
        params = [
            {
                "params": self.pretrained_model.parameters(),
                "lr": float(self.model_args["pretrained_lr"]),
                "weight_decay": float(self.model_args["weight_decay"]),
            },
            {
                "params": self.age_head.parameters(),
                "lr": float(self.model_args["head_lr"]),
            },
        ]
        optimizer = optim.Adam(
            params,
            # betas=self.model_args["adam_betas"],
        )

        return [optimizer]
        
    def mlm_predict(self, model, gene_ids, values):
        src_key_padding_mask = gene_ids.eq(self.vocab.vocab[self.vocab.pad_token])
        output_dict = model(
            gene_ids,
            values,
            src_key_padding_mask=src_key_padding_mask,
            batch_labels=None,
        )
        mlm_preds = output_dict["mlm_output"]
        
        return mlm_preds
    
    def get_embeddings(self, model, gene_ids, values):
        src_key_padding_mask = gene_ids.eq(self.vocab.vocab[self.vocab.pad_token])
        transformer_output = model._encode(
            gene_ids,
            values,
            src_key_padding_mask=src_key_padding_mask,
            batch_labels=None,
        )
            
        return transformer_output
    
    def get_cell_embeddings(self, model, gene_ids, values):
        # with torch.no_grad():
        #     model.eval()
        src_key_padding_mask = gene_ids.eq(self.vocab.vocab[self.vocab.pad_token])
        output_dict = model(
            gene_ids,
            values,
            src_key_padding_mask=src_key_padding_mask,
            batch_labels=None,
        )
        cell_embeddings = output_dict["cell_emb"]
            
        return cell_embeddings
    
    def get_attention_map(self, model, gene_ids, values, h=4, layer=None):
        num_attn_layers = self.model_args["nlayers"] - 1
        # Use inplace operations where possible
        src_key_padding_mask = gene_ids.eq_(self.vocab.vocab[self.vocab.pad_token])
        
        # Process embeddings in a memory-efficient way
        with torch.no_grad():  # Disable gradient tracking if not needed
            src_embs = model.encoder(gene_ids)
            val_embs = model.value_encoder(values)
            total_embs = src_embs.add_(val_embs)  # Inplace addition
            del src_embs, val_embs  # Explicitly free memory
            
            if self.model_args["domain_spec_batchnorm"]:
                total_embs = model.bn(total_embs.permute(0, 2, 1)).permute(0, 2, 1)
            
            # Process through layers
            for layer in model.transformer_encoder.layers[:num_attn_layers]:
                total_embs = layer(total_embs, src_key_padding_mask=src_key_padding_mask)
                
            # Get QKV more efficiently
            qkv = model.transformer_encoder.layers[num_attn_layers].self_attn.Wqkv(total_embs)
            del total_embs  # Free memory
            
            # Calculate attention scores in chunks if sequence length is large
            b, s, _ = qkv.shape
            d = qkv.size(-1) // (3 * h)  # Calculate d based on input size
            
            # Reshape more memory efficiently
            qkv = qkv.view(b, s, 3, h, d)
            for i in range(5):
                print(f"d is {b, s, 3, h, d}")
            
            # Extract only Q and K, immediately delete qkv
            q = qkv[:, :, 0, :, :].contiguous()
            k = qkv[:, :, 1, :, :].contiguous()
            del qkv  # Explicitly free memory
            
            # Compute attention scores with reduced precision if possible
            q = q.permute(0, 2, 1, 3)
            k = k.permute(0, 2, 3, 1)
            
            # Normalize by sqrt(d_k) during the matrix multiplication
            attn_scores = (q @ k)
            del q, k  # Clean up
            
            # Optional: If memory is still an issue, you can process in chunks:
            """
            chunk_size = 1024  # Adjust based on your memory constraints
            attn_scores = []
            for i in range(0, s, chunk_size):
                chunk_q = q[:, :, i:i+chunk_size, :]
                chunk_scores = (chunk_q @ k) / math.sqrt(d)
                attn_scores.append(chunk_scores)
            attn_scores = torch.cat(attn_scores, dim=2)
            """
            
            return attn_scores
    
    def from_pretrained(self, model_args, vocab):
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
            ecs_threshold=model_args["ecs_thres"],
            explicit_zero_prob=False,
            use_fast_transformer=model_args["fast_transformer"],
            # use_fast_transformer=False,
            pre_norm=model_args["pre_norm"],
        )
        
        if self.model_args["pretrained_file"]!="None":
            try:
                model.load_state_dict(torch.load(model_args["pretrained_file"], map_location="cpu"))
                print(f'Loading all model params from {model_args["pretrained_file"]}')
            except:
                # only load params that are in the model and match the size
                model_dict = model.state_dict()
                pretrained_dict = torch.load(model_args["pretrained_file"], map_location="cpu")
                pretrained_dict = {
                    k: v
                    for k, v in pretrained_dict.items()
                    if k in model_dict and v.shape == model_dict[k].shape
                }
                for k, v in pretrained_dict.items():
                    print(f"Loading params {k} with shape {v.shape}")
                model_dict.update(pretrained_dict)
                model.load_state_dict(model_dict)
            
        return model

    
class FintuneAltumAgeModel_EmbData(pl.LightningModule):
    def __init__(self, model_args, vocab):
        super().__init__()
        self.vocab = vocab
        self.model_args= model_args
        self.age_head = FintuneAgeHead_Altumage_EMB(
            emb_dim=model_args["layer_size"],
            input_dim=model_args["cpg_sites"],
            inner_dim=32,
            out_dim=1
        )
        self.pretrained_model = self.from_pretrained(
            model_args,
            vocab,
        )
        self.pretrained_model.eval()
        self.validation_step_outputs = []
        self.test_step_outputs = []
        
    def forward(self, gene_ids, values):
        with torch.no_grad():
            embs = self.get_embeddings(self.pretrained_model, gene_ids, values)[:, 1:, :]
        
        pred_age = self.age_head(embs)
        
        return pred_age
        
    
    def training_step(self, batch, batch_idx):
        gene_id, value, ages_label = batch
        pred_age = self(gene_id, value)
        
        mse_loss = nn.MSELoss()(pred_age.squeeze(-1), ages_label)
        mse_loss = mse_loss.mean()
        mae_loss= nn.L1Loss()(pred_age.squeeze(-1), ages_label)
        mae_loss = mae_loss.mean()
        
        self.log("train_mse_loss", mse_loss, prog_bar=True, sync_dist=True, on_epoch=True)
        self.log("train_mae_loss", mae_loss, prog_bar=True, sync_dist=True, on_epoch=True)
        
        return mse_loss
        
    def validation_step(self, batch, batch_idx, dataloader_idx):
        if dataloader_idx == 0: # for valid set
            gene_id, value, ages_label = batch
            pred_age = self(gene_id, value)
            
            mse_loss = nn.MSELoss()(pred_age.squeeze(-1), ages_label)
            mse_loss = mse_loss.mean()
            mae_loss= nn.L1Loss()(pred_age.squeeze(-1), ages_label)
            mae_loss = mae_loss.mean()
            
            self.log("valid_mse_loss", mse_loss, prog_bar=True, sync_dist=True, on_epoch=True)
            self.log("valid_mae_loss", mae_loss, prog_bar=True, sync_dist=True, on_epoch=True)
            
            result={}
            result = {
                'pred_age': pred_age.detach().cpu(),
                'label': ages_label.detach().cpu(),
            }
            
            self.validation_step_outputs.append(result)
            
        elif dataloader_idx == 1: # for test set
            gene_id, value, ages_label = batch
            pred_age = self(gene_id, value)
            
            mse_loss = nn.MSELoss()(pred_age.squeeze(-1), ages_label)
            mse_loss = mse_loss.mean()
            mae_loss= nn.L1Loss()(pred_age.squeeze(-1), ages_label)
            mae_loss = mae_loss.mean()
            
            self.log("test_mse_loss", mse_loss, prog_bar=True, sync_dist=True, on_epoch=True)
            self.log("test_mae_loss", mae_loss, prog_bar=True, sync_dist=True, on_epoch=True)
            
            result={}
            result = {
                'pred_age': pred_age.detach().cpu(),
                'label': ages_label.detach().cpu(),
            }
            
            self.test_step_outputs.append(result)
            
        
        return result
    
    def on_validation_epoch_end(self):  
        valid_metrics = regression_metric(self.validation_step_outputs)
        for key, value in valid_metrics.items():
            key = f"valid_{key}"
            self.log(key, value, prog_bar=True, on_epoch=True, sync_dist=True)
        
        test_metrics = regression_metric(self.test_step_outputs)
        for key, value in test_metrics.items():
            key = f"test_{key}"
            self.log(key, value, prog_bar=True, on_epoch=True, sync_dist=True)
            
        self.validation_step_outputs.clear()
        self.test_step_outputs.clear()

    def configure_optimizers(self):
        params = [
            {
                "params": self.pretrained_model.parameters(),
                "lr": float(self.model_args["pretrained_lr"]),
            },
            {
                "params": self.age_head.parameters(),
                "lr": float(self.model_args["head_lr"]),
            },
        ]
        optimizer = optim.Adam(
            params,
            betas=(0.9, 0.999),
            eps=1e-7,
        )
        
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=30, min_lr=0.00001)
        
        scheduler_dict = {"scheduler": scheduler, "interval": "epoch", "monitor": "train_mse_loss"}
        
        return [optimizer], [scheduler_dict]

    def get_embeddings(self, model, gene_ids, values):
        with torch.no_grad():
            src_key_padding_mask = gene_ids.eq(self.vocab.vocab[self.vocab.pad_token])
            transformer_output = model._encode(
                gene_ids,
                values,
                src_key_padding_mask=src_key_padding_mask,
                batch_labels=None,
            )
            
        return transformer_output
    
    def get_cell_embeddings(self, model, gene_ids, values):
        # with torch.no_grad():
        #     model.eval()
        src_key_padding_mask = gene_ids.eq(self.vocab.vocab[self.vocab.pad_token])
        output_dict = model(
            gene_ids,
            values,
            src_key_padding_mask=src_key_padding_mask,
            batch_labels=None,
        )
        cell_embeddings = output_dict["cell_emb"]
            
        return cell_embeddings
    
    @classmethod
    def from_pretrained(cls, model_args, vocab):
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
            ecs_threshold=model_args["ecs_thres"],
            explicit_zero_prob=False,
            use_fast_transformer=model_args["fast_transformer"],
            # use_fast_transformer=False,
            pre_norm=model_args["pre_norm"],
        )

        try:
            model.load_state_dict(torch.load(model_args["pretrained_file"], map_location="cpu"))
            print(f'Loading all model params from {model_args["pretrained_file"]}')
        except:
            # only load params that are in the model and match the size
            model_dict = model.state_dict()
            pretrained_dict = torch.load(model_args["pretrained_file"], map_location="cpu")
            pretrained_dict = {
                k: v
                for k, v in pretrained_dict.items()
                if k in model_dict and v.shape == model_dict[k].shape
            }
            for k, v in pretrained_dict.items():
                print(f"Loading params {k} with shape {v.shape}")
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)
            
        return model
    
class FintuneAltumAgeModel_RawData(pl.LightningModule):
    def __init__(self, model_args):
        super().__init__()
        self.model_args= model_args
        self.age_head = FintuneAgeHead_Altumage(
            input_dim=model_args["cpg_sites"],
            inner_dim=32,
            out_dim=1
        )
        self.validation_step_outputs = []
        self.test_step_outputs = []
        
    def forward(self, gen_data):

        pred_age = self.age_head(gen_data)
        
        return pred_age
    
    def training_step(self, batch, batch_idx):
        gen_data, ages_label = batch
        pred_age = self(gen_data)
        
        mse_loss = nn.MSELoss()(pred_age.squeeze(-1), ages_label)
        mse_loss = mse_loss.mean()
        mae_loss= nn.L1Loss()(pred_age.squeeze(-1), ages_label)
        mae_loss = mae_loss.mean()
        
        self.log("train_mse_loss", mse_loss, prog_bar=True, sync_dist=True, on_epoch=True)
        self.log("train_mae_loss", mae_loss, prog_bar=True, sync_dist=True, on_epoch=True)
        
        return mse_loss
        
    def validation_step(self, batch, batch_idx, dataloader_idx):
        if dataloader_idx == 0: # for valid set
            gen_data, ages_label = batch
            pred_age = self(gen_data)
            mse_loss = nn.MSELoss()(pred_age.squeeze(-1), ages_label)
            mse_loss = mse_loss.mean()
            mae_loss= nn.L1Loss()(pred_age.squeeze(-1), ages_label)
            mae_loss = mae_loss.mean()
            
            self.log("valid_mse_loss", mse_loss, prog_bar=True, sync_dist=True, on_epoch=True)
            self.log("valid_mae_loss", mae_loss, prog_bar=True, sync_dist=True, on_epoch=True)
            
            result={}
            result = {
                'pred_age': pred_age.detach().cpu(),
                'label': ages_label.detach().cpu(),
            }
            
            self.validation_step_outputs.append(result)
            
        elif dataloader_idx == 1: # for test set
            gen_data, ages_label = batch
            pred_age = self(gen_data)
            
            mse_loss = nn.MSELoss()(pred_age.squeeze(-1), ages_label)
            mse_loss = mse_loss.mean()
            mae_loss= nn.L1Loss()(pred_age.squeeze(-1), ages_label)
            mae_loss = mae_loss.mean()
            
            self.log("test_mse_loss", mse_loss, prog_bar=True, sync_dist=True, on_epoch=True)
            self.log("test_mae_loss", mae_loss, prog_bar=True, sync_dist=True, on_epoch=True)
            
            result={}
            result = {
                'pred_age': pred_age.detach().cpu(),
                'label': ages_label.detach().cpu(),
            }
            
            self.test_step_outputs.append(result)
            
        
        return result
    
    def on_validation_epoch_end(self):  
        valid_metrics = regression_metric(self.validation_step_outputs)
        for key, value in valid_metrics.items():
            key = f"valid_{key}"
            self.log(key, value, prog_bar=True, on_epoch=True, sync_dist=True)
        
        test_metrics = regression_metric(self.test_step_outputs)
        for key, value in test_metrics.items():
            key = f"test_{key}"
            self.log(key, value, prog_bar=True, on_epoch=True, sync_dist=True)
            
        self.validation_step_outputs.clear()
        self.test_step_outputs.clear()

    def configure_optimizers(self):
        params = [
            {
                "params": self.age_head.parameters(),
                "lr": float(self.model_args["head_lr"]),
            },
        ]
        optimizer = optim.Adam(
            params,
            betas=(0.9, 0.999),
            eps=1e-7,
        )
        
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=30, min_lr=0.0001)   # 0.00001
        
        scheduler_dict = {"scheduler": scheduler, "interval": "epoch", "monitor": "train_mse_loss"}
        
        return [optimizer], [scheduler_dict]
    
    
class FintuneDeadDayModel_RawData(pl.LightningModule):
    def __init__(self, model_args):
        super().__init__()
        self.model_args= model_args
        self.age_head = FintuneAgeHead_Altumage(
            input_dim=model_args["cpg_sites"],
            inner_dim=32,
            out_dim=1
        )
        self.validation_step_outputs = []
        
    def forward(self, gen_data):

        pred_age = self.age_head(gen_data)
        
        return pred_age
    
    def training_step(self, batch, batch_idx):
        gen_data, ages_label = batch
        pred_age = self(gen_data)
        
        mse_loss = nn.MSELoss()(pred_age.squeeze(-1), ages_label)
        mse_loss = mse_loss.mean()
        mae_loss= nn.L1Loss()(pred_age.squeeze(-1), ages_label)
        mae_loss = mae_loss.mean()
        
        self.log("train_mse_loss", mse_loss, prog_bar=True, sync_dist=True, on_epoch=True)
        self.log("train_mae_loss", mae_loss, prog_bar=True, sync_dist=True, on_epoch=True)
        
        return mse_loss
        
    def validation_step(self, batch, batch_idx):

        gen_data, ages_label = batch
        pred_age = self(gen_data)
        mse_loss = nn.MSELoss()(pred_age.squeeze(-1), ages_label)
        mse_loss = mse_loss.mean()
        mae_loss= nn.L1Loss()(pred_age.squeeze(-1), ages_label)
        mae_loss = mae_loss.mean()
        
        self.log("valid_mse_loss", mse_loss, prog_bar=True, sync_dist=True, on_epoch=True)
        self.log("valid_mae_loss", mae_loss, prog_bar=True, sync_dist=True, on_epoch=True)
        
        result={}
        result = {
            'pred_age': pred_age.detach().cpu(),
            'label': ages_label.detach().cpu(),
        }
        
        self.validation_step_outputs.append(result)
    
            
        
        return result
    
    def on_validation_epoch_end(self):  
        valid_metrics = regression_metric(self.validation_step_outputs)
        for key, value in valid_metrics.items():
            key = f"valid_{key}"
            self.log(key, value, prog_bar=True, on_epoch=True, sync_dist=True)
            
        self.validation_step_outputs.clear()


    def configure_optimizers(self):
        params = [
            {
                "params": self.age_head.parameters(),
                "lr": float(self.model_args["head_lr"]),
            },
        ]
        optimizer = optim.Adam(
            params,
            betas=(0.9, 0.999),
            eps=1e-7,
        )
        
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=30, min_lr=0.0001)   # 0.00001
        
        scheduler_dict = {"scheduler": scheduler, "interval": "epoch", "monitor": "train_mse_loss"}
        
        return [optimizer], [scheduler_dict]
    
    
###############Disease Prediction######################

class textCNN(nn.Module):
    def __init__(self, param):
        super(textCNN, self).__init__()
        ci = 1  # input chanel size
        kernel_num = param['kernel_num'] # output chanel size
        kernel_size = param['kernel_size']
        # vocab_size = param['vocab_size']
        embed_dim = param['embed_dim']
        dropout = param['dropout']
        class_num = param['class_num']
        self.param = param
        # self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=1)
        self.conv11 = nn.Conv2d(ci, kernel_num, (kernel_size[0], embed_dim))
        self.conv12 = nn.Conv2d(ci, kernel_num, (kernel_size[1], embed_dim))
        self.conv13 = nn.Conv2d(ci, kernel_num, (kernel_size[2], embed_dim))
        self.dropout = nn.Dropout(dropout)
        self.fc_disease = nn.Linear(len(kernel_size) * kernel_num, class_num)
        self.bn1 = nn.BatchNorm1d(embed_dim)
        self.bn2 = nn.BatchNorm1d(embed_dim)
        self.bn3 = nn.BatchNorm1d(embed_dim)
        # self.fc_age = nn.Linear(len(kernel_size) * kernel_num, 1)
        self.sigmoid = nn.Sigmoid()
        self.elu = nn.ELU()

    def init_embed(self, embed_matrix):
        self.embed.weight = nn.Parameter(torch.Tensor(embed_matrix))

    
    def conv_and_pool(self, x, conv, bn):
        # x: (batch, 1, sentence_length,  )
        x = x.permute(0, 2, 1)
        x = bn(x)
        x = x.permute(0, 2, 1)
        x = x.unsqueeze(1)
        x = conv(x)
        x = self.elu(x.squeeze(3))
        # x: (batch, kernel_num, H_out, 1)
        # x = x.squeeze(3)
        # x: (batch, kernel_num, H_out)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        #  (batch, kernel_num)
        return x

    def forward(self, x):
        # x: (batch, sentence_length)
        # x = self.embed(x)
        # x: (batch, sentence_length, embed_dim)
        # TODO init embed matrix with pre-trained
        # x = x.unsqueeze(1)
        # x: (batch, 1, sentence_length, embed_dim)
        
        x1 = self.conv_and_pool(x, self.conv11, self.bn1)  # (batch, kernel_num)
        x2 = self.conv_and_pool(x, self.conv12, self.bn2)  # (batch, kernel_num)
        x3 = self.conv_and_pool(x, self.conv13, self.bn3)  # (batch, kernel_num)
        x = torch.cat((x1, x2, x3), 1)  # (batch, 3 * kernel_num)
        x = self.dropout(x)
        logit_disease = self.sigmoid(self.fc_disease(x))
        # logit_age = self.sigmoid(self.fc_age(x))
        # return logit_disease, logit_age
        return logit_disease

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm1d):
                if m.affine:
                    nn.init.constant_(m.weight, 1.0)
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
                
class ResNet1D_Disease(nn.Module):
    def __init__(
            self,
            model_args,
            in_planes,
            main_planes,
            dropout=0.2,
    ):
        super(ResNet1D_Disease, self).__init__()
        self.net = nn.Sequential(
            conv1d_3x3(in_planes, main_planes),
            ResBlock1D_Disease(main_planes * 1, main_planes * 1, stride=2),
            ResBlock1D_Disease(main_planes * 1, main_planes * 1, stride=1),
            ResBlock1D_Disease(main_planes * 1, main_planes * 1, stride=2),
            ResBlock1D_Disease(main_planes * 1, main_planes * 1, stride=1),
            ResBlock1D_Disease(main_planes * 1, main_planes * 1, stride=2),
            ResBlock1D_Disease(main_planes * 1, main_planes * 1, stride=1),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
        )
        self.fc_disease = nn.Linear(main_planes, model_args["class_num"])
        self.sigmoid = nn.Sigmoid()
        
        for m in self.modules():
            classname = m.__class__.__name__
            if classname.find('Linear') != -1:
                nn.init.normal_(m.weight, std=0.001)
                if isinstance(m.bias, nn.Parameter):
                    nn.init.constant_(m.bias, 0.0)
            elif classname.find('Conv') != -1:
                nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif classname.find('BatchNorm1d') != -1:
                if m.affine:
                    nn.init.constant_(m.weight, 1.0)
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.net(x)
        logit_disease = self.sigmoid(self.fc_disease(x))
        return logit_disease
    
class FintuneDiseaseModel_EMB(pl.LightningModule):
    def __init__(self, model_args, vocab, scaler):
        super().__init__()
        self.vocab = vocab
        self.model_args= model_args
        self.scaler = scaler
        self.pretrained_model = self.from_pretrained(
            model_args,
            vocab,
        )
        if model_args["architecture"]=="textcnn":
            self.disease_head = textCNN(model_args)
        elif model_args["architecture"]=="resnet":
            self.disease_head = ResNet1D_Disease(
                model_args,
                in_planes=model_args["embed_dim"],
                main_planes=32,
            )
        
        self.valid_step_outputs = []
        self.test_step_outputs = []
        
    def forward(self, gene_ids, values):
        embs = self.get_embeddings(self.pretrained_model, gene_ids, values)[:, 1:, :] # [batch_size, seq_len, emb_dim]
        pred_disease = self.disease_head(embs)
        
        return pred_disease
    
    def training_step(self, batch, batch_idx):

        gene_id, masked_value, target_value, ages_label, ages_label_norm, disease_label = batch
        if self.model_args["datatype"] == "disease_biclass_train_2994":
            disease_label = disease_label.reshape(-1, 1)
        pred_disease = self(gene_id, masked_value)
        

        # disease predition
        cls_loss = nn.BCELoss()(pred_disease, disease_label)
        cls_loss = cls_loss.mean()
        
        loss = cls_loss
        
        self.log("train_loss", loss, prog_bar=True, sync_dist=True, on_epoch=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):

        sample_id, gene_id, masked_value, target_value, ages_label, ages_label_norm, disease_label = batch
        if self.model_args["datatype"] == "disease_biclass_train_2994":
            disease_label = disease_label.reshape(-1, 1)
        pred_disease = self(gene_id, masked_value)
       
        # disease predition
        cls_loss = nn.BCELoss()(pred_disease, disease_label)
        cls_loss = cls_loss.mean()
        
        loss = cls_loss
        
        self.log("valid_loss", loss, prog_bar=True, sync_dist=True, on_epoch=True)
        
        result={}
        result = {
            'sample_id,': sample_id,
            'pred_disease': pred_disease.detach().cpu(),
            'label_disease': disease_label.detach().cpu(),
        }
        
        self.valid_step_outputs.append(result)
        
        return result
    
    def on_validation_epoch_end(self):  
        valid_metrics = disease_new_metric(self.valid_step_outputs)
        for key, value in valid_metrics.items():
            key = f"valid_{key}"
            self.log(key, value, prog_bar=True, on_epoch=True, sync_dist=True)
            
        self.valid_step_outputs.clear()
        
    def test_step(self, batch, batch_idx):
        data_name, sample_id, gene_id, masked_value, target_value = batch
        pred_disease = self(gene_id, target_value)
        
        result={}
        result = {
            'data_name': data_name,
            'sample_id': sample_id,
            'pred_disease': pred_disease.detach().cpu(),
        }
        self.test_step_outputs.append(result)
        
        return result
    
    def on_test_epoch_end(self):  
        
        with open(f'{self.model_args["output_dir"]}/prediction_disease_results_intervention.pkl', 'wb') as f:
            pickle.dump(self.test_step_outputs, f)
        print(f"Prediction disease results saved to {self.model_args['output_dir']}/prediction_disease_results_intervention.pkl")
    
        
    def configure_optimizers(self):
        params = [
            {
                "params": self.pretrained_model.parameters(),
                "lr": float(self.model_args["pretrained_lr"]),
                "weight_decay": float(self.model_args["weight_decay"]),
            },
            {
                "params": self.disease_head.parameters(),
                "lr": float(self.model_args["head_lr"]),
            },
        ]
        optimizer = optim.Adam(
            params,
            # betas=self.model_args["adam_betas"],
        )

        return [optimizer]
    
    def get_embeddings(self, model, gene_ids, values):
        src_key_padding_mask = gene_ids.eq(self.vocab.vocab[self.vocab.pad_token])
        transformer_output = model._encode(
            gene_ids,
            values,
            src_key_padding_mask=src_key_padding_mask,
            batch_labels=None,
        )
            
        return transformer_output
    
    def get_cell_embeddings(self, model, gene_ids, values):
        # with torch.no_grad():
        #     model.eval()
        src_key_padding_mask = gene_ids.eq(self.vocab.vocab[self.vocab.pad_token])
        output_dict = model(
            gene_ids,
            values,
            src_key_padding_mask=src_key_padding_mask,
            batch_labels=None,
        )
        cell_embeddings = output_dict["cell_emb"]
            
        return cell_embeddings
    
    def from_pretrained(self, model_args, vocab):
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
            ecs_threshold=model_args["ecs_thres"],
            explicit_zero_prob=False,
            use_fast_transformer=model_args["fast_transformer"],
            pre_norm=model_args["pre_norm"],
        )
        
        if self.model_args["pretrained_file"]!="None":
            try:
                model.load_state_dict(torch.load(model_args["pretrained_file"], map_location="cpu"))
                print(f'Loading all model params from {model_args["pretrained_file"]}')
            except:
                # only load params that are in the model and match the size
                model_dict = model.state_dict()
                pretrained_dict = torch.load(model_args["pretrained_file"], map_location="cpu")
                pretrained_dict = {
                    k: v
                    for k, v in pretrained_dict.items()
                    if k in model_dict and v.shape == model_dict[k].shape
                }
                for k, v in pretrained_dict.items():
                    print(f"Loading params {k} with shape {v.shape}")
                model_dict.update(pretrained_dict)
                model.load_state_dict(model_dict)
            
        return model
    
    
class FintuneDiseaseCategoryModel_EMB(pl.LightningModule):
    def __init__(self, model_args, vocab, scaler):
        super().__init__()
        self.vocab = vocab
        self.model_args= model_args
        self.scaler = scaler
        self.pretrained_model = self.from_pretrained(
            model_args,
            vocab,
        )
        if model_args["architecture"]=="textcnn":
            self.disease_head = textCNN(model_args)
        elif model_args["architecture"]=="resnet":
            self.disease_head = ResNet1D_Disease(
                model_args,
                in_planes=model_args["embed_dim"],
                main_planes=32,
            )
        
        self.valid_step_outputs = []
        self.test_step_outputs = []
        self.infer_step_outputs =[]
        
    def forward(self, gene_ids, values):
        embs = self.get_embeddings(self.pretrained_model, gene_ids, values)[:, 1:, :] # [batch_size, seq_len, emb_dim]
        pred_disease = self.disease_head(embs)
        
        return pred_disease
    
    def training_step(self, batch, batch_idx):

        sample_id, gene_id, masked_value, target_value, ages_label, ages_label_norm, disease_label = batch
        if self.model_args["datatype"] == "disease_biclass_train_2994":
            disease_label = disease_label.reshape(-1, 1)
        pred_disease = self(gene_id, target_value)
        
        # disease predition
        cls_loss = nn.BCELoss()(pred_disease, disease_label)
        cls_loss = cls_loss.mean()
        
        loss = cls_loss
        
        self.log("train_loss", loss, prog_bar=True, sync_dist=True, on_epoch=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx, dataloader_idx):
        if dataloader_idx == 0: # for valid set
            sample_id, gene_id, masked_value, target_value, ages_label, ages_label_norm, disease_label = batch
            if self.model_args["datatype"] == "disease_biclass_train_2994":
                disease_label = disease_label.reshape(-1, 1)
            pred_disease = self(gene_id, masked_value)
        
            # disease predition
            cls_loss = nn.BCELoss()(pred_disease, disease_label)
            cls_loss = cls_loss.mean()
            
            loss = cls_loss
            
            self.log("valid_loss", loss, prog_bar=True, sync_dist=True, on_epoch=True)
            
            result={}
            result = {
                'sample_id': sample_id,
                'pred_disease': pred_disease.detach().cpu(),
                'label_disease': disease_label.detach().cpu(),
            }
            
            self.valid_step_outputs.append(result)
            
        elif dataloader_idx == 1: # for test set
            sample_id, gene_id, masked_value, target_value, ages_label, ages_label_norm, disease_label = batch
            if self.model_args["datatype"] == "disease_biclass_train_2994":
                disease_label = disease_label.reshape(-1, 1)
            pred_disease = self(gene_id, masked_value)
        
            # disease predition
            cls_loss = nn.BCELoss()(pred_disease, disease_label)
            cls_loss = cls_loss.mean()
            
            loss = cls_loss
            
            self.log("test_loss", loss, prog_bar=True, sync_dist=True, on_epoch=True)
            
            result={}
            result = {
                'sample_id': sample_id,
                'pred_disease': pred_disease.detach().cpu(),
                'label_disease': disease_label.detach().cpu(),
            }
            
            self.test_step_outputs.append(result)
        
        return result
    
    # def on_validation_epoch_end(self):  
    #     valid_metrics = disease_category_metric(self.valid_step_outputs, self.model_args, data_type="valid")
    #     for key, value in valid_metrics.items():
    #         key = f"valid_{key}"
    #         self.log(key, value, prog_bar=True, on_epoch=True, sync_dist=True)
            
    #     self.valid_step_outputs.clear()
        
    #     test_metrics = disease_category_metric(self.test_step_outputs, self.model_args, data_type="test")
    #     for key, value in test_metrics.items():
    #         key = f"test_{key}"
    #         self.log(key, value, prog_bar=True, on_epoch=True, sync_dist=True)
            
    #     self.test_step_outputs.clear()
        
    def on_validation_epoch_end(self):  
        valid_metrics = disease_category_metric(self.valid_step_outputs, self.model_args, data_type="valid")
        
        test_metrics = disease_category_metric(self.test_step_outputs, self.model_args, data_type="test")
    
    def test_step(self, batch, batch_idx):
        data_name, sample_id, gene_id, masked_value, target_value = batch
        pred_disease = self(gene_id, target_value)
        
        result={}
        result = {
            'data_name': data_name,
            'sample_id': sample_id,
            'pred_disease': pred_disease.detach().cpu(),
        }
        self.infer_step_outputs.append(result)
        
        return result
    
    def on_test_epoch_end(self):  
        with open(f'{self.model_args["output_dir"]}/prediction_disease_methyGPT_results_intervention.pkl', 'wb') as f:
            pickle.dump(self.infer_step_outputs, f)
        print(f"Prediction disease results saved to {self.model_args['output_dir']}/prediction_disease_methyGPT_results_intervention.pkl")
    
        
    def configure_optimizers(self):
        params = [
            {
                "params": self.pretrained_model.parameters(),
                "lr": float(self.model_args["pretrained_lr"]),
                "weight_decay": float(self.model_args["weight_decay"]),
            },
            {
                "params": self.disease_head.parameters(),
                "lr": float(self.model_args["head_lr"]),
            },
        ]
        optimizer = optim.Adam(
            params,
            # betas=self.model_args["adam_betas"],
        )

        return [optimizer]
    
    def get_embeddings(self, model, gene_ids, values):
        src_key_padding_mask = gene_ids.eq(self.vocab.vocab[self.vocab.pad_token])
        transformer_output = model._encode(
            gene_ids,
            values,
            src_key_padding_mask=src_key_padding_mask,
            batch_labels=None,
        )
            
        return transformer_output
    
    def get_cell_embeddings(self, model, gene_ids, values):
        # with torch.no_grad():
        #     model.eval()
        src_key_padding_mask = gene_ids.eq(self.vocab.vocab[self.vocab.pad_token])
        output_dict = model(
            gene_ids,
            values,
            src_key_padding_mask=src_key_padding_mask,
            batch_labels=None,
        )
        cell_embeddings = output_dict["cell_emb"]
            
        return cell_embeddings
    
    def from_pretrained(self, model_args, vocab):
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
            ecs_threshold=model_args["ecs_thres"],
            explicit_zero_prob=False,
            use_fast_transformer=model_args["fast_transformer"],
            pre_norm=model_args["pre_norm"],
        )
        
        if self.model_args["pretrained_file"]!="None":
            try:
                model.load_state_dict(torch.load(model_args["pretrained_file"], map_location="cpu"))
                print(f'Loading all model params from {model_args["pretrained_file"]}')
            except:
                # only load params that are in the model and match the size
                model_dict = model.state_dict()
                pretrained_dict = torch.load(model_args["pretrained_file"], map_location="cpu")
                pretrained_dict = {
                    k: v
                    for k, v in pretrained_dict.items()
                    if k in model_dict and v.shape == model_dict[k].shape
                }
                for k, v in pretrained_dict.items():
                    print(f"Loading params {k} with shape {v.shape}")
                model_dict.update(pretrained_dict)
                model.load_state_dict(model_dict)
            
        return model
    
    
class FintuneDiseaseCategoryModel_RawData(pl.LightningModule):
    def __init__(self, model_args):
        super().__init__()
    
        self.model_args= model_args
 
        self.disease_head = FintuneAgeHead_Altumage(
            input_dim=model_args["cpg_sites"],
            inner_dim=32,
            out_dim=model_args["class_num"]
        )
        self.sigmoid = nn.Sigmoid()
        self.valid_step_outputs = []
        self.test_step_outputs = []
        
    def forward(self, gen_data):

        pred_disease = self.disease_head(gen_data)
        pred_disease = self.sigmoid(pred_disease)
        
        return pred_disease
    
    def training_step(self, batch, batch_idx):

        sample_id, gen_data, ages_label, disease_label = batch
        if self.model_args["datatype"] == "disease_biclass_train_2994":
            disease_label = disease_label.reshape(-1, 1)
        pred_disease = self(gen_data)
        
        # disease predition
        cls_loss = nn.BCELoss()(pred_disease, disease_label)
        cls_loss = cls_loss.mean()
        
        loss = cls_loss
        
        self.log("train_loss", loss, prog_bar=True, sync_dist=True, on_epoch=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx, dataloader_idx):
        if dataloader_idx == 0: # for valid set
            sample_id, gen_data, ages_label, disease_label = batch
            if self.model_args["datatype"] == "disease_biclass_train_2994":
                disease_label = disease_label.reshape(-1, 1)
            pred_disease = self(gen_data)
        
            # disease predition
            cls_loss = nn.BCELoss()(pred_disease, disease_label)
            cls_loss = cls_loss.mean()
            
            loss = cls_loss
            
            self.log("valid_loss", loss, prog_bar=True, sync_dist=True, on_epoch=True)
            
            result={}
            result = {
                'sample_id': sample_id,
                'pred_disease': pred_disease.detach().cpu(),
                'label_disease': disease_label.detach().cpu(),
            }
            
            self.valid_step_outputs.append(result)
            
        elif dataloader_idx == 1: # for test set
            sample_id, gen_data, ages_label, disease_label = batch
            if self.model_args["datatype"] == "disease_biclass_train_2994":
                disease_label = disease_label.reshape(-1, 1)
            pred_disease = self(gen_data)
        
            # disease predition
            cls_loss = nn.BCELoss()(pred_disease, disease_label)
            cls_loss = cls_loss.mean()
            
            loss = cls_loss
            
            self.log("test_loss", loss, prog_bar=True, sync_dist=True, on_epoch=True)
            
            result={}
            result = {
                'sample_id': sample_id,
                'pred_disease': pred_disease.detach().cpu(),
                'label_disease': disease_label.detach().cpu(),
            }
            
            self.test_step_outputs.append(result)
        
        return result
    
    # def on_validation_epoch_end(self):  
    #     valid_metrics = disease_category_metric(self.valid_step_outputs)
    #     for key, value in valid_metrics.items():
    #         key = f"valid_{key}"
    #         self.log(key, value, prog_bar=True, on_epoch=True, sync_dist=True)
            
    #     self.valid_step_outputs.clear()
        
    #     test_metrics = disease_category_metric(self.test_step_outputs)
    #     for key, value in test_metrics.items():
    #         key = f"test_{key}"
    #         self.log(key, value, prog_bar=True, on_epoch=True, sync_dist=True)
            
    #     self.test_step_outputs.clear()
    def on_validation_epoch_end(self):  
        valid_metrics = disease_category_metric(self.valid_step_outputs, self.model_args, data_type="valid")
        
        test_metrics = disease_category_metric(self.test_step_outputs, self.model_args, data_type="test")

    def test_step(self, batch, batch_idx):
        data_name, sample_id, gene_id, masked_value, target_value = batch
        pred_disease = self(target_value)
        
        result={}
        result = {
            'data_name': data_name,
            'sample_id': sample_id,
            'pred_disease': pred_disease.detach().cpu(),
        }
        self.infer_step_outputs.append(result)
        
        return result
    
    def on_test_epoch_end(self):  
        with open(f'{self.model_args["output_dir"]}/prediction_disease_mlp_results_intervention.pkl', 'wb') as f:
            pickle.dump(self.infer_step_outputs, f)
        print(f"Prediction disease results saved to {self.model_args['output_dir']}/prediction_disease_mlp_results_intervention.pkl")
        
    def configure_optimizers(self):
        params = [
            {
                "params": self.disease_head.parameters(),
                "lr": float(self.model_args["head_lr"]),
            },
        ]
        optimizer = optim.Adam(
            params,
            betas=(0.9, 0.999),
            eps=1e-7,
        )
        
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=30, min_lr=0.0001)   # 0.00001
        
        scheduler_dict = {"scheduler": scheduler, "interval": "epoch", "monitor": "train_loss"}
        
        return [optimizer], [scheduler_dict]
    

    

class FintuneDiseaseModel_BI(pl.LightningModule):
    def __init__(self, model_args, vocab, scaler):
        super().__init__()
        self.vocab = vocab
        self.model_args= model_args
        self.scaler = scaler
        self.pretrained_model = self.from_pretrained(
            model_args,
            vocab,
        )
        self.disease_head = nn.Sequential(
                                nn.BatchNorm1d(model_args["embed_dim"]),
                                nn.Linear(model_args["embed_dim"], model_args["class_num"]),
                                nn.Sigmoid()
                            )
        
        self.valid_step_outputs = []
        
    def forward(self, gene_ids, values):
        embs = self.get_cell_embeddings(self.pretrained_model, gene_ids, values) # [batch_size, seq_len, emb_dim]
        pred_disease = self.disease_head(embs)
        
        return pred_disease
    
    def training_step(self, batch, batch_idx):

        gene_id, masked_value, target_value, ages_label, ages_label_norm, disease_label = batch
        disease_label = disease_label.reshape(-1, 1)
        pred_disease = self(gene_id, target_value)
        

        # disease predition
        cls_loss = nn.BCELoss()(pred_disease, disease_label)
        cls_loss = cls_loss.mean()
        
        loss = cls_loss
        
        self.log("train_loss", loss, prog_bar=True, sync_dist=True, on_epoch=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):

        gene_id, masked_value, target_value, ages_label, ages_label_norm, disease_label = batch
        disease_label = disease_label.reshape(-1, 1)
        pred_disease = self(gene_id, target_value)
        
        
        # disease predition
        cls_loss = nn.BCELoss()(pred_disease, disease_label)
        cls_loss = cls_loss.mean()
        
        loss = cls_loss
        
        self.log("valid_loss", loss, prog_bar=True, sync_dist=True, on_epoch=True)
        
        result={}
        result = {
            'pred_disease': pred_disease.detach().cpu(),
            'label_disease': disease_label.detach().cpu(),
        }
        
        self.valid_step_outputs.append(result)
        
        return result
    
    def on_validation_epoch_end(self):  
        valid_metrics = disease_new_metric(self.valid_step_outputs)
        for key, value in valid_metrics.items():
            key = f"valid_{key}"
            self.log(key, value, prog_bar=True, on_epoch=True, sync_dist=True)
            
        self.valid_step_outputs.clear()
        
    def configure_optimizers(self):
        params = [
            {
                "params": self.pretrained_model.parameters(),
                "lr": float(self.model_args["pretrained_lr"]),
                "weight_decay": float(self.model_args["weight_decay"]),
            },
            {
                "params": self.disease_head.parameters(),
                "lr": float(self.model_args["head_lr"]),
            },
        ]
        optimizer = optim.Adam(
            params,
            # betas=self.model_args["adam_betas"],
        )

        return [optimizer]
    
    def get_embeddings(self, model, gene_ids, values):
        src_key_padding_mask = gene_ids.eq(self.vocab.vocab[self.vocab.pad_token])
        transformer_output = model._encode(
            gene_ids,
            values,
            src_key_padding_mask=src_key_padding_mask,
            batch_labels=None,
        )
            
        return transformer_output
    
    def get_cell_embeddings(self, model, gene_ids, values):
        # with torch.no_grad():
        #     model.eval()
        src_key_padding_mask = gene_ids.eq(self.vocab.vocab[self.vocab.pad_token])
        output_dict = model(
            gene_ids,
            values,
            src_key_padding_mask=src_key_padding_mask,
            batch_labels=None,
        )
        cell_embeddings = output_dict["cell_emb"]
            
        return cell_embeddings
    
    def from_pretrained(self, model_args, vocab):
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
            ecs_threshold=model_args["ecs_thres"],
            explicit_zero_prob=False,
            use_fast_transformer=model_args["fast_transformer"],
            pre_norm=model_args["pre_norm"],
        )
        
        if self.model_args["pretrained_file"]!="None":
            try:
                model.load_state_dict(torch.load(model_args["pretrained_file"], map_location="cpu"))
                print(f'Loading all model params from {model_args["pretrained_file"]}')
            except:
                # only load params that are in the model and match the size
                model_dict = model.state_dict()
                pretrained_dict = torch.load(model_args["pretrained_file"], map_location="cpu")
                pretrained_dict = {
                    k: v
                    for k, v in pretrained_dict.items()
                    if k in model_dict and v.shape == model_dict[k].shape
                }
                for k, v in pretrained_dict.items():
                    print(f"Loading params {k} with shape {v.shape}")
                model_dict.update(pretrained_dict)
                model.load_state_dict(model_dict)
            
        return model
    
    
class FintuneDiseaseAgeModel_EMB(pl.LightningModule):
    def __init__(self, model_args, vocab, scaler):
        super().__init__()
        self.vocab = vocab
        self.model_args= model_args
        self.scaler = scaler
        self.pretrained_model = self.from_pretrained(
            model_args,
            vocab,
        )
        if model_args["architecture"]=="textcnn":
            self.disease_head = textCNN(model_args)
        elif model_args["architecture"]=="resnet":
            self.disease_head = ResNet1D_Disease(
                model_args,
                in_planes=model_args["embed_dim"],
                main_planes=32,
            )
        
        self.valid_step_outputs = []
        
    def forward(self, gene_ids, values):
        embs = self.get_embeddings(self.pretrained_model, gene_ids, values)[:, 1:, :] # [batch_size, seq_len, emb_dim]
        pred_disease, pred_age = self.disease_head(embs)
        
        return pred_disease, pred_age
    
    def training_step(self, batch, batch_idx):

        gene_id, masked_value, target_value, ages_label, ages_label_norm, disease_label = batch
        pred_disease, pred_age_norm = self(gene_id, masked_value)
        
        # age prediction
        pred_age_norm= pred_age_norm.squeeze()
        
        mse_loss_norm = nn.MSELoss()(pred_age_norm, ages_label_norm.squeeze())
        mse_loss_norm = mse_loss_norm.mean()
        
        pred_age = torch.tensor(self.scaler.inverse_transform(pred_age_norm.detach().to(torch.float).cpu().numpy().reshape(-1, 1))).to(device=self.device)
        
        mse_loss = nn.MSELoss()(pred_age.squeeze(), ages_label)
        mse_loss = mse_loss.mean()
        
        # disease predition
        cls_loss = nn.BCELoss()(pred_disease, disease_label)
        cls_loss = cls_loss.mean()
        
        loss = mse_loss_norm + cls_loss
        
        
        self.log("train_mse_loss_norm", mse_loss_norm, prog_bar=True, sync_dist=True, on_epoch=True)
        self.log("train_mse_loss", mse_loss, prog_bar=True, sync_dist=True, on_epoch=True)
        self.log("train_cls", cls_loss, prog_bar=True, sync_dist=True, on_epoch=True)
        self.log("train_loss", loss, prog_bar=True, sync_dist=True, on_epoch=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):

        gene_id, masked_value, target_value, ages_label, ages_label_norm, disease_label = batch
        pred_disease, pred_age_norm = self(gene_id, masked_value)
        
        # age prediction
        pred_age_norm= pred_age_norm.squeeze()
        
        mse_loss_norm = nn.MSELoss()(pred_age_norm, ages_label_norm.squeeze())
        mse_loss_norm = mse_loss_norm.mean()
        
        pred_age = torch.tensor(self.scaler.inverse_transform(pred_age_norm.detach().to(torch.float).cpu().numpy().reshape(-1, 1))).to(device=self.device)
        
        mse_loss = nn.MSELoss()(pred_age.squeeze(), ages_label)
        mse_loss = mse_loss.mean()
        
        # disease predition
        cls_loss = nn.BCELoss()(pred_disease, disease_label)
        cls_loss = cls_loss.mean()
        
        loss = mse_loss_norm + cls_loss
        
        
        self.log("valid_mse_loss_norm", mse_loss_norm, prog_bar=True, sync_dist=True, on_epoch=True)
        self.log("valid_mse_loss", mse_loss, prog_bar=True, sync_dist=True, on_epoch=True)
        self.log("valid_cls", cls_loss, prog_bar=True, sync_dist=True, on_epoch=True)
        self.log("valid_loss", loss, prog_bar=True, sync_dist=True, on_epoch=True)
        
        result={}
        result = {
            'pred_age': pred_age.detach().cpu(),
            'pred_disease': pred_disease.detach().cpu(),
            'label_age': ages_label.detach().cpu(),
            'label_disease': disease_label.detach().cpu(),
        }
        
        self.valid_step_outputs.append(result)
        
        return result
    
    def on_validation_epoch_end(self):  
        valid_metrics = disease_age_metric(self.valid_step_outputs)
        for key, value in valid_metrics.items():
            key = f"valid_{key}"
            self.log(key, value, prog_bar=True, on_epoch=True, sync_dist=True)
            
        self.valid_step_outputs.clear()
        
    def configure_optimizers(self):
        params = [
            {
                "params": self.pretrained_model.parameters(),
                "lr": float(self.model_args["pretrained_lr"]),
                "weight_decay": float(self.model_args["weight_decay"]),
            },
            {
                "params": self.disease_head.parameters(),
                "lr": float(self.model_args["head_lr"]),
            },
        ]
        optimizer = optim.Adam(
            params,
            # betas=self.model_args["adam_betas"],
        )

        return [optimizer]
    
    def get_embeddings(self, model, gene_ids, values):
        src_key_padding_mask = gene_ids.eq(self.vocab.vocab[self.vocab.pad_token])
        transformer_output = model._encode(
            gene_ids,
            values,
            src_key_padding_mask=src_key_padding_mask,
            batch_labels=None,
        )
            
        return transformer_output
    
    def get_cell_embeddings(self, model, gene_ids, values):
        # with torch.no_grad():
        #     model.eval()
        src_key_padding_mask = gene_ids.eq(self.vocab.vocab[self.vocab.pad_token])
        output_dict = model(
            gene_ids,
            values,
            src_key_padding_mask=src_key_padding_mask,
            batch_labels=None,
        )
        cell_embeddings = output_dict["cell_emb"]
            
        return cell_embeddings
    
    def from_pretrained(self, model_args, vocab):
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
            ecs_threshold=model_args["ecs_thres"],
            explicit_zero_prob=False,
            use_fast_transformer=model_args["fast_transformer"],
            pre_norm=model_args["pre_norm"],
        )
        
        if self.model_args["pretrained_file"]!="None":
            try:
                model.load_state_dict(torch.load(model_args["pretrained_file"], map_location="cpu"))
                print(f'Loading all model params from {model_args["pretrained_file"]}')
            except:
                # only load params that are in the model and match the size
                model_dict = model.state_dict()
                pretrained_dict = torch.load(model_args["pretrained_file"], map_location="cpu")
                pretrained_dict = {
                    k: v
                    for k, v in pretrained_dict.items()
                    if k in model_dict and v.shape == model_dict[k].shape
                }
                for k, v in pretrained_dict.items():
                    print(f"Loading params {k} with shape {v.shape}")
                model_dict.update(pretrained_dict)
                model.load_state_dict(model_dict)
            
        return model
    
    
    
class FintuneDeadDayModel_EMB_MLM(pl.LightningModule):
    def __init__(self, model_args, vocab, scaler):
        super().__init__()
        self.vocab = vocab
        self.model_args= model_args
        self.scaler = scaler
        self.pretrained_model = self.from_pretrained(
            model_args,
            vocab,
        )
        self.age_head = ResNet1D(
            in_planes=model_args["layer_size"],
            main_planes=32,
            out_planes=1
        )
        self.valid_step_outputs = []
        
    def forward(self, gene_ids, values):
        # pred_mlm = self.mlm_predict(self.pretrained_model, gene_ids, values)
        embs = self.get_embeddings(self.pretrained_model, gene_ids, values)[:, 1:, :]
        embs = embs.permute(0, 2, 1)
        pred_age = self.age_head(embs)
        
        return pred_age
    
    def training_step(self, batch, batch_idx):
        gene_id, masked_value, target_value, ages_label, ages_label_norm = batch
        if self.model_args["mask_ratio"]==0:
            pred_age_norm = self(gene_id, target_value)
        else:
            pred_age_norm = self(gene_id, masked_value)
        
        # age prediction
        pred_age_norm= pred_age_norm.squeeze()
        mse_loss_norm = nn.MSELoss()(pred_age_norm, ages_label_norm.squeeze())
        mse_loss_norm = mse_loss_norm.mean()
        mae_loss_norm = nn.L1Loss()(pred_age_norm, ages_label_norm.squeeze())
        mae_loss_norm = mae_loss_norm.mean()
        
        pred_age = torch.tensor(self.scaler.inverse_transform(pred_age_norm.detach().to(torch.float).cpu().numpy().reshape(-1, 1))).to(device=self.device)
        
        mse_loss = nn.MSELoss()(pred_age.squeeze(), ages_label)
        mse_loss = mse_loss.mean()
        mae_loss= nn.L1Loss()(pred_age.squeeze(), ages_label)
        mae_loss = mae_loss.mean()
        
        loss = mse_loss_norm 
        
        
        self.log("train_mse_loss_norm", mse_loss_norm, prog_bar=True, sync_dist=True, on_epoch=True)
        self.log("train_mae_loss_norm", mae_loss_norm, prog_bar=True, sync_dist=True, on_epoch=True)
        
        self.log("train_mse_loss", mse_loss, prog_bar=True, sync_dist=True, on_epoch=True)
        self.log("train_mae_loss", mae_loss, prog_bar=True, sync_dist=True, on_epoch=True)
        
        return loss
        
    def validation_step(self, batch, batch_idx):
    
        split = "valid"
        gene_id, masked_value, target_value, ages_label, ages_label_norm = batch
        if self.model_args["mask_ratio"]==0:
            pred_age_norm = self(gene_id, target_value)
        else:
            pred_age_norm = self(gene_id, masked_value)
        
        # age prediction
        pred_age_norm= pred_age_norm.squeeze()
        mse_loss_norm = nn.MSELoss()(pred_age_norm, ages_label_norm.squeeze())
        mse_loss_norm = mse_loss_norm.mean()
        mae_loss_norm = nn.L1Loss()(pred_age_norm, ages_label_norm.squeeze())
        mae_loss_norm = mae_loss_norm.mean()
        
        pred_age = torch.tensor(self.scaler.inverse_transform(pred_age_norm.detach().to(torch.float).cpu().numpy().reshape(-1, 1))).to(device=self.device)
        
        mse_loss = nn.MSELoss()(pred_age.squeeze(), ages_label)
        mse_loss = mse_loss.mean()
        mae_loss= nn.L1Loss()(pred_age.squeeze(), ages_label)
        mae_loss = mae_loss.mean()
        
        loss = mse_loss_norm 
        
        
        self.log(f"{split}_mse_loss_norm", mse_loss_norm, prog_bar=True, sync_dist=True, on_epoch=True)
        self.log(f"{split}_loss_norm", mae_loss_norm, prog_bar=True, sync_dist=True, on_epoch=True)
        
        self.log(f"{split}_mse_loss", mse_loss, prog_bar=True, sync_dist=True, on_epoch=True)
        self.log(f"{split}_mae_loss", mae_loss, prog_bar=True, sync_dist=True, on_epoch=True)
    
        
        result={}
        result = {
            'pred_age': pred_age.detach().cpu(),
            'label': ages_label.detach().cpu(),
        }
        
        self.valid_step_outputs.append(result)
        
        return result
    
    def on_validation_epoch_end(self):  
        
        valid_metrics = regression_metric(self.valid_step_outputs)
        for key, value in valid_metrics.items():
            key = f"valid_{key}"
            self.log(key, value, prog_bar=True, on_epoch=True, sync_dist=True)
           
        self.valid_step_outputs.clear()


    def configure_optimizers(self):
        params = [
            {
                "params": self.pretrained_model.parameters(),
                "lr": float(self.model_args["pretrained_lr"]),
                "weight_decay": float(self.model_args["weight_decay"]),
            },
            {
                "params": self.age_head.parameters(),
                "lr": float(self.model_args["head_lr"]),
            },
        ]
        optimizer = optim.Adam(
            params,
            # betas=self.model_args["adam_betas"],
        )

        return [optimizer]
        
    def mlm_predict(self, model, gene_ids, values):
        src_key_padding_mask = gene_ids.eq(self.vocab.vocab[self.vocab.pad_token])
        output_dict = model(
            gene_ids,
            values,
            src_key_padding_mask=src_key_padding_mask,
            batch_labels=None,
        )
        mlm_preds = output_dict["mlm_output"]
        
        return mlm_preds
    
    def get_embeddings(self, model, gene_ids, values):

        src_key_padding_mask = gene_ids.eq(self.vocab.vocab[self.vocab.pad_token])
        transformer_output = model._encode(
            gene_ids,
            values,
            src_key_padding_mask=src_key_padding_mask,
            batch_labels=None,
        )
                
        return transformer_output
    
    def get_cell_embeddings(self, model, gene_ids, values):
        # with torch.no_grad():
        #     model.eval()
        src_key_padding_mask = gene_ids.eq(self.vocab.vocab[self.vocab.pad_token])
        output_dict = model(
            gene_ids,
            values,
            src_key_padding_mask=src_key_padding_mask,
            batch_labels=None,
        )
        cell_embeddings = output_dict["cell_emb"]
            
        return cell_embeddings
    
    def from_pretrained(self, model_args, vocab):
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
            ecs_threshold=model_args["ecs_thres"],
            explicit_zero_prob=False,
            use_fast_transformer=model_args["fast_transformer"],
            # use_fast_transformer=False,
            pre_norm=model_args["pre_norm"],
        )
        
        if self.model_args["pretrained_file"]!="None":
            try:
                model.load_state_dict(torch.load(model_args["pretrained_file"], map_location="cpu"))
                print(f'Loading all model params from {model_args["pretrained_file"]}')
            except:
                # only load params that are in the model and match the size
                model_dict = model.state_dict()
                pretrained_dict = torch.load(model_args["pretrained_file"], map_location="cpu")
                pretrained_dict = {
                    k: v
                    for k, v in pretrained_dict.items()
                    if k in model_dict and v.shape == model_dict[k].shape
                }
                for k, v in pretrained_dict.items():
                    print(f"Loading params {k} with shape {v.shape}")
                model_dict.update(pretrained_dict)
                model.load_state_dict(model_dict)
            
        return model
    
    

class FintuneDiseaseModel_RawData(pl.LightningModule):
    def __init__(self, model_args):
        super().__init__()
        self.model_args= model_args
        self.age_head = FintuneAgeHead_Altumage(
            input_dim=model_args["cpg_sites"],
            inner_dim=32,
            out_dim=1
        )
        self.validation_step_outputs = []
        
    def forward(self, gen_data):

        pred_age = self.age_head(gen_data)
        
        pred_age = nn.Sigmoid()(pred_age)
        
        return pred_age
    
    def training_step(self, batch, batch_idx):
        gen_data, ages_label = batch
        ages_label = ages_label.reshape(-1, 1)
        pred_age = self(gen_data)
        
        # disease predition
        cls_loss = nn.BCELoss()(pred_age, ages_label)
        cls_loss = cls_loss.mean()
        
 
        
        self.log("train_loss", cls_loss, prog_bar=True, sync_dist=True, on_epoch=True)
        
        return cls_loss
        
    def validation_step(self, batch, batch_idx):

        gen_data, ages_label = batch
        ages_label = ages_label.reshape(-1, 1)
        pred_age = self(gen_data)
        
        # disease predition
        cls_loss = nn.BCELoss()(pred_age, ages_label)
        cls_loss = cls_loss.mean()
        
        
        self.log("valid_loss", cls_loss, prog_bar=True, sync_dist=True, on_epoch=True)

        
        result={}
        result = {
            'pred_disease': pred_age.detach().cpu(),
            'label_disease': ages_label.detach().cpu(),
        }
        
        self.validation_step_outputs.append(result)
    
            
        
        return result
    
    def on_validation_epoch_end(self):  
        valid_metrics = disease_new_metric(self.validation_step_outputs)
        for key, value in valid_metrics.items():
            key = f"valid_{key}"
            self.log(key, value, prog_bar=True, on_epoch=True, sync_dist=True)
            
        self.validation_step_outputs.clear()


    def configure_optimizers(self):
        params = [
            {
                "params": self.age_head.parameters(),
                "lr": float(self.model_args["head_lr"]),
            },
        ]
        optimizer = optim.Adam(
            params,
            betas=(0.9, 0.999),
            eps=1e-7,
        )
        
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=30, min_lr=0.0001)   # 0.00001
        
        scheduler_dict = {"scheduler": scheduler, "interval": "epoch", "monitor": "train_loss"}
        
        return [optimizer], [scheduler_dict]
    

    
    