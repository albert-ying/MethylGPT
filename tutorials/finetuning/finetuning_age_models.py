import sys
from pathlib import Path
import methylgpt.modules.scGPT.scgpt as scgpt

current_directory = Path(__file__).parent.absolute()
from scgpt.model.model import TransformerModel
from torch import nn, optim
import numpy as np
import torch
import lightning as pl
from fintuning_age_metrics import regression_metric
import torch.optim as optim

    
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
    
class methyGPT_Age_Model(pl.LightningModule):
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
        
    def forward(self, gene_ids, values):
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