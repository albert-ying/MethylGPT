import sys
from pathlib import Path
import methylgpt.modules.scGPT.scgpt as scgpt

current_directory = Path(__file__).parent.absolute()
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, median_absolute_error
from scipy.stats import spearmanr, pearsonr
import numpy as np
import torch

def regression_metric(validation_step_outputs):
    preds_item=[]
    labels_item=[]
    for batch in validation_step_outputs:
        preds_item.append(batch["pred_age"])
        labels_item.append(batch["label"])
        
    preds = torch.cat(preds_item).detach().to(torch.float).cpu().numpy()
    labels = torch.cat(labels_item).detach().to(torch.float).cpu().numpy()
    
    R2 = r2_score(labels, preds)
    
    rmse = np.sqrt(mean_squared_error(labels, preds))
    mae = mean_absolute_error(labels, preds)
    mae_altumAge = median_absolute_error(labels, preds)
    
    try:
        pearson_r = pearsonr(labels, preds)[0]
    except:
        pearson_r = -1e-9
    try:
        sp_cor = spearmanr(labels, preds)[0]
    except:
        sp_cor = -1e-9
        
    return {
        "r2": R2,
        "rmse": rmse,
        "mae": mae,
        "p_r": pearson_r,
        "s_r": sp_cor,
        "medae": mae_altumAge
    }