import sys
from pathlib import Path
sys.version
sys.path.append('/mnt/environments/scgpt_env/lib/python3.10/site-packages')
sys.path.append("/home/A.Y/project/methylGPT/modules/scGPT/")
current_directory = Path(__file__).parent.absolute()
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, median_absolute_error
from scipy.stats import spearmanr, pearsonr
from scgpt.model.model import AdversarialDiscriminator, TransformerModel
from scgpt.tokenizer import tokenize_and_pad_batch, random_mask_value
import torch.nn.functional as F
from torch import nn, optim
from sklearn import preprocessing
import polars as pls
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
import pickle
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, auc

def elasticnet_metric(labels, preds):

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
    
def cal_baseline_metric(baseline_name):
    parquet_file = "/home/A.Y/project/methylGPT/finetuning/age_prediction/age_data/20Jun2024_predicted_age_part1_preprared.parquet"
    df = pls.read_parquet(parquet_file)

    # df_train = df.filter(df['split'] == 'train')
    df_val = df.filter(df['split'] == 'val')
    
    col_name = "Predicted_" + baseline_name 
    
    preds = df_val[col_name].to_numpy()
    labels = df_val["Age"].to_numpy()
    
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
        "pearson_r": pearson_r,
        "spearman_r": sp_cor,
        "medae": mae_altumAge
    }
    
class EvalMultiLabel(object):  
    def __init__(self, y_trues, y_preds):     
        self.y_trues = y_trues
        self.y_preds = y_preds
    
        def some_samples(y_trues,y_preds):
            
            if len(y_trues) == len(y_preds):
                tp = 0
                fn = 0
                fp = 0
                tn = 0
                for i in range(len(y_trues)):
                    y_true = y_trues[i]
                    y_pred = y_preds[i]
                    tpi,fni,fpi,tni = single_sample(y_true,y_pred)
                    tp = tp + tpi
                    fn = fn + fni
                    fp = fp + fpi
                    tn = tn + tni
                return tp,fn,fp,tn
            else:
                print('Different length between y_trues and y_preds!')
                return 0,0,0,0
              
        def single_sample(y_true,y_pred):
                 
            y_true = list(set(y_true))
            y_pred = list(set(y_pred))
            y_ = list(set(y_true) | set(y_pred))
            K = len(y_)
            tp1 = 0
            fn1 = 0
            fp1 = 0
            tn1 = 0
            for i in range(len(y_)):
                if y_[i] in y_true and y_[i] in y_pred:
                    tp1 = tp1 + 1/K
                elif y_[i] in y_true and y_[i] not in y_pred:
                    fn1 = fn1 + 1/K
                elif y_[i] not in y_true and y_[i] in y_pred:
                    fp1 = fp1 + 1/K  
                elif y_[i] not in y_true and y_[i] not in y_pred:
                    tn1 = tn1 + 1/K
            return tp1,fn1,fp1,tn1
        
        self.tp,self.fn,self.fp,self.tn = some_samples(self.y_trues,self.y_preds)
        try:
            self.recall = self.tp/(self.tp+self.fn)
        except:
            self.recall = 0
        try:
            self.precision = self.tp/(self.tp+self.fp)
        except:
            self.precision = 0
        try:
            self.f1 = 2*self.recall*self.precision/(self.precision+self.recall)
        except:
            self.f1 = 0
        

def disease_age_metric(validation_step_outputs):
    preds_age_item=[]
    labels_age_item=[]
    preds_disease_item=[]
    labels_disease_item=[]
    threshold = 0.5
    for batch in validation_step_outputs:
        preds_age_item.append(batch["pred_age"])
        labels_age_item.append(batch["label_age"])
        preds_disease_item.append(batch["pred_disease"])
        labels_disease_item.append(batch["label_disease"])
        
    preds_age = torch.cat(preds_age_item).detach().to(torch.float).cpu().numpy()
    labels_age = torch.cat(labels_age_item).detach().to(torch.float).cpu().numpy()
    preds_disease = torch.cat(preds_disease_item).detach().to(torch.float).cpu().numpy()
    preds_disease = np.where(preds_disease >= threshold, 1, 0)
    labels_disease = torch.cat(labels_disease_item).detach().to(torch.float).cpu().numpy()
    
    # age metrics
    R2 = r2_score(labels_age, preds_age)
    
    rmse = np.sqrt(mean_squared_error(labels_age, preds_age))
    mae = mean_absolute_error(labels_age, preds_age)
    mae_altumAge = median_absolute_error(labels_age, preds_age)
    
    try:
        pearson_r = pearsonr(labels_age, preds_age)[0]
    except:
        pearson_r = -1e-9
    try:
        sp_cor = spearmanr(labels_age, preds_age)[0]
    except:
        sp_cor = -1e-9
    
    
    EML = EvalMultiLabel(labels_disease, preds_disease)
        

        
    try:
        accuracy_micro = accuracy_score(labels_disease, preds_disease) 
        precision_micro = precision_score(labels_disease, preds_disease, average='micro')  
        recall_micro = recall_score(labels_disease, preds_disease, average='micro')  
        f1_micro = f1_score(labels_disease, preds_disease, average='micro') 
        auc_micro = roc_auc_score(labels_disease, preds_disease, average='micro')
    except:
        accuracy_micro = 0
        precision_micro = 0
        recall_micro = 0
        f1_micro = 0
        auc_micro = 0
        
    try:
        accuracy_macro = accuracy_score(labels_disease, preds_disease) 
        precision_macro = precision_score(labels_disease, preds_disease, average='macro')  
        recall_macro = recall_score(labels_disease, preds_disease, average='macro')  
        f1_macro = f1_score(labels_disease, preds_disease, average='macro')  
        auc_macro = roc_auc_score(labels_disease, preds_disease, average='macro')
    except:
        accuracy_macro = 0
        precision_macro = 0
        recall_macro = 0
        f1_macro = 0
        auc_macro = 0
    
    try:
        accuracy_wt = accuracy_score(labels_disease, preds_disease) 
        precision_wt = precision_score(labels_disease, preds_disease, average='weighted')  
        recall_wt = recall_score(labels_disease, preds_disease, average='weighted')  
        f1_wt = f1_score(labels_disease, preds_disease, average='weighted')  
        auc_wt = roc_auc_score(labels_disease, preds_disease, average='weighted')
    except:
        accuracy_wt = 0
        precision_wt = 0
        recall_wt = 0
        f1_wt = 0
        auc_wt = 0
        
        return {
            "r2": R2,
            "rmse": rmse,
            "mae": mae,
            "p_r": pearson_r,
            "s_r": sp_cor,
            "medae": mae_altumAge,
            "acc_micro": accuracy_micro,
            "pr_micro": precision_micro,
            "rec_micro": recall_micro,
            "f1_micro": f1_micro,
            "auc_micro": auc_micro,
            "acc_macro": accuracy_macro,
            "pr_macro": precision_macro,
            "rec_macro": recall_macro,
            "f1_macro": f1_macro,
            "auc_macro": auc_macro,
            "acc_wt": accuracy_wt,
            "pr_wt": precision_wt,
            "rec_wt": recall_wt,
            "f1_wt": f1_wt,
            "auc_wt": auc_wt,
            "rec_eml": EML.recall,
            "pr_eml": EML.precision,
            "f1_eml": EML.f1
        }
        

def disease_metric(validation_step_outputs):
    preds_disease_item=[]
    labels_disease_item=[]
    threshold = 0.5
    for batch in validation_step_outputs:
        preds_disease_item.append(batch["pred_disease"])
        labels_disease_item.append(batch["label_disease"])
        
    preds_disease = torch.cat(preds_disease_item).detach().to(torch.float).cpu().numpy()
    preds_disease = np.where(preds_disease >= threshold, 1, 0)
    labels_disease = torch.cat(labels_disease_item).detach().to(torch.float).cpu().numpy()
    
    
    
    EML = EvalMultiLabel(labels_disease, preds_disease)
        

        
    try:
        accuracy_micro = accuracy_score(labels_disease, preds_disease) 
        precision_micro = precision_score(labels_disease, preds_disease, average='micro')  
        recall_micro = recall_score(labels_disease, preds_disease, average='micro')  
        f1_micro = f1_score(labels_disease, preds_disease, average='micro') 
        auc_micro = roc_auc_score(labels_disease, preds_disease, average='micro')
    except:
        accuracy_micro = 0
        precision_micro = 0
        recall_micro = 0
        f1_micro = 0
        auc_micro = 0
        
    try:
        accuracy_macro = accuracy_score(labels_disease, preds_disease) 
        precision_macro = precision_score(labels_disease, preds_disease, average='macro')  
        recall_macro = recall_score(labels_disease, preds_disease, average='macro')  
        f1_macro = f1_score(labels_disease, preds_disease, average='macro')  
        auc_macro = roc_auc_score(labels_disease, preds_disease, average='macro')
    except:
        accuracy_macro = 0
        precision_macro = 0
        recall_macro = 0
        f1_macro = 0
        auc_macro = 0
    
    try:
        accuracy_wt = accuracy_score(labels_disease, preds_disease) 
        precision_wt = precision_score(labels_disease, preds_disease, average='weighted')  
        recall_wt = recall_score(labels_disease, preds_disease, average='weighted')  
        f1_wt = f1_score(labels_disease, preds_disease, average='weighted')  
        auc_wt = roc_auc_score(labels_disease, preds_disease, average='weighted')
    except:
        accuracy_wt = 0
        precision_wt = 0
        recall_wt = 0
        f1_wt = 0
        auc_wt = 0
        
        return {
            "acc_micro": accuracy_micro,
            "pr_micro": precision_micro,
            "rec_micro": recall_micro,
            "f1_micro": f1_micro,
            "auc_micro": auc_micro,
            "acc_macro": accuracy_macro,
            "pr_macro": precision_macro,
            "rec_macro": recall_macro,
            "f1_macro": f1_macro,
            "auc_macro": auc_macro,
            "acc_wt": accuracy_wt,
            "pr_wt": precision_wt,
            "rec_wt": recall_wt,
            "f1_wt": f1_wt,
            "auc_wt": auc_wt,
            "rec_eml": EML.recall,
            "pr_eml": EML.precision,
            "f1_eml": EML.f1
        }
        

def disease_new_metric(
    validation_step_outputs,
    step: float = 0.001,
):
    T = np.arange(0, 1 + step, step)[None, :]  # threshold value [1,1001]
    tp = 0; fn = 0; fp = 0; tn = 0
    
    sample_ids =[]
    preds_disease_item=[]
    labels_disease_item=[]
    
    # with open(f'/home/A.Y/project/methylGPT/finetuning/age_prediction/eval_results/prediction_results_preprocessed_disease_category_test_296.pkl', 'wb') as f:
    #     pickle.dump(validation_step_outputs, f)
    # import ipdb; ipdb.set_trace()
    for batch in validation_step_outputs:
        preds_disease_item.append(batch["pred_disease"])
        labels_disease_item.append(batch["label_disease"])
        # sample_ids.append(batch["sample_id"])
        
    targets = torch.cat(labels_disease_item).detach().to(torch.float).cpu().numpy()
    predictions = torch.cat(preds_disease_item).detach().to(torch.float).cpu().numpy()
    # with open(f'/home/A.Y/project/methylGPT/finetuning/age_prediction/eval_results/prediciton_preprocessed_disease_category_valid_295.pkl', 'wb') as f:
    #     pickle.dump(predictions, f)
    # with open(f'/home/A.Y/project/methylGPT/finetuning/age_prediction/eval_results/target_preprocessed_disease_category_valid_295.pkl', 'wb') as f:
    #     pickle.dump(targets, f)
    # import ipdb; ipdb.set_trace()
    fprs = []
    tprs = []
    roc_aucs = []
    # for i in range(predictions.shape[1]):
    #     fpr, tpr, thresholds = roc_curve(targets[:,i].reshape(-1, 1), predictions[:,i].reshape(-1, 1))
    #     roc_auc = auc(fpr, tpr)
    #     fprs.append(fpr)
    #     tprs.append(tpr)
    #     roc_aucs.append(roc_auc)
        
        
    # with open(f'/home/A.Y/project/methylGPT/finetuning/age_prediction/eval_results/preprocessed_disease_category_test_296_fprs.pkl', 'wb') as f:
    #     pickle.dump(fprs, f)
    # with open(f'/home/A.Y/project/methylGPT/finetuning/age_prediction/eval_results/preprocessed_disease_category_test_296_tprs.pkl', 'wb') as f:
    #     pickle.dump(tprs, f)
    # with open(f'/home/A.Y/project/methylGPT/finetuning/age_prediction/eval_results/preprocessed_disease_category_test_296_aucs.pkl', 'wb') as f:
    #     pickle.dump(roc_aucs, f)
        
        
    # import ipdb; ipdb.set_trace()
    targets = targets.reshape(-1, 1)    #[n,1]
    predictions = predictions.reshape(-1, 1)


    outputs_T = np.greater_equal(predictions, T)
    tp += np.sum(np.logical_and(outputs_T, targets), axis=0)
    tn += np.sum(np.logical_and(np.logical_not(outputs_T), np.logical_not(targets)), axis=0)
    fp += np.sum(np.logical_and(outputs_T, np.logical_not(targets)), axis=0)
    fn += np.sum(np.logical_and(np.logical_not(outputs_T), targets), axis=0)
    # prec = tp / (tp + fp).astype(float)  # precision
    # recall = tp / (tp + fn).astype(float)  # recall
    sens = tp / (tp + fn).astype(float)  # senstivity
    spec = tn / (tn + fp).astype(float)  # spec
    # TPR = tp / (tp + fn).astype(float)
    # FPR = fp / (tn + fp).astype(float)n
    ACC = (tp + tn) / (tp + tn + fp + fn).astype(float)  # accuracy
    AUC = np.trapz(y=sens, x=spec)
    

    fpr, tpr, thresholds = roc_curve(targets, predictions)

    # Step 3: Calculate AUC
    roc_auc = auc(fpr, tpr)
    
    
    # Save fpr as a pickle file
    with open('/home/A.Y/project/methylGPT/finetuning/age_prediction/eval_results/preprocessed_disease_category_valid_295.pkl', 'wb') as f:
        pickle.dump({"tpr": tpr, "fpr": fpr, "auc":roc_auc}, f)

    import ipdb; ipdb.set_trace()
    
    
    
    # threshold = T[:, np.nanargmax(ACC)][0]
    # ACC = np.nanmax(ACC)
    threshold = 0.6140000224113464  
    predictions = np.where(predictions >= threshold, 1, 0)
    accuracy_macro = accuracy_score(targets, predictions) 
    
    
    # prec[np.isnan(prec)] = 0
    # # F1 = 2 * ((prec * sens) / (prec + sens))
    # # threshold = T[:, np.nanargmax(F1)][0]
    # # F1 = np.nanmax(F1)   # F1 Score
    # # Recall = torch.tensor(recall)
    # PR1 = auc(recall, prec)
    # PR = torch.tensor(np.trapz(y=recall, x=prec))  # average precision-recall value
    # import ipdb; ipdb.set_trace()



    return {"AUC":AUC, "ACC":ACC, "Thresh":threshold}   
    
    
    
def disease_category_metric(
    validation_step_outputs,
    model_args,
    step: float = 0.001,
    data_type: str = "valid",
):
    T = np.arange(0, 1 + step, step)[None, :]  # threshold value [1,1001]
    tp = 0; fn = 0; fp = 0; tn = 0
    
    sample_ids =[]
    preds_disease_item=[]
    labels_disease_item=[]
    
    if model_args["mode"]=='valid':
        if data_type=="valid":
    
            with open(f'/home/A.Y/project/methylGPT/finetuning/age_prediction/eval_results/disease_category_prediction_results_methGPT_random_valid_3351.pkl', 'wb') as f:
                pickle.dump(validation_step_outputs, f)
        elif data_type=="test":
            with open(f'/home/A.Y/project/methylGPT/finetuning/age_prediction/eval_results/disease_category_prediction_results_methGPT_random_test_3351.pkl', 'wb') as f:
                pickle.dump(validation_step_outputs, f)
    # import ipdb; ipdb.set_trace()
    for batch in validation_step_outputs:
        preds_disease_item.append(batch["pred_disease"])
        labels_disease_item.append(batch["label_disease"])
        sample_ids.append(batch["sample_id"])
        
    targets = torch.cat(labels_disease_item).detach().to(torch.float).cpu().numpy()
    predictions = torch.cat(preds_disease_item).detach().to(torch.float).cpu().numpy()
    
    if model_args["mode"]=='valid':
        
        fpr, tpr, thresholds = roc_curve(targets.reshape(-1, 1), predictions.reshape(-1, 1))
        roc_auc = auc(fpr, tpr)
       
        with open(f'/home/A.Y/project/methylGPT/finetuning/age_prediction/eval_results/preprocess_disease_category_{data_type}_3351_methGPT_random.pkl', 'wb') as f:
            pickle.dump({"fpr": fpr, "tpr": tpr, "auc": roc_auc}, f)

    targets = targets.reshape(-1, 1)    #[n,1]
    predictions = predictions.reshape(-1, 1)


    outputs_T = np.greater_equal(predictions, T)
    tp += np.sum(np.logical_and(outputs_T, targets), axis=0)
    tn += np.sum(np.logical_and(np.logical_not(outputs_T), np.logical_not(targets)), axis=0)
    fp += np.sum(np.logical_and(outputs_T, np.logical_not(targets)), axis=0)
    fn += np.sum(np.logical_and(np.logical_not(outputs_T), targets), axis=0)
    # prec = tp / (tp + fp).astype(float)  # precision
    # recall = tp / (tp + fn).astype(float)  # recall
    sens = tp / (tp + fn).astype(float)  # senstivity
    spec = tn / (tn + fp).astype(float)  # spec
    # TPR = tp / (tp + fn).astype(float)
    # FPR = fp / (tn + fp).astype(float)n
    ACC = (tp + tn) / (tp + tn + fp + fn).astype(float)  # accuracy
    AUC = np.trapz(y=sens, x=spec)
    

    fpr, tpr, thresholds = roc_curve(targets, predictions)

    roc_auc = auc(fpr, tpr)
    
    
    threshold = T[:, np.nanargmax(ACC)][0]
    ACC = np.nanmax(ACC)


    return {"AUC":roc_auc, "ACC":ACC, "Thresh":threshold}
    
    