import hydra
import torch
import numpy as np
import torch.nn as nn
import math
from torchvision import models
from pytorch_lightning import LightningModule
from segmentation_models_pytorch import Unet
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import precision_recall_fscore_support
import csv

from ...metricsHardSegmentation import *
from ...utils import *

class ensembleSegmentationNet(LightningModule):
    def __init__(self, path_fcn, path_deeplab, path_unet_eff, path_unet_res, in_channels=3, classes=1, sgm_threshold=0.5, decision_fusion = "median", type_agg = "soft"):
        super().__init__()
        self.save_hyperparameters()

        # ** load correct checkpoints ** #
        # fcn
        self.fcn = load_model_from_checkpoint(path = path_fcn, model_type = "fcn")
        print("Loaded FCN")
        # Deeplab
        self.deeplab = load_model_from_checkpoint(path = path_deeplab, model_type = "deeplab")
        print("Loaded DEEPLAB")
        # unet
        self.unet_eff = load_model_from_checkpoint(path = path_unet_eff, model_type = "unet")
        print("Loaded UNET EFF")
        self.unet_res = load_model_from_checkpoint(path = path_unet_res, model_type = "unet")
        print("Loaded UNET RES")

        self.unet_eff.freeze()
        self.unet_res.freeze()
        self.fcn.freeze()
        self.deeplab.freeze()

        self.num_classes=classes
        self.sgm_threshold= sgm_threshold
        self.decision_fusion = decision_fusion
        self.type_agg = type_agg

        self.all_preds = []
        self.all_labels = []


    # operations performed on the input data to produce the model's output.
    def forward(self, x, cat_id):
        out_fcn = self.fcn(x)  
        out_deeplab = self.deeplab(x)
        out_unet_eff = self.unet_eff(x) 
        out_unet_res = self.unet_res(x) 
        
        if self.num_classes==1:
            out_fcn = torch.sigmoid(out_fcn)
            out_deeplab = torch.sigmoid(out_deeplab)
            out_unet_eff = torch.sigmoid(out_unet_eff)
            out_unet_res = torch.sigmoid(out_unet_res)

            if(self.type_agg == "hard"):
                # To binarize before aggregation
                out_fcn = (out_fcn > 0.48).float()
                out_deeplab = (out_deeplab > 0.41).float()
                out_unet_eff = (out_unet_eff > 0.01).float()
                out_unet_res = (out_unet_res > 0.01).float()

            out = torch.cat((out_fcn, out_deeplab, out_unet_eff, out_unet_res), dim=1)

            if self.decision_fusion == "median":
                out = torch.median(out, dim=1)[0]
                out = out.unsqueeze(1)
            elif self.decision_fusion == "mean":
                out = torch.mean(out, dim=1)
                out = out.unsqueeze(1)
            elif self.decision_fusion == "max":
                out = torch.max(out, dim=1)[0]
                out = out.unsqueeze(1)
            elif self.decision_fusion == "min":
                out = torch.min(out, dim=1)[0]
                out = out.unsqueeze(1)
            elif self.decision_fusion == "product":
                out = torch.prod(out, dim=1, keepdim=True)
            elif self.decision_fusion == "mv":
                summed_tensor = out.sum(dim=1)
                out = (summed_tensor >= 2).int()
            else:
                print("Input Error: untreated decision function")
        else:
            out_fcn = torch.nn.functional.softmax(out_fcn, dim=1)
            out_deeplab = torch.nn.functional.softmax(out_deeplab, dim=1)
            out_unet_eff = torch.nn.functional.softmax(out_unet_eff, dim=1)
            out_unet_res = torch.nn.functional.softmax(out_unet_res, dim=1)

            if(self.type_agg == "hard"):
                indices = torch.argmax(out_fcn, dim=1)
                one_hot = torch.nn.functional.one_hot(indices, num_classes=4)
                one_hot = one_hot.permute(0, 3, 1, 2)
                out_fcn = one_hot

                indices = torch.argmax(out_deeplab, dim=1)
                one_hot = torch.nn.functional.one_hot(indices, num_classes=4)
                one_hot = one_hot.permute(0, 3, 1, 2)
                out_deeplab = one_hot

                indices = torch.argmax(out_unet_eff, dim=1)
                one_hot = torch.nn.functional.one_hot(indices, num_classes=4)
                one_hot = one_hot.permute(0, 3, 1, 2)
                out_unet_eff = one_hot

                indices = torch.argmax(out_unet_res, dim=1)
                one_hot = torch.nn.functional.one_hot(indices, num_classes=4)
                one_hot = one_hot.permute(0, 3, 1, 2)
                out_unet_res = one_hot

            out = torch.cat((out_fcn.unsqueeze(1), out_deeplab.unsqueeze(1), out_unet_eff.unsqueeze(1), out_unet_res.unsqueeze(1)), dim=1)
            
            if self.decision_fusion == "median":
                out = torch.median(out, dim=1)[0]
            elif self.decision_fusion == "mean":
                out = torch.mean(out.float(), dim=1)
            elif self.decision_fusion == "max":
                out = torch.max(out, dim=1)[0]
            elif self.decision_fusion == "min":
                out = torch.min(out, dim=1)[0]
            elif self.decision_fusion == "product":
                out = torch.prod(out, dim=1)
            elif self.decision_fusion == "mv":
                summed_tensor = out.sum(dim=1)
                out = (summed_tensor >= 2).int()
            elif self.decision_fusion == "weight": #weighted max
                weights = torch.tensor([1.0, 1.0, 3.0, 1.0], dtype=torch.float32)
                weights = weights.to(out.device)
                weights /= weights.sum()
                weights = weights.view(1, 4, 1, 1, 1)
                out = (out * weights)
                out = torch.max(out, dim=1)[0]
            else:
                print("Input Error: untreated decision function")

        plot_all_results(img = x, fcn = out_fcn, deeplab = out_deeplab, unet_eff = out_unet_eff,unet_res = out_unet_res , ensemble = out, dec_fun = self.decision_fusion, cat_id = cat_id)

        return out

    
    def predict_hard_mask(self, x, sgm_threshold=0.5):
        out_fcn = self.fcn(x)    
        out_deeplab = self.deeplab(x)
        out_unet_eff = self.unet_eff(x) 
        out_unet_res = self.unet_res(x) 

        if self.num_classes==1:

            out_fcn = torch.sigmoid(out_fcn)
            out_deeplab = torch.sigmoid(out_deeplab)
            out_unet_eff = torch.sigmoid(out_unet_eff)
            out_unet_res = torch.sigmoid(out_unet_res)

            if(self.type_agg == "hard"):
                # To binarize before aggregation
                out_fcn = (out_fcn > 0.48).float()
                out_deeplab = (out_deeplab > 0.41).float()
                out_unet_eff = (out_unet_eff > 0.01).float()
                out_unet_res = (out_unet_res > 0.01).float()            

            out = torch.cat((out_fcn, out_deeplab, out_unet_eff, out_unet_res), dim=1)

            if self.decision_fusion == "median":
                out = torch.median(out, dim=1)[0]
                out = out.unsqueeze(1)
            elif self.decision_fusion == "mean":
                out = torch.mean(out, dim=1)
                out = out.unsqueeze(1)
            elif self.decision_fusion == "max":
                out = torch.max(out, dim=1)[0]
                out = out.unsqueeze(1)
            elif self.decision_fusion == "min":
                out = torch.min(out, dim=1)[0]
                out = out.unsqueeze(1)
            elif self.decision_fusion == "product":
                out = torch.prod(out, dim=1, keepdim=True)
            elif self.decision_fusion == "mv":
                summed_tensor = out.sum(dim=1)
                out = (summed_tensor >= 2).int()
            else:
                print("RAMO ELSE")
        else:
            out_fcn = torch.nn.functional.softmax(out_fcn, dim=1)
            out_deeplab = torch.nn.functional.softmax(out_deeplab, dim=1)
            out_unet_eff = torch.nn.functional.softmax(out_unet_eff, dim=1)
            out_unet_res = torch.nn.functional.softmax(out_unet_res, dim=1)

            if(self.type_agg == "hard"):
                indices = torch.argmax(out_fcn, dim=1)
                one_hot = torch.nn.functional.one_hot(indices, num_classes=4)
                one_hot = one_hot.permute(0, 3, 1, 2)
                out_fcn = one_hot

                indices = torch.argmax(out_deeplab, dim=1)
                one_hot = torch.nn.functional.one_hot(indices, num_classes=4)
                one_hot = one_hot.permute(0, 3, 1, 2)
                out_deeplab = one_hot

                indices = torch.argmax(out_unet_eff, dim=1)
                one_hot = torch.nn.functional.one_hot(indices, num_classes=4)
                one_hot = one_hot.permute(0, 3, 1, 2)
                out_unet_eff = one_hot

                indices = torch.argmax(out_unet_res, dim=1)
                one_hot = torch.nn.functional.one_hot(indices, num_classes=4)
                one_hot = one_hot.permute(0, 3, 1, 2)
                out_unet_res = one_hot
            
            out = torch.cat((out_fcn.unsqueeze(1), out_deeplab.unsqueeze(1), out_unet_eff.unsqueeze(1), out_unet_res.unsqueeze(1)), dim=1)
            
            if self.decision_fusion == "median":
                out = torch.median(out, dim=1)[0]
            elif self.decision_fusion == "mean":
                out = torch.mean(out.float(), dim=1)
            elif self.decision_fusion == "max":
                out = torch.max(out, dim=1)[0]
            elif self.decision_fusion == "min":
                out = torch.min(out, dim=1)[0]
            elif self.decision_fusion == "product":
                out = torch.prod(out, dim=1)
            elif self.decision_fusion == "mv":
                summed_tensor = out.sum(dim=1)
                out = (summed_tensor >= 2).int()
            elif self.decision_fusion == "weight": #weighted max
                weights = torch.tensor([1.0, 1.0, 3.0, 1.0], dtype=torch.float32)
                weights = weights.to(out.device)
                weights /= weights.sum()
                weights = weights.view(1, 4, 1, 1, 1)
                out = (out * weights)
                out = torch.max(out, dim=1)[0]
            else:
                print("Input Error: untreated decision function")
        
        if self.num_classes == 1:  
            out = (out > sgm_threshold).float()
        else:
            
            max_elements, max_idxs = torch.max(out, dim=1)
            out = max_idxs
            
        return out
   
    def test_step(self, batch, batch_idx):
        
        images, masks, cat_id = batch
        
        if self.num_classes == 1:
            logits_hard = self.predict_hard_mask(images, self.sgm_threshold)
            prob = logits_hard.cpu().detach().numpy().flatten()
            self.all_preds.extend(prob)
            label = masks.cpu().detach().numpy().flatten()
            self.all_labels.extend(label)
        else:
            logits_hard = self.predict_hard_mask(images)
            single_mask_pred_multiclass = torch.squeeze(logits_hard) 
            single_mask_true_multiclass =  torch.argmax(masks, dim=1)
            single_mask_true_multiclass = single_mask_true_multiclass.squeeze(0) 
            
            prob = single_mask_pred_multiclass.cpu().detach().numpy().flatten()
            self.all_preds.extend(prob)

            label = single_mask_true_multiclass.cpu().detach().numpy().flatten()
            self.all_labels.extend(label)


    def on_test_epoch_end(self):
        if self.num_classes > 1:
            compute_met = MultiClassMetrics_manual_v2()
            met = compute_met(self.all_labels, self.all_preds, version_number = 0)
            self.log_dict(met)
        else:
            compute_met = BinaryMetrics_manual()
            met = compute_met(self.all_labels, self.all_preds, version_number = 0)
            self.log_dict(met) 
    
    
        

