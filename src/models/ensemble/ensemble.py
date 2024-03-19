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
    # in_ch (input channels): Questo parametro indica il numero di canali delle immagini di input che la rete prevede di ricevere (3 nel caso di immagini a colori)
    def __init__(self, path_fcn, path_deeplab, path_unet_eff, path_unet_res, in_channels=3, classes=1, lr=5e-7, epochs=1000, len_dataset=0, batch_size=0, loss=nn.BCEWithLogitsLoss(), 
                sgm_type="hard", sgm_threshold=0.5, max_lr=1e-3, encoder_name="efficientnet-b7", encoder_weights="imagenet", 
                model_type="unet", decision_fusion = "median"):
        super().__init__()
        self.save_hyperparameters()

        # ** load correct checkpoints ** #
        # fcn
        self.fcn = load_model_from_checkpoint(path = path_fcn, model_type = "fcn")
        print("FCN caricato")

        # Deeplab
        self.deeplab = load_model_from_checkpoint(path = path_deeplab, model_type = "deeplab")
        print("DEEPLAB caricato")

        # unet
        self.unet_eff = load_model_from_checkpoint(path = path_unet_eff, model_type = "unet")
        print("UNET caricato")
        self.unet_res = load_model_from_checkpoint(path = path_unet_res, model_type = "unet")
        print("UNET caricato")

        self.unet_eff.freeze()
        self.unet_res.freeze()
        self.fcn.freeze()
        self.deeplab.freeze()

        self.num_classes=classes
        self.lr = lr
        self.loss = loss
        self.sgm_type=sgm_type
        self.sgm_threshold= sgm_threshold
        self.epochs=epochs
        self.len_dataset= len_dataset
        self.batch_size = batch_size
        self.max_lr = max_lr
        self.decision_fusion = decision_fusion

        self.all_preds = []
        self.all_labels = []


    # operations performed on the input data to produce the model's output.
    def forward(self, x, cat_id):
        out_fcn = self.fcn(x)  # caso binario = torch.Size([2, 1, 448, 448]) # caso multiclasse torch.Size([2, 4, 448, 448])
        out_deeplab = self.deeplab(x)
        out_unet_eff = self.unet_eff(x) 
        out_unet_res = self.unet_res(x) 
        
        if self.num_classes==1:
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
            else:
                print("RAMO ELSE")
        else:
            out = torch.cat((out_fcn.unsqueeze(1), out_deeplab.unsqueeze(1), out_unet_eff.unsqueeze(1), out_unet_res.unsqueeze(1)), dim=1)
            #out = (out_fcn+out_unet+out_deeplab)/3 # orch.stack((out_fcn, out_deeplab, out_unet), dim=0)
            #print(out.shape) 
            # print(out.shape) # torch.Size([2, 3, 4, 448, 448])
            # Calcola la media lungo la dimensione aggiunta (ora dimensione 1), ottenendo [2, 4, 448, 448]
            out = torch.mean(out, dim = 1)
            #print(out.shape) #torch.Size([2, 4, 448, 448])


        #plot_all_results(img = x, fcn = out_fcn, deeplab = out_deeplab, unet_eff = out_unet_eff,unet_res = out_unet_res , ensemble = out, dec_fun = self.decision_fusion, cat_id = cat_id)

        if self.num_classes == 1:  
            out = torch.sigmoid(out)
        else:
            out = torch.nn.functional.softmax(out, dim = 1)

        return out

    
    def predict_hard_mask(self, x, sgm_threshold=0.5):
        out_fcn = self.fcn(x)    
        #print(out_fcn.shape) # --> torch.Size([4, 1, 448, 448])
        out_deeplab = self.deeplab(x)
        out_unet_eff = self.unet_eff(x) 
        out_unet_res = self.unet_res(x) 

        if self.num_classes==1:
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
            else:
                print("RAMO ELSE")
        else:
            out = torch.cat((out_fcn.unsqueeze(1), out_deeplab.unsqueeze(1), out_unet_eff.unsqueeze(1), out_unet_res.unsqueeze(1)), dim=1)
            out = torch.mean(out, dim = 1)
        
        if self.num_classes == 1:  
            out = torch.sigmoid(out)
            out = (out > sgm_threshold).float()
        else:
            # out = [1, 4, 448, 448]
            out = torch.nn.functional.softmax(out, dim = 1)
            max_elements, max_idxs = torch.max(out, dim=1)
            out = max_idxs
            # out = [1, 448, 448]
        
        return out
    

    def predict_step(self, batch, batch_idx):
        return self(batch[0]) #invokes the 'forward' method of the class

        
    def test_step(self, batch, batch_idx):
        #self._common_step(batch, batch_idx, "test")
        images, masks, cat_id = batch
        logits = self(images, cat_id)

        loss = self.loss(logits , masks)
        self.log('test_loss', loss)

        
        if self.num_classes == 1:
            logits_hard = self.predict_hard_mask(images, self.sgm_threshold)
            compute_met = BinaryMetrics()
            met = compute_met(masks, logits_hard, cat_id) # met is a list
            #return loss, met
            self.log_dict(met)
        else:
            # creo matrici numpy per calcolo metriche 
            # ottengo 2 matrici 448x448 dove per ogni pixel ho 0,1,2,3 a seconda della classe predetta
            for i in range(images.shape[0]):
                logits_hard = self.predict_hard_mask(images[i:i+1, :, :, :])
                # print(logits_hard.shape) [1, 448, 448]
                #single_mask_pred_multiclass = logits_hard.unsqueeze(0)
                single_mask_pred_multiclass = torch.squeeze(logits_hard) #[448, 448]
                # print(single_mask_pred_multiclass.shape) [448, 448]
                single_mask_true_multiclass =  torch.argmax(masks[i:i+1, :, :, :], dim=1)
                single_mask_true_multiclass = single_mask_true_multiclass.squeeze(0) #[448,448]
                
                prob = single_mask_pred_multiclass.cpu().detach().numpy().flatten()
                self.all_preds.extend(prob)

                label = single_mask_true_multiclass.cpu().detach().numpy().flatten()
                self.all_labels.extend(label)


    def on_test_epoch_end(self):
        if self.num_classes > 1:
            #print(len(self.all_labels)) = 17260544
            #print(len(self.all_preds)) = 17260544
            compute_met = MultiClassMetrics_manual_v2()
            met = compute_met(self.all_labels, self.all_preds)
            self.log_dict(met) 
    
    
        

