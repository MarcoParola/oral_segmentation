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
import csv

from ...metricsHardSegmentation import *


class unetSegmentationNet(LightningModule):
    # in_ch (input channels): This parameter indicates the number of input image channels that the network expects to receive (3 in the case of color images)
    def __init__(self, in_channels=3, classes=1, lr=5e-7, epochs=1000, len_dataset=0, batch_size=0, loss=nn.BCEWithLogitsLoss(), 
                sgm_type="hard", sgm_threshold=0.5, max_lr=1e-3, encoder_name="efficientnet-b7", encoder_weights="imagenet", 
                model_type="unet", version_number = 0):
        super().__init__()
        self.save_hyperparameters(ignore=['loss'])
        
        self.model = Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes, #a number of classes for output (output shape - (batch, classes, h, w))
        )

        self.num_classes=classes
        self.lr = lr
        self.loss = loss
        self.sgm_type=sgm_type
        self.sgm_threshold= sgm_threshold
        self.epochs=epochs
        self.len_dataset= len_dataset
        self.batch_size = batch_size
        self.max_lr = max_lr
        self.version_number=version_number

        self.all_preds = []
        self.all_labels = []

    # operations performed on the input data to produce the model's output.
    def forward(self, x):
        out = self.model(x)
        return out

    def predict_hard_mask(self, x, sgm_threshold=0.5, step = "train"):
        out = self.model(x)
        if self.num_classes == 1:  
            out = torch.sigmoid(out)
            out = (out > sgm_threshold).float()
        else:
            # out = [1, 4, 448, 448]
            out = torch.nn.functional.softmax(out, dim = 1)
            if (step == "test"):
                max_elements, max_idxs = torch.max(out, dim=1)
                out = max_idxs
        return out

    def predict_step(self, batch, batch_idx):
        return self(batch[0]) #invokes the 'forward' method of the class

    #specifies what should happen in a single training step, 
    #i.e. how input (batch) data is used to calculate loss and metrics during network training.
    def training_step(self, batch, batch_idx):
        loss = self._common_step(batch, batch_idx, "train")
        return loss


    def validation_step(self, batch, batch_idx):
        loss = self._common_step(batch, batch_idx, "val") 
        return loss
        
    def test_step(self, batch, batch_idx):
        images, masks, cat_id = batch
        logits = self(images)

        loss = self.loss(logits, masks)
        self.log('test_loss', loss)

        logits_hard = self.predict_hard_mask(images, self.sgm_threshold)

        if self.num_classes == 1:
            prob = logits_hard.cpu().detach().numpy().flatten()
            self.all_preds.extend(prob)
            label = masks.cpu().detach().numpy().flatten()
            self.all_labels.extend(label)
        else:    
            logits_hard = self.predict_hard_mask(images, step = "test")
            single_mask_pred_multiclass = torch.squeeze(logits_hard) 
            single_mask_true_multiclass =  torch.argmax(masks, dim=1)
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
            met = compute_met(self.all_labels, self.all_preds, self.version_number)
            self.log_dict(met)
        else:
            compute_met = BinaryMetrics_manual()
            met = compute_met(self.all_labels, self.all_preds, self.version_number) # met is a dict
            self.log_dict(met)
    

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        sch = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr = self.max_lr, epochs=self.epochs, steps_per_epoch = int(math.ceil(self.len_dataset / self.batch_size)))
        return [optimizer], [sch]
        

    def _common_step(self, batch, batch_idx, stage):
        img, actual_mask, cat_id = batch
        mask_predicted = self(img)
        #print(mask_predicted.shape) torch.Size([4, 4, 448, 448])
        #print(actual_mask.shape) torch.Size([4, 4, 448, 448])
        if (self.num_classes>1):
            _, actual_mask_classes = actual_mask.max(dim=1) 
            loss = self.loss(mask_predicted, actual_mask_classes)
        else:
            loss = self.loss(mask_predicted, actual_mask)
        self.log(f"{stage}_loss", loss, on_step=True)

        mask_predicted_hard = self.predict_hard_mask(img, self.sgm_threshold)

        if self.num_classes == 1:
            compute_met = BinaryMetrics()
            met = compute_met(actual_mask, mask_predicted_hard, cat_id)
            self.log(f"{stage}_acc", met["pixel_acc"])
            self.log(f"{stage}_jaccard", met["jaccard"])
            self.log(f"{stage}_dice", met["dice"])
        else:
            compute_met = MultiClassMetrics()
            met = compute_met(actual_mask, mask_predicted_hard, cat_id)
            self.log(f"{stage}_acc", met["pixel_acc"])
            self.log(f"{stage}_dice", met["dice"])
            
        return loss
    
    
