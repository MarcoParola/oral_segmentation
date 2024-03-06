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
from .unet_modules import U_Net


class unetSegmentationNet(LightningModule):
    # in_ch (input channels): Questo parametro indica il numero di canali delle immagini di input che la rete prevede di ricevere (3 nel caso di immagini a colori)
    def __init__(self, in_channels=3, classes=1, lr=5e-7, epochs=1000, len_dataset=0, batch_size=0, loss=nn.BCEWithLogitsLoss(), 
                sgm_type="hard", sgm_threshold=0.5, max_lr=1e-3, encoder_name="efficientnet-b7", encoder_weights="imagenet", 
                model_type="unet"):
        super().__init__()
        self.save_hyperparameters()
        
        self.model = Unet(
            encoder_name=self.hparams.encoder_name,
            encoder_weights=self.hparams.encoder_weights,
            in_channels=self.hparams.in_channels,
            classes=self.hparams.classes, #a number of classes for output (output shape - (batch, classes, h, w))
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

    # operations performed on the input data to produce the model's output.
    def forward(self, x):
        out = self.model(x)      

        if self.num_classes == 1:  
            out = torch.sigmoid(out)
        else:
            out = torch.nn.functional.softmax(out, dim = 1) # dim = è l'indice che indica le classi (B,C,H,W)
        # Nel caso multiclasse è meglio utilizzare una softmax in modo che la somma delle probabilità delle classi faccia 1
        # se tale somma non fa uno l'allenamento della rete diventa molto complicato perchè (se la classe è 1) 
        # devo forzare ad avere come output +inf,-inf,-inf in modo da avere, dopo l'applicazione della singmoide, un qualcosa
        # che si avvicini a 1,0,0
        # "Sigmoid is used for binary classification methods where we only have 2 classes, while SoftMax applies to multiclass problems. 
        # In fact, the SoftMax function is an extension of the Sigmoid function."
        return out

    def predict_hard_mask(self, x, sgm_threshold=0.5):
        out = self.model(x)
        out = (out > sgm_threshold).float()
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
        #self._common_step(batch, batch_idx, "test")
        images, masks, cat_id = batch
        logits = self(images)

        loss = self.loss(logits , masks)
        self.log('test_loss', loss)

        logits_hard = self.predict_hard_mask(images, self.sgm_threshold)

        if self.num_classes == 1:
            compute_met = BinaryMetrics()
            met = compute_met(masks, logits_hard) # met is a list
            #return loss, met
            self.log_dict({'test_loss': loss, 'test_acc': met[0], 'test_dice': met[1], 'test_precision': met[2], 'test_specificity': met[3], 'test_recall': met[4], 'jaccard': met[5]})
            self.log('test_acc', met[0])
            self.log('test_dice', met[1])
            self.log('test_precision', met[2])
            self.log('test_specificity', met[3])
            self.log('test_recall', met[4])
            self.log('test_jaccard', met[5]) 
        else:
            compute_met = MultiClassMetrics()
            # masks.shape = logits_hard.shape = [1, 3, 448, 448]
            met = compute_met(masks, logits_hard)
            # met = pixel_acc, dice, precision, recall
            self.log("test_acc", met[0])
            self.log("test_precision", met[2])
            self.log("test_recall", met[3])
            self.log("test_dice", met[1]) 


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        sch = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr = self.max_lr, epochs=self.epochs, steps_per_epoch = int(math.ceil(self.len_dataset / self.batch_size)))
        return [optimizer], [sch]
        

    def _common_step(self, batch, batch_idx, stage):
        img, actual_mask, cat_id = batch
        mask_predicted = self.model(img)
        loss = self.loss(mask_predicted, actual_mask)
        self.log(f"{stage}_loss", loss, on_step=True)

        mask_predicted_hard = self.predict_hard_mask(img, self.sgm_threshold)

        if self.num_classes == 1:
            compute_met = BinaryMetrics()
            met = compute_met(actual_mask, mask_predicted_hard)
            # met = pixel_acc, dice, precision, specificity, recall, jaccard 
            self.log(f"{stage}_acc", met[0])
            self.log(f"{stage}_jaccard", met[5])
            self.log(f"{stage}_dice", met[1])
        else:
            compute_met = MultiClassMetrics()
            met = compute_met(actual_mask, mask_predicted_hard)
            # met = pixel_acc, dice, precision, recall
            self.log(f"{stage}_acc", met[0])
            self.log(f"{stage}_dice", met[1])
            
        return loss
