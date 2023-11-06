import hydra
import torch
import numpy as np
import torch.nn as nn
from torchvision import models
from pytorch_lightning import LightningModule
import pytorch_lightning as pl
import torch.optim as optim

from ..compute_metrics_2 import BinaryMetrics

class DeeplabSegmentationNet(pl.LightningModule):
    def __init__(self, num_classes, lr, loss=nn.BCEWithLogitsLoss(), pretrained=True):
        super().__init__()
        self.save_hyperparameters()
        self.model = models.segmentation.deeplabv3_resnet50(pretrained=pretrained)
        self.model.classifier[4] = torch.nn.Conv2d(256, num_classes, kernel_size=(1, 1), stride=(1, 1)) 
        self.num_classes = num_classes
        self.criterion = loss
        self.lr=lr

    def forward(self, x):
        return self.model(x)['out']

    #la variabile batch è fornita automaticamente dal DataLoader
    def training_step(self, batch, batch_idx):
        images, masks = batch
        outputs = self(images)
        loss = self.criterion(outputs, masks)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        images, masks = batch
        outputs = self(images)
        loss = self.criterion(outputs, masks)
        self.log('val_loss', loss)
        return loss

    def test_step(self, batch, batch_idx):
        images, masks = batch
        outputs = self(images)
        outputs = (outputs > 0.5).float()

        loss = self.criterion(outputs, masks)
        self.log('test_loss', loss)
        compute_met = BinaryMetrics()
        met = compute_met(masks, outputs) # met is a list
        #return loss, met
        self.log_dict({'test_loss': loss, 'test_acc': met[0], 'test_dice': met[1], 'test_precision': met[2], 'test_specificity': met[3], 'test_recall': met[4]})

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
