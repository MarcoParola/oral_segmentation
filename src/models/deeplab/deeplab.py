import hydra
import torch
import numpy as np
import torch.nn as nn
import math
from torchvision import models
from pytorch_lightning import LightningModule
import pytorch_lightning as pl
import torch.optim as optim

from ...metricsHardSegmentation import BinaryMetrics

class DeeplabSegmentationNet(pl.LightningModule):
    def __init__(self, num_classes, lr=5e-7, epochs=1000, len_dataset=0, batch_size=0, loss=nn.BCEWithLogitsLoss(), pretrained=True, sgm_type="hard", sgm_threshold=0.5):
        super().__init__()
        self.save_hyperparameters()
        self.model = models.segmentation.deeplabv3_resnet50(pretrained=pretrained)
        # Questa riga modifica l'ultimo strato del classificatore del modello DeepLabV3 per avere num_classes canali di uscita. 
        # Questo è necessario perché il modello preaddestrato avrà un numero diverso di canali nell'ultimo strato a seconda del dataset su cui è stato preaddestrato. 
        # Qui si adatta il modello alle esigenze specifiche del problema di segmentazione.
        self.model.classifier[4] = torch.nn.Conv2d(256, num_classes, kernel_size=(1, 1), stride=(1, 1)) # TODO fai check sul valore 256
        self.num_classes = num_classes
        self.criterion = loss
        self.lr=lr
        self.sgm_type=sgm_type
        self.sgm_threshold= sgm_threshold
        self.epochs=epochs
        self.len_dataset= len_dataset
        self.batch_size = batch_size

    #To run data through your model only. Called with output = model(input_data)
    def forward(self, x):
        out = self.model(x)['out']
        out = (out > self.sgm_threshold).float()
        return out

    def predict_soft_mask(self, x):
        out = self.model(x)['out']
        return out

    #la variabile batch è fornita automaticamente dal DataLoader
    def training_step(self, batch, batch_idx):
        loss = self._common_step(batch, batch_idx, "train")
        #images, masks = batch
        #outputs = self.model(images)['out']
        #loss = self.criterion(outputs, masks)
        #self.log('train_loss', loss)
        #compute_met = BinaryMetrics()
        #met = compute_met(masks, outputs)
        #self.log('train_acc', met[0])
        #self.log('train_jaccard', met[1])
        #self.log('train_dice', met[5])
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._common_step(batch, batch_idx, "val")
        #images, masks = batch
        #outputs = self.model(images)['out']
        #loss = self.criterion(outputs, masks)
        #self.log('val_loss', loss)
        #compute_met = BinaryMetrics()
        #met = compute_met(masks, outputs)
        #self.log('val_acc', met[0])
        #self.log('val_jaccard', met[1])
        #self.log('val_dice', met[5])
        return loss

    def test_step(self, batch, batch_idx):
        images, masks = batch
        outputs = self(images)

        loss = self.criterion(outputs, masks)
        self.log('test_loss', loss)
        compute_met = BinaryMetrics()
        met = compute_met(masks, outputs) # met is a list
        #return loss, met
        self.log_dict({'test_loss': loss, 'test_acc': met[0], 'test_dice': met[1], 'test_precision': met[2], 'test_specificity': met[3], 'test_recall': met[4], 'jaccard': met[5]})
        self.log('test_acc', met[0])
        self.log('test_dice', met[1])
        self.log('test_precision', met[2])
        self.log('test_specificity', met[3])
        self.log('test_recall', met[4])
        self.log('test_jaccard', met[5])

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        sch = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr = 0.01, epochs=self.epochs, steps_per_epoch = int(math.ceil(self.len_dataset / self.batch_size)))
        return [optimizer], [sch]

    def _common_step(self, batch, batch_idx, stage):
        img, actual_mask = batch
        mask_predicted = self.model(img)['out']
        loss = self.criterion(mask_predicted, actual_mask)
        self.log(f"{stage}_loss", loss, on_step=True)
        compute_met = BinaryMetrics()
        met = compute_met(actual_mask, mask_predicted)
        self.log(f"{stage}_acc", met[0])
        self.log(f"{stage}_jaccard", met[5])
        self.log(f"{stage}_dice", met[1])
        return loss
         