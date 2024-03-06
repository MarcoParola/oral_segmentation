import hydra
import torch
import numpy as np
import torch.nn as nn
import math
from torchvision import models
from pytorch_lightning import LightningModule

from ...metricsHardSegmentation import *

class FcnSegmentationNet(LightningModule):
    def __init__(self, classes=1, lr=5e-7, epochs=1000, len_dataset=0, batch_size=0, loss=nn.BCEWithLogitsLoss(), sgm_type="hard", sgm_threshold=0.5, max_lr=1e-3, model_type="fcn"):
        super(FcnSegmentationNet, self).__init__() # call parent's constructor function to inherit its methods
        self.save_hyperparameters()

        self.model = models.segmentation.fcn_resnet50(pretrained=True) # use a pretrained network
        #replacing the fifth classifier layer of the pre-trained model with a new convolution layer
        #which will transform the 512 feature maps into a number of channels equal to num_classes.
        self.model.classifier[4] = nn.Conv2d(512, classes, kernel_size=(1, 1), stride=(1, 1))
        
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
        out = self.model(x)['out']
        if self.num_classes == 1:  
            out = torch.sigmoid(out)
        else:
            out = torch.nn.functional.softmax(out, dim = 1) # dim = è l'indice che indica le classi (B,C,H,W)
        return out

    def predict_hard_mask(self, x, sgm_threshold=0.5):
        out = self.model(x)['out']
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

        loss = self.loss(logits, masks)
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
        mask_predicted = self.model(img)['out']
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
