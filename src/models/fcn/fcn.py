import hydra
import torch
import numpy as np
import torch.nn as nn
import math
from torchvision import models
from pytorch_lightning import LightningModule

from ...metricsHardSegmentation import BinaryMetrics

class FcnSegmentationNet(LightningModule):
    def __init__(self, num_classes, lr=5e-7, epochs=1000, len_dataset=0, batch_size=0, loss=nn.BCEWithLogitsLoss(), sgm_type="hard", sgm_threshold=0.5, model_type="fcn"):
        super(FcnSegmentationNet, self).__init__() # call parent's constructor function to inherit its methods
        self.save_hyperparameters()

        self.pretrained_model = models.segmentation.fcn_resnet50(pretrained=True) # use a pretrained network
        #replacing the fifth classifier layer of the pre-trained model with a new convolution layer
        #which will transform the 512 feature maps into a number of channels equal to num_classes.
        self.pretrained_model.classifier[4] = nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1))
        self.lr = lr
        self.loss = loss
        self.sgm_type=sgm_type
        self.sgm_threshold= sgm_threshold
        self.epochs=epochs
        self.len_dataset= len_dataset
        self.batch_size = batch_size

    # operations performed on the input data to produce the model's output.
    def forward(self, x):
        out = self.pretrained_model(x)['out']
        out = (out > self.sgm_threshold).float()
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
        images, masks = batch
        outputs = self(images)

        loss = self.loss(outputs, masks)
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
        mask_predicted = self.pretrained_model(img)['out']
        loss = self.loss(mask_predicted, actual_mask)
        self.log(f"{stage}_loss", loss, on_step=True)
        compute_met = BinaryMetrics()
        met = compute_met(actual_mask, mask_predicted)
        self.log(f"{stage}_acc", met[0])
        self.log(f"{stage}_jaccard", met[5])
        self.log(f"{stage}_dice", met[1])
        return loss
