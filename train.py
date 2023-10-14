import os
import hydra
import torch
import pytorch_lightning as pl
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils.multiclass import unique_labels

from src.models.fcn import FcnSegmentationNet
from src.models.deeplab import DeeplabSegmentationNet
from src.dataset import OralSegmentationDataset
from torch.utils.data import DataLoader

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models

from src.utils import *


@hydra.main(version_base=None, config_path="./config", config_name="config")
def main(cfg):

    if cfg.train.seed == -1:
        random_data = os.urandom(4)
        seed = int.from_bytes(random_data, byteorder="big")
        cfg.train.seed = seed
    torch.manual_seed(cfg.train.seed)

    callbacks = list()
    callbacks.append(get_early_stopping(cfg))
    loggers = get_loggers(cfg)

    # model
    model_deeplab = DeeplabSegmentationNet(
        #model=cfg.model.name,
        #weights=cfg.model.weights,
        num_classes=1,
        lr=cfg.train.lr,
    )

    
    model_fcn = FcnSegmentationNet(
        #model=cfg.model.name,
        #weights=cfg.model.weights,
        num_classes=1,
        lr=cfg.train.lr,
    )
    

    # datasets and dataloaders
    train_img_tranform, val_img_tranform, test_img_tranform, img_tranform = get_transformations(cfg)
    train_dataset = OralSegmentationDataset("data/train.json",transform=img_tranform)
    val_dataset = OralSegmentationDataset("data/val.json",transform=img_tranform)
    test_dataset = OralSegmentationDataset("data/test.json",transform=img_tranform)  
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64)
    test_loader = DataLoader(test_dataset, batch_size=64)

    # training
    trainer = pl.Trainer(
        logger=loggers,
        callbacks=callbacks,
        accelerator=cfg.train.accelerator,
        devices=cfg.train.devices,
        log_every_n_steps=1,
        max_epochs=cfg.train.max_epochs,
    )

    '''
    images, masks = next(iter(test_loader))
    outputs_fcn = model_fcn(images)
    outputs_deeplab = model_deeplab(images)
    print(outputs_fcn.shape, outputs_deeplab.shape, masks.shape, images.shape)

    for i in range(5):
        image, mask = images[i], masks[i]
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(12, 3))
        ax1.imshow(image.squeeze().permute(1,2,0))
        ax2.imshow(image.squeeze().permute(1,2,0), alpha=0.5)
        ax3.imshow(image.squeeze().permute(1,2,0), alpha=0.5)
        ax2.imshow(mask.permute(1,2,0).numpy(), alpha=0.6, cmap='gray')
        ax3.imshow(outputs_fcn[i].detach().permute(1,2,0).numpy(), alpha=0.6, cmap='gray')
        ax4.imshow(outputs_deeplab[i].detach().permute(1,2,0).numpy(), alpha=0.6, cmap='gray')
        print(outputs_fcn[i].max(), outputs_fcn[i].min(), outputs_deeplab[i].max(), outputs_deeplab[i].min())
        plt.show()
    '''

    trainer.fit(model_fcn, train_loader, val_loader)

    # Evaluate the model on the test set
    #trainer.test(model, test_loader)

    # plot some segmentation predictions in a plot containing three subfigure: image - actual - predicted
    images, masks = next(iter(test_loader))
    outputs = model_fcn(images)
    for i in range(20):
        image, mask = images[i], masks[i]
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4))
        ax1.imshow(image.squeeze().permute(1,2,0))
        ax2.imshow(image.squeeze().permute(1,2,0), alpha=0.5)
        ax3.imshow(image.squeeze().permute(1,2,0), alpha=0.5)
        ax2.imshow(mask.permute(1,2,0).numpy(), alpha=0.6, cmap='gray')
        ax3.imshow(outputs[i].detach().permute(1,2,0).numpy(), alpha=0.6, cmap='gray')
        print(outputs[i].shape, outputs[i].max(), outputs[i].min())
        plt.show()
    

if __name__ == "__main__":
    main()
