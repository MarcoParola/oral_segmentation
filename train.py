import os
import hydra
import torch
import pytorch_lightning as pl
import numpy as np
import matplotlib.pyplot as plt
import argparse
from sklearn.utils.multiclass import unique_labels
import torch.nn as nn

from src.models.fcn import FcnSegmentationNet
from src.models.deeplab import DeeplabSegmentationNet
from src.models.unet import unetSegmentationNet

from src.datasets import BinarySegmentationDataset
from src.datasets import MultiClassSegmentationDataset
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
    callbacks.append(get_early_stopping(cfg)) # utils function
    loggers = get_loggers(cfg)  # utils function


    # datasets and dataloaders
    train_img_tranform, val_img_tranform, test_img_tranform, img_tranform = get_transformations(cfg)    # utils function

    if cfg.model.num_classes == 1:
        train_dataset = BinarySegmentationDataset(cfg.dataset.train, transform=img_tranform)
        val_dataset = BinarySegmentationDataset(cfg.dataset.val, transform=val_img_tranform)
        test_dataset = BinarySegmentationDataset(cfg.dataset.test, transform=test_img_tranform)
    else:
        train_dataset = MultiClassSegmentationDataset(cfg.dataset.train, transform=img_tranform, n_classes = cfg.model.num_classes)
        val_dataset = MultiClassSegmentationDataset(cfg.dataset.val, transform=val_img_tranform, n_classes = cfg.model.num_classes)
        test_dataset = MultiClassSegmentationDataset(cfg.dataset.test, transform=test_img_tranform, n_classes = cfg.model.num_classes)

    bs = cfg.train.batch_size
    if(cfg.model.num_classes>1):
        classes = cfg.model.num_classes + 1
        loss_fun = nn.CrossEntropyLoss()
    else:
        classes = 1
        loss_fun = nn.BCEWithLogitsLoss()

    train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True, num_workers=11)
    val_loader = DataLoader(val_dataset, batch_size=bs, num_workers=11)
    test_loader = DataLoader(test_dataset, batch_size=bs, num_workers=11)

    # model
    if (cfg.model.model_type == "fcn"):
        print("Run FCN")
        print("lr: "+ str(cfg.train.lr))
        print("max_lr: "+ str(cfg.train.max_lr))
        print("classi: " + str(classes))
        model = FcnSegmentationNet(classes=classes, lr=cfg.train.lr, epochs=cfg.train.max_epochs, loss = loss_fun, sgm_type = cfg.model.sgm_type, sgm_threshold=cfg.model.sgm_threshold, len_dataset = train_dataset.__len__(), batch_size = bs, max_lr=cfg.train.max_lr, model_type="fcn")
    elif(cfg.model.model_type == "deeplab"):
        print("Run DeepLAb")
        print("lr: "+ str(cfg.train.lr))
        print("max_lr: "+ str(cfg.train.max_lr))
        print("classi: " + str(classes))
        model = DeeplabSegmentationNet(classes=classes, lr=cfg.train.lr, epochs=cfg.train.max_epochs, loss = loss_fun, sgm_type = cfg.model.sgm_type, sgm_threshold=cfg.model.sgm_threshold, len_dataset = train_dataset.__len__(), batch_size = bs, max_lr=cfg.train.max_lr, model_type="deeplab")
    elif (cfg.model.model_type == "unet"):
        print("Run Unet")
        print("lr: "+ str(cfg.train.lr))
        print("max_lr: "+ str(cfg.train.max_lr))
        print("classi: " + str(classes))
        print("encoder_name: " +str(cfg.model.encoder_name))
        model = unetSegmentationNet(classes=classes, lr=cfg.train.lr, epochs=cfg.train.max_epochs, 
                                    sgm_type = cfg.model.sgm_type, loss = loss_fun, sgm_threshold=cfg.model.sgm_threshold, 
                                    len_dataset = train_dataset.__len__(), batch_size = bs, max_lr=cfg.train.max_lr, 
                                    model_type="unet", encoder_name=cfg.model.encoder_name)
    else :
        print("The model type doesn't exist")
        return

    # training (automates everything)
    trainer = pl.Trainer(
        logger=loggers,
        callbacks=callbacks,
        accelerator=cfg.train.accelerator,
        devices=cfg.train.devices,
        log_every_n_steps=1,
        max_epochs=cfg.train.max_epochs,
    )
    trainer.fit(model, train_loader, val_loader)
    

if __name__ == "__main__":
    main()
