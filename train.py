# Per eseguire segmentazione binaria: python train.py 
# Per eseguire segmentazione multiclasse aggiungere: model.num_classes=3

import os
import hydra
import torch
import pytorch_lightning as pl
import numpy as np
import matplotlib.pyplot as plt
import argparse
from sklearn.utils.multiclass import unique_labels

from src.models.fcn import FcnSegmentationNet
from src.models.deeplab import DeeplabSegmentationNet
from src.models.deeplabFE import ModelFE
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


    # datasets and dataloaders
    train_img_tranform, val_img_tranform, test_img_tranform, img_tranform = get_transformations(cfg)
    train_dataset = OralSegmentationDataset(cfg.dataset.train, transform=img_tranform)
    val_dataset = OralSegmentationDataset(cfg.dataset.val, transform=img_tranform)
    test_dataset = OralSegmentationDataset(cfg.dataset.test, transform=img_tranform)  
    train_loader = DataLoader(train_dataset, batch_size=cfg.train.batch_size, shuffle=True, num_workers=11)
    val_loader = DataLoader(val_dataset, batch_size=cfg.train.batch_size, num_workers=11)
    test_loader = DataLoader(test_dataset, batch_size=cfg.train.batch_size, num_workers=11)

    # model
    model = DeeplabSegmentationNet(num_classes=cfg.model.num_classes, lr=cfg.train.lr, epochs=cfg.train.max_epochs, sgm_type = cfg.model.sgm_type, sgm_threshold=cfg.model.sgm_threshold, len_dataset = train_dataset.__len__(), batch_size = cfg.train.batch_size)
    #model = FcnSegmentationNet(num_classes=cfg.model.num_classes, lr=cfg.train.lr, epochs=cfg.train.max_epochs, sgm_type = cfg.model.sgm_type, sgm_threshold=cfg.model.sgm_threshold, len_dataset = train_dataset.__len__(), batch_size = cfg.train.batch_size) 


    # training
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
