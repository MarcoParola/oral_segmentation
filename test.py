# Per eseguire soft segmentation: python test.py
# Per eseguire hard segmentation: python test.py model.sgm_type=hard

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
from src.dataset import OralSegmentationDataset
from torch.utils.data import DataLoader

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models

from src.utils import *

@hydra.main(version_base=None, config_path="./config", config_name="config")
def main(cfg):

    # load a LightningModule
    checkpoint = torch.load(cfg.checkpoints.path, map_location=lambda storage, loc: storage)
    model = DeeplabSegmentationNet.load_from_checkpoint(cfg.checkpoints.path)

    # disable randomness, dropout, etc...
    model.eval()

    # datasets and dataloaders
    train_img_tranform, val_img_tranform, test_img_tranform, img_tranform = get_transformations(cfg)
    test_dataset = OralSegmentationDataset(cfg.dataset.test, transform=img_tranform)
    test_loader = DataLoader(test_dataset, batch_size=cfg.train.batch_size, num_workers=11)

    # Evaluate the model on the test set
    trainer = pl.Trainer()
    trainer.test(model, test_loader)

    """
    # plot some segmentation predictions in a plot containing three subfigure: image - actual - predicted
    images, masks = next(iter(test_loader))
    #images = images.to('cuda')
    model = model.to('cpu')
    outputs = model(images)
    print(cfg.model.sgm_type)
    if cfg.model.sgm_type == "hard":
        outputs = (outputs > 0.5).float()

    cartella_destinazione = "photo_output"

    # Verifica se la cartella esiste
    if os.path.exists(cartella_destinazione):
        # Se la cartella esiste, svuotala
        for root, dirs, files in os.walk(cartella_destinazione):
            for file in files:
                file_path = os.path.join(root, file)
                os.remove(file_path)
            for dir in dirs:
                dir_path = os.path.join(root, dir)
                os.rmdir(dir_path)
        print(f"Cartella '{cartella_destinazione}' svuotata con successo.")
    else:
        # Se la cartella non esiste, creala
        os.makedirs(cartella_destinazione)
        print(f"Cartella '{cartella_destinazione}' creata con successo.")

    for i in range(20):
        image, mask = images[i], masks[i]
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4))
        ax1.imshow(image.squeeze().permute(1,2,0))
        ax2.imshow(image.squeeze().permute(1,2,0), alpha=0.5)
        ax3.imshow(image.squeeze().permute(1,2,0), alpha=0.5)
        ax2.imshow(mask.permute(1,2,0).numpy(), alpha=0.6, cmap='gray')
        ax3.imshow(outputs[i].detach().permute(1,2,0).numpy(), alpha=0.6, cmap='gray')
        print(outputs[i].shape, outputs[i].max(), outputs[i].min())
        #plt.show()
        # Salva la figura come immagine in un file
        nome_file = os.path.join(cartella_destinazione, f"immagine_{i}.png")
        plt.savefig(nome_file)

        # Chiudi la figura dopo aver salvato l'immagine
        plt.close(fig)
    """

if __name__ == "__main__":
    main()