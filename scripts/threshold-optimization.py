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
from src.models.unet import unetSegmentationNet


from src.models.deeplabFE import ModelFE
from src.dataset import OralSegmentationDataset
from torch.utils.data import DataLoader

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models

from src.utils import *

@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg):
    if (cfg.checkpoints.version == "last"):
        folder_checkpoint, version_number = get_last_version(cfg.checkpoints.root_path)
    else:
        folder_checkpoint = "version_" + str(cfg.checkpoints.version)
        version_number = cfg.checkpoints.version
    
    path_checkpoint = cfg.checkpoints.root_path + "/" + folder_checkpoint + "/checkpoints"

    # check if the forder exists 
    if not os.path.exists(path_checkpoint):
        print(f"Version {cfg.checkpoints.version} doesn't exist.")
        return None

    # Check output folder
    folder_path = f"DatiROC/version_{version_number}"

    # Assicurati che la cartella esista
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    files = os.listdir(path_checkpoint)
    print(os.path.join(path_checkpoint, files[0]))
    check_path = os.path.join(path_checkpoint, files[0])
    checkpoint = torch.load(check_path)
    print(checkpoint["hyper_parameters"])

    hyper_parameters = checkpoint["hyper_parameters"]

    # extract hyperparameters
    model_type = hyper_parameters["model_type"]

    model = get_model(cfg, hyper_parameters, model_type, check_path, num_classes=cfg.model.num_classes)

    if(model == False):
        return

    model.eval()
    model = model.to('cpu')
    train_img_tranform, val_img_tranform, test_img_tranform, img_tranform = get_transformations(cfg)
    test_dataset = OralSegmentationDataset(cfg.dataset.test, transform=test_img_tranform)
    test_loader = DataLoader(test_dataset, batch_size=1, num_workers=11)

    thresholds = np.arange(0, 1, 0.02)

    precisions = []
    recalls = []
    
    for threshold in thresholds:
        print(threshold)

        tp = 0
        tn = 0
        fp = 0
        fn = 0

        for i, (image, mask) in enumerate(test_loader):
            
            image = image.to('cpu')
            mask = mask.to('cpu')
            output = model(image)
            output = output > threshold
            tp += ((output == 1) & (mask == 1)).sum()
            tn += ((output == 0) & (mask == 0)).sum()
            fp += ((output == 1) & (mask == 0)).sum()
            fn += ((output == 0) & (mask == 1)).sum()

        print(f"tp: {tp}, tn: {tn}, fp: {fp}, fn: {fn}")
        # compute precision and recall
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        precisions.append(precision)
        recalls.append(recall)

    # compute Geometric Mean (G-Mean)
    gmeans = np.sqrt(precision * (1-recall))
    ix = np.argmax(gmeans)

    # Plot della curva ROC
    plt.figure()
    plt.plot(fprs, tprs, color="red", lw=3)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    for i in range(len(thresholds)):
        plt.scatter(precisions[i], recalls[i], color='red')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('Precision')
    plt.ylabel('Recall')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")

    # marker per il punto ottimale
    plt.scatter(precisions[ix], recalls[ix], marker='o', color='black', label='Best')
    plt.text(precisions[ix], recalls[ix], thresholds[ix])
    print(f"Best Threshold={thresholds[ix]}, G-Mean={gmeans[ix]}")
    # Salvare l'immagine
    plt.savefig(f"DatiROC/version_{version_number}/roc_curve.png", bbox_inches='tight')
    plt.close()

    '''
    #plot di dice in base alla threshold
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, dices, marker='o', linestyle='-', color='b')
    plt.title('Curva Dice Score vs Threshold')
    plt.xlabel('Threshold')
    plt.ylabel('Dice Score')
    plt.grid(True)
    plt.savefig(f"DatiROC/version_{version_number}/dice_threshold.png", bbox_inches='tight')
    plt.close()
    '''



if __name__ == "__main__":
    main()