import os
import hydra
import torch
import pytorch_lightning as pl
import numpy as np
import matplotlib.pyplot as plt
import argparse
from sklearn.utils.multiclass import unique_labels

from src.datasets import BinarySegmentationDataset
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
    sgm_threshold = hyper_parameters["sgm_threshold"]
    n_classes = hyper_parameters["classes"]

    model = get_model(hyper_parameters = hyper_parameters, model_type = model_type, check_path = check_path, sgm_threshold = sgm_threshold,num_classes=n_classes)

    if(model == False):
        return

    model.eval()
    model = model.to('cpu')
    train_img_tranform, val_img_tranform, test_img_tranform, img_tranform = get_transformations(cfg)
    test_dataset = BinarySegmentationDataset(cfg.dataset.test, transform=test_img_tranform)
    
    test_loader = DataLoader(test_dataset, batch_size=1, num_workers=cfg.train.num_workers)

    # define an non-omonogeneous threshold list
    thresholds = np.concatenate(
        [np.arange(0, 0.05, 0.01), 
        np.arange(0.05, 0.12, 0.05), 
        np.arange(0.12, 0.15, 0.02),
        np.arange(0.15, 0.9, 0.06),
        np.arange(0.9, 1.01, 0.01)]
    )

    eps=1e-5

    precisions = []
    recalls = []
    dices = []

    
    for threshold in thresholds:
        print(threshold)

        tp = 0
        tn = 0
        fp = 0
        fn = 0
        count_img = 0
        for i, (image, mask, cat_id) in enumerate(test_loader):
            
            image = image.to('cpu')
            mask = mask.to('cpu')
            with torch.no_grad():
                output = model(image) 

            pixel_pre = output.permute(0, 2, 3, 1)  # reorder from NCHW to NHWC
            pixel_pre = pixel_pre.numpy()  
            pixel_values_pre = pixel_pre.flatten()
            output = torch.sigmoid(output)
            
            pixel = output.permute(0, 2, 3, 1)  # reorder from NCHW to NHWC
            pixel = pixel.numpy()  
            pixel_values = pixel.flatten()    
            
            output = output > threshold
        
            tp += ((output == 1) & (mask == 1)).sum()
            #tn += ((output == 0) & (mask == 0)).sum()
            fp += ((output == 1) & (mask == 0)).sum()
            fn += ((output == 0) & (mask == 1)).sum()
            
        #print(f"tp: {tp}, tn: {tn}, fp: {fp}, fn: {fn}")
        precision = (tp + eps) / (tp + fp + eps)
        recall = (tp + eps) / (tp + fn + eps)
        dice = 2 * tp / (2 * tp + fp + fn)
        print(f"DICE: {dice}, precision: {precision}, recall: {recall}")
        precisions.append(precision)
        recalls.append(recall)
        dices.append(dice)

    # compute Geometric Mean (G-Mean)
    precisions = np.array(precisions)
    recalls = np.array(recalls)
    gmeans = np.sqrt(precisions * recalls)
    ix_gmeans = np.argmax(gmeans)
    ix_dice = np.argmax(dices)



    '''
    #plot di dice in base alla threshold
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, dices, label='Dice Curve')
    plt.scatter(thresholds[ix_dice], dices[ix_dice], marker='o', color='black', label='Best')
    plt.text(thresholds[ix_dice], dices[ix_dice], f"Best Threshold={thresholds[ix_dice]}")
    plt.title('Curva Dice Score vs Threshold')
    plt.xlabel('Threshold')
    plt.ylabel('Dice Score')
    plt.grid(True)
    plt.savefig(f"DatiROC/version_{version_number}/dice_threshold.png", bbox_inches='tight')
    plt.close()
    '''

    print(f"Best Threshold={thresholds[ix_dice]}, Dice={dices[ix_dice]}")
    
    # Plot di precision e recall in base alla threshold
    plt.figure()
    plt.plot(precisions, recalls, color="red", lw=3, label='Precision-Recall Curve')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('Precision')
    plt.ylabel('Recall')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")

    # marker per il punto ottimale
    plt.scatter(precisions[ix_gmeans], recalls[ix_gmeans], marker='o', color='black', label='Best')
    plt.text(precisions[ix_gmeans], recalls[ix_gmeans], thresholds[ix_gmeans])
    print(f"Best Threshold={thresholds[ix_gmeans]}, G-Mean={gmeans[ix_gmeans]}")
    # Salvare l'immagine
    plt.savefig(f"DatiROC/version_{version_number}/roc_curve.png", bbox_inches='tight')
    plt.close()
    

if __name__ == "__main__":
    main()