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
from src.models.ensemble import ensembleSegmentationNet

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models

from src.utils import *

@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg):

    for root, dirs, files in os.walk("DatiROC"):
        for file in files:
            file_path = os.path.join(root, file)
            os.remove(file_path)
        for dir in dirs:
            dir_path = os.path.join(root, dir)
            os.rmdir(dir_path) 

    print(f"dec_fus: {cfg.ensemble.dec_fus}")  

    check_path_fcn = find_path(root_path = cfg.checkpoints.root_path, version = cfg.ensemble.check_fcn_bin)
    check_path_dl = find_path(root_path = cfg.checkpoints.root_path, version = cfg.ensemble.check_dl_bin)
    check_path_unet_eff = find_path(root_path = cfg.checkpoints.root_path, version = cfg.ensemble.check_unet_eff_bin)
    check_path_unet_res = find_path(root_path = cfg.checkpoints.root_path, version = cfg.ensemble.check_unet_res_bin)

    model = ensembleSegmentationNet(path_fcn = check_path_fcn, 
                                    path_deeplab = check_path_dl, 
                                    path_unet_eff = check_path_unet_eff, 
                                    path_unet_res = check_path_unet_res, 
                                    classes=cfg.model.num_classes, 
                                    sgm_threshold=cfg.model.sgm_threshold, 
                                    decision_fusion = cfg.ensemble.dec_fus)

    if(model == False):
        return

    model.eval()
    model = model.to('cpu')
    train_img_tranform, val_img_tranform, test_img_tranform, img_tranform = get_transformations(cfg)
    test_dataset = BinarySegmentationDataset(cfg.dataset.test, transform=test_img_tranform)
    
    test_loader = DataLoader(test_dataset, batch_size=1, num_workers=cfg.train.num_workers)
    
    thresholds = np.concatenate(
        [np.arange(0, 0.02, 0.01), 
        np.arange(0.02, 0.12, 0.05), 
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
        
        for i, (image, mask, cat_id) in enumerate(test_loader):
            
            image = image.to('cpu')
            mask = mask.to('cpu')
            with torch.no_grad():
                output = model(image, cat_id = 0)
            
            output = output > threshold            
        
            tp += ((output == 1) & (mask == 1)).sum()
            #tn += ((output == 0) & (mask == 0)).sum()
            fp += ((output == 1) & (mask == 0)).sum()
            fn += ((output == 0) & (mask == 1)).sum()

        #print(f"tp: {tp}, tn: {tn}, fp: {fp}, fn: {fn}")
        precision = (tp ) / (tp + fp + eps)
        recall = (tp ) / (tp + fn + eps)
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


    # Dice plot
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, dices, label='Dice Curve')
    plt.scatter(thresholds[ix_dice], dices[ix_dice], marker='o', color='black', label='Best')
    plt.text(thresholds[ix_dice], dices[ix_dice], f"Best Threshold={thresholds[ix_dice]}")
    plt.title('Curva Dice Score vs Threshold')
    plt.xlabel('Threshold')
    plt.ylabel('Dice Score')
    plt.grid(True)
    plt.savefig(f"DatiROC/dice_threshold.png", bbox_inches='tight')
    plt.close()

    print(f"Best Threshold={thresholds[ix_dice]}, Dice={dices[ix_dice]}")
    
    # Precision and recall plot
    plt.figure()
    plt.plot(precisions, recalls, color="red", lw=3, label='Precision-Recall Curve')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('Precision')
    plt.ylabel('Recall')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")

    plt.scatter(precisions[ix_gmeans], recalls[ix_gmeans], marker='o', color='black', label='Best')
    plt.text(precisions[ix_gmeans], recalls[ix_gmeans], thresholds[ix_gmeans])
    print(f"Best Threshold={thresholds[ix_gmeans]}, G-Mean={gmeans[ix_gmeans]}")
    
    plt.savefig(f"DatiROC/roc_curve.png", bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    main()