
import os
import hydra
import torch
import pytorch_lightning as pl
import numpy as np
import matplotlib.pyplot as plt
import argparse
from sklearn.utils.multiclass import unique_labels

from src.models.ensemble import ensembleSegmentationNet

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

    for root, dirs, files in os.walk("photo_ensemble"):
        for file in files:
            file_path = os.path.join(root, file)
            os.remove(file_path)
        for dir in dirs:
            dir_path = os.path.join(root, dir)
            os.rmdir(dir_path)

    print(f"dec_fus: {cfg.ensemble.dec_fus}")  
    print(f"type_aggr: {cfg.ensemble.type_aggr}")  
    
    if cfg.model.num_classes == 1:

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
                                        decision_fusion = cfg.ensemble.dec_fus,
                                        type_agg = cfg.ensemble.type_aggr)
    else:
        classes = cfg.model.num_classes + 1

        check_path_fcn = find_path(root_path = cfg.checkpoints.root_path, version = cfg.ensemble.check_fcn_mul)
        check_path_dl = find_path(root_path = cfg.checkpoints.root_path, version = cfg.ensemble.check_dl_mul)
        check_path_unet_eff = find_path(root_path = cfg.checkpoints.root_path, version = cfg.ensemble.check_unet_eff_mul)
        check_path_unet_res = find_path(root_path = cfg.checkpoints.root_path, version = cfg.ensemble.check_unet_res_mul)

        model = ensembleSegmentationNet(path_fcn = check_path_fcn, 
                                        path_deeplab = check_path_dl, 
                                        path_unet_eff = check_path_unet_eff, 
                                        path_unet_res=check_path_unet_res,  
                                        classes=classes, sgm_threshold=cfg.model.sgm_threshold, 
                                        decision_fusion = cfg.ensemble.dec_fus)

    model.eval()
    print(cfg.ensemble.dec_fus)
    train_img_tranform, val_img_tranform, test_img_tranform, img_tranform = get_transformations(cfg)

    if cfg.model.num_classes == 1:
        test_dataset = BinarySegmentationDataset(cfg.dataset.test, transform=test_img_tranform)
    else:
        test_dataset = MultiClassSegmentationDataset(cfg.dataset.test, transform=test_img_tranform)

    test_loader = DataLoader(test_dataset, batch_size=2, num_workers=11)


    # Evaluate the model on the test set
    trainer = pl.Trainer(
        logger=loggers,
        accelerator=cfg.train.accelerator,
        devices=cfg.train.devices,)
    trainer.test(model, test_loader)

    cartella_destinazione = f"photo_output/{cfg.ensemble.dec_fus}"
    if os.path.exists(cartella_destinazione):
        print(f"Name folder: {cartella_destinazione}")
        for root, dirs, files in os.walk(cartella_destinazione):
            for file in files:
                file_path = os.path.join(root, file)
                os.remove(file_path)
            for dir in dirs:
                dir_path = os.path.join(root, dir)
                os.rmdir(dir_path)
    else:
        # Se la cartella non esiste, creala
        os.makedirs(cartella_destinazione)
        print(f"Name folder: {cartella_destinazione}")

    count_img = 0; 

    print(cfg.model.sgm_threshold)
    for image, mask, cat_id in test_loader:
        # plot some segmentation predictions in a plot containing three subfigure: image - actual - predicted
        model = model.to('cpu')
        with torch.no_grad():
            output = model(image, cat_id)
        
        for i in range(image.size(0)):
            if cfg.model.num_classes == 1:

                output = (output > cfg.model.sgm_threshold).float()

                fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4))
                ax1.imshow(image[i].squeeze().permute(1,2,0))
                ax1.set_title('Image')
                ax2.imshow(image[i].squeeze().permute(1,2,0), alpha=0.5)
                ax2.set_title('Doctor segment')
                ax3.imshow(image[i].squeeze().permute(1,2,0), alpha=0.5)
                ax2.imshow(mask[i].squeeze(0).numpy(), alpha=0.6, cmap='gray')
                ax3.imshow(output[i].squeeze(0).detach().numpy(), alpha=0.6, cmap='gray')
                ax3.set_title('Mask predicted')

                ax1.set_xticks([])
                ax1.set_yticks([])
                ax2.set_xticks([])
                ax2.set_yticks([])
                ax3.set_xticks([])
                ax3.set_yticks([])
            else:

                indices = torch.argmax(output, dim=1)
                one_hot = torch.nn.functional.one_hot(indices, num_classes=4)
                one_hot = one_hot.permute(0, 3, 1, 2)
                output = one_hot

                fig, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(1, 6, figsize=(12, 4))
                ax1.imshow(image[i].squeeze().permute(1,2,0))
                ax1.set_title('Image')
                ax2.imshow(image[i].squeeze().permute(1,2,0), alpha=0.5)
                ax2.imshow(mask[i, cat_id[i].item(), :, :].squeeze(0).numpy(), alpha=0.6, cmap='gray')
                ax2.set_title('Doctor segment')
                ax3.imshow(image[i].squeeze().permute(1,2,0), alpha=0.5)
                ax3.imshow(output[i, 0, :, :].squeeze(0).detach().numpy(), alpha=0.6, cmap='gray')
                ax3.set_title('healthy tissue')
                ax4.imshow(image[i].squeeze().permute(1,2,0), alpha=0.5)
                ax4.imshow(output[i, 1, :, :].squeeze(0).detach().numpy(), alpha=0.6, cmap='gray')
                ax4.set_title('Mask cat 1')
                ax5.imshow(image[i].squeeze().permute(1,2,0), alpha=0.5)
                ax5.imshow(output[i, 2, :, :].squeeze(0).detach().numpy(), alpha=0.6, cmap='gray')
                ax5.set_title('Mask cat 2')
                ax6.imshow(image[i].squeeze().permute(1,2,0), alpha=0.5)
                ax6.imshow(output[i, 3, :, :].squeeze(0).detach().numpy(), alpha=0.6, cmap='gray')
                ax6.set_title('Mask cat 3')

                ax1.set_xticks([])
                ax1.set_yticks([])
                ax2.set_xticks([])
                ax2.set_yticks([])
                ax3.set_xticks([])
                ax3.set_yticks([])
                ax4.set_xticks([])
                ax4.set_yticks([])
                ax5.set_xticks([])
                ax5.set_yticks([])
                ax6.set_xticks([])
                ax6.set_yticks([])

                plt.suptitle(f"True category: {cat_id[i]}")
            
            nome_file = os.path.join(cartella_destinazione, f"immagine_{count_img}.png")
            count_img = count_img + 1
            plt.savefig(nome_file)

            plt.close(fig)
    
        
    

if __name__ == "__main__":
    main()
