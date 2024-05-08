import os
import hydra
import torch
import pytorch_lightning as pl
import numpy as np
import matplotlib.pyplot as plt
import argparse
from sklearn.utils.multiclass import unique_labels

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

    # to visualize log on tensorboard
    loggers = get_loggers(cfg)

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

    files = os.listdir(path_checkpoint)
    print(os.path.join(path_checkpoint, files[0]))
    check_path = os.path.join(path_checkpoint, files[0])
    checkpoint = torch.load(check_path)
    print(checkpoint["hyper_parameters"]) 

    hyper_parameters = checkpoint["hyper_parameters"]

    # extract hyperparameters
    model_type = hyper_parameters["model_type"]
    classes = hyper_parameters["classes"]
    print(f"num classi: {classes}")

    model = get_model(hyper_parameters = hyper_parameters, model_type = model_type, check_path = check_path, sgm_threshold = cfg.model.sgm_threshold, num_classes=classes, version_number=version_number)
    if(model == False):
        return

    # disable randomness, dropout, etc...
    model.eval()

    # datasets and dataloaders
    train_img_tranform, val_img_tranform, test_img_tranform, img_tranform = get_transformations(cfg)

    if classes == 1:
        test_dataset = BinarySegmentationDataset(cfg.dataset.test, transform=test_img_tranform)
    else:
        test_dataset = MultiClassSegmentationDataset(cfg.dataset.test, transform=test_img_tranform, n_classes = classes - 1)
    if model_type == "deeplab":
        test_loader = DataLoader(test_dataset, batch_size=2, num_workers = cfg.train.num_workers)
    else:
        test_loader = DataLoader(test_dataset, batch_size=1, num_workers = cfg.train.num_workers)

    # Evaluate the model on the test set
    trainer = pl.Trainer(
        logger=loggers,
        accelerator=cfg.train.accelerator,
        devices=cfg.train.devices,)
    trainer.test(model, test_loader)
    
    cartella_destinazione = f"{cfg.test.save_output_path}/version_{version_number}"

    # Check if the folder exist 
    if os.path.exists(cartella_destinazione):
        for root, dirs, files in os.walk(cartella_destinazione):
            for file in files:
                file_path = os.path.join(root, file)
                os.remove(file_path)
            for dir in dirs:
                dir_path = os.path.join(root, dir)
                os.rmdir(dir_path)
    else:
        os.makedirs(cartella_destinazione)
        print(f"Name folder: {cartella_destinazione}")

    count_img = 0
    
    for image, mask, cat_id in test_loader:
        # plot some segmentation predictions in a plot containing three subfigure: image - actual - predicted
        model = model.to('cpu')
        with torch.no_grad():
            output = model(image)

        if classes>1:
            output = torch.nn.functional.softmax(output, dim = 1)
            indices = torch.argmax(output, dim=1)
            one_hot = torch.nn.functional.one_hot(indices, num_classes=4)
            one_hot = one_hot.permute(0, 3, 1, 2)
            output = one_hot
        else:
            output = torch.sigmoid(output)
            out_net = output.clone()
            output = (output > cfg.model.sgm_threshold).float()
            
        for i in range(image.size(0)):
            if classes == 1:
                
                fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4))
                ax1.imshow(image[i].squeeze().permute(1,2,0))
                ax1.set_title("Image")
                ax2.imshow(image[i].squeeze().permute(1,2,0), alpha=0.5)
                ax3.imshow(image[i].squeeze().permute(1,2,0), alpha=0.5)
                ax2.imshow(mask[i].squeeze(0).numpy(), alpha=0.6, cmap='gray')
                ax2.set_title("Ground true")
                ax3.imshow(output[i].squeeze(0).detach().numpy(), alpha=0.6, cmap='gray')
                ax3.set_title("Predicted mask")

                ax1.set_xticks([])
                ax1.set_yticks([])
                ax2.set_xticks([])
                ax2.set_yticks([])
                ax3.set_xticks([])
                ax3.set_yticks([])
                
            else:            
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
            
            # Save the figure into file
            nome_file = os.path.join(cartella_destinazione, f"immagine_{count_img}.png")
            count_img = count_img + 1

            plt.savefig(nome_file)

            plt.close(fig)
        

            


if __name__ == "__main__":
    main()