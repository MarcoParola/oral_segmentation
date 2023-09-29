import os
import hydra
import torch
import argparse
import pytorch_lightning
#from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
#from sklearn.metrics import confusion_matrix
#from sklearn.utils.multiclass import unique_labels

from src.classifier import OralClassifierModule
from src.datamodule import OralClassificationDataModule

from src.utils import *


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg):

    
    loggers = get_loggers(cfg)
    callbacks = list()
    callbacks.append(get_early_stopping(cfg))
    
    model = OralClassifierModule(
        model=cfg.model.name,
        weights=cfg.model.weights,
        num_classes=cfg.model.num_classes,
    )
    model.load_state_dict(torch.load(os.path.join(cfg.train.save_path, cfg.load_model)))
    print(cfg.load_model)
    

    # datasets and transformations
    train_img_tranform, val_img_tranform, test_img_tranform, img_tranform = get_transformations(cfg)
    data = OralClassificationDataModule(
        train=cfg.dataset.train,
        val=cfg.dataset.val,
        test=cfg.dataset.test,
        batch_size=cfg.train.batch_size,
        train_transform = train_img_tranform,
        val_transform = val_img_tranform,
        test_transform = test_img_tranform,
        transform = img_tranform,
    )

    trainer = pytorch_lightning.Trainer(
        logger=loggers,
        callbacks=callbacks,
        accelerator=cfg.train.accelerator,
        devices=cfg.train.devices,
        log_every_n_steps=1,
        max_epochs=cfg.train.max_epochs,
    )

    # extract features from datamodule
    features = trainer.extract_features(model, data)
    print(features.shape)


    '''
    # prediction
    predictions = trainer.predict(model, data)   # TODO: inferenza su piu devices
    predictions = torch.cat(predictions, dim=0)
    predictions = torch.argmax(predictions, dim=1)
    gt = torch.cat([y for _, y in data.test_dataloader()], dim=0)

    print(classification_report(gt, predictions))

    class_names = np.array(['Class 0', 'Class 1', 'Class 2'])
    log_dir = 'logs/oral/' + get_last_version('logs/oral')
    log_confusion_matrix(gt, predictions, classes=class_names, log_dir=log_dir) # TODO cambia nome, perchè loggo anche acc

    # save model
    model_path = os.path.join(cfg.train.save_path, cfg.model.name)
    os.makedirs(model_path, exist_ok=True)
    model_name = 'model' + '_' + str(len(os.listdir(model_path))) + '.pt'
    model_name = os.path.join(model_path, model_name)
    torch.save(model.state_dict(), model_name)
    '''

if __name__ == "__main__":
    main()