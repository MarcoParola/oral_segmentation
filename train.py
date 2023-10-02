import os
import hydra
import torch
import pytorch_lightning
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils.multiclass import unique_labels

from src.models.fcnSegmentation import FcnSegmentationNet
from src.datamodule import OralSegmentationDataModule

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
    model = FcnSegmentationNet(
        #model=cfg.model.name,
        #weights=cfg.model.weights,
        num_classes=cfg.model.num_classes,
        lr=cfg.train.lr,
    )

    # datasets and transformations
    train_img_tranform, val_img_tranform, test_img_tranform, img_tranform = get_transformations(cfg)
    data = OralSegmentationDataModule(
        train=cfg.dataset.train,
        val=cfg.dataset.val,
        test=cfg.dataset.test,
        batch_size=cfg.train.batch_size,
        train_transform = train_img_tranform,
        val_transform = val_img_tranform,
        test_transform = test_img_tranform,
        transform = img_tranform,
    )
    # TODO magari poi lo leviamo, ma non è banale fare augmentation con le maschere
    train_img_tranform, val_img_tranform, test_img_tranform = None, None, None

    # training
    trainer = pytorch_lightning.Trainer(
        logger=loggers,
        callbacks=callbacks,
        accelerator=cfg.train.accelerator,
        devices=cfg.train.devices,
        log_every_n_steps=1,
        max_epochs=cfg.train.max_epochs,
    )
    trainer.fit(model, data)


    # prediction
    predictions = trainer.predict(model, data)  
    #print('oooooooo',type(predictions[0]), len(predictions[0])) 
    predictions = predictions[0]
    test = data.test_dataloader()
    #print(type(test), len(test))
    test = next(iter(test))
    #print(type(test), type(test[0]), test[0].shape, test[1].shape, len(test))
    images, masks = test[0], test[1]
    for i in range(len(images)):
        image, mask = images[i], masks[i]
        #print(i, type(image), image.shape, type(mask), mask.shape)  
        plt.imshow(image.squeeze().permute(1,2,0))
        plt.imshow(mask.squeeze().detach().numpy(), alpha=0.4)
        plt.show()

        plt.imshow(image.squeeze().permute(1,2,0))
        print(i, 'aaaaaaaaaaa', predictions[i].shape, predictions[i], image)
        plt.imshow(predictions[i].squeeze().detach().permute(1,2,0).numpy(), alpha=0.7)
        plt.show()
    
    '''
    predictions = torch.cat(predictions, dim=0)
    predictions = torch.argmax(predictions, dim=1)
    gt = torch.cat([y for _, y in data.test_dataloader()], dim=0)

    print(classification_report(gt, predictions))

    class_names = np.array(['Class 0', 'Class 1', 'Class 2'])
    log_dir = 'logs/oral/' + get_last_version('logs/oral')
    log_confusion_matrix(gt, predictions, classes=class_names, log_dir=log_dir) 

    # save model
    model_path = os.path.join(cfg.train.save_path, cfg.model.name)
    os.makedirs(model_path, exist_ok=True)
    model_name = 'model' + '_' + str(len(os.listdir(model_path))) + '.pt'
    model_name = os.path.join(model_path, model_name)
    torch.save(model.state_dict(), model_name)
    '''

if __name__ == "__main__":
    main()