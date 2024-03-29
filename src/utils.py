import os
import re
import hydra
import torchvision
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from sklearn.metrics import mean_squared_error
from torch.utils.tensorboard import SummaryWriter

from src.models.fcn import FcnSegmentationNet
from src.models.deeplab import DeeplabSegmentationNet
from src.models.unet import unetSegmentationNet


def get_loggers(cfg):
    """Returns a list of loggers
    cfg: hydra config
    """
    loggers = list()
    if cfg.log.wandb:
        from pytorch_lightning.loggers import WandbLogger
        import wandb
        hyperparameters = hp_from_cfg(cfg)
        wandb.init(entity=cfg.wandb.entity, project=cfg.wandb.project)
        wandb.config.update(hyperparameters)
        wandb_logger = WandbLogger()
        loggers.append(wandb_logger)
    
    if cfg.log.tensorboard:
        from pytorch_lightning.loggers import TensorBoardLogger
        tensorboard_logger = TensorBoardLogger(cfg.log.path , name="oral")
        loggers.append(tensorboard_logger)

    return loggers



def get_early_stopping(cfg):
    """Returns an EarlyStopping callback
    cfg: hydra config
    """
    early_stopping_callback = EarlyStopping(
        monitor='val_loss',
        mode='min',
        patience=30,
    )
    return early_stopping_callback



def get_transformations(cfg):
    """Returns the transformations for the dataset
    cfg: hydra config
    """
    img_tranform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(cfg.dataset.resize, antialias=True),
        torchvision.transforms.CenterCrop(cfg.dataset.resize),
        torchvision.transforms.ToTensor(),
    ])
    val_img_tranform, test_img_tranform = img_tranform, img_tranform

    train_img_tranform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(cfg.dataset.resize, antialias=True),
        torchvision.transforms.CenterCrop(cfg.dataset.resize),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.RandomHorizontalFlip(p=0.5),
        torchvision.transforms.RandomVerticalFlip(p=0.5),
        torchvision.transforms.RandomRotation(degrees=45),
        #torchvision.transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        torchvision.transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.8, 1.2), shear=0),
    ])
    return train_img_tranform, val_img_tranform, test_img_tranform, img_tranform
    


def log_metrics(actual, predicted, classes, log_dir):
    """Logs the confusion matrix to tensorboard
    actual: ground truth
    predicted: predictions
    classes: list of classes
    log_dir: path to the log directory
    """
    writer = SummaryWriter(log_dir=log_dir)

    # log metrics on tensorboard
    writer.add_scalar('mse', mean_squared_error(actual, predicted))
    writer.close()



def get_last_version(path):
    """Return the last version of the folder in path
    path: path to the folder containing the versions
    """
    folders = os.listdir(path)
    # get the folders starting with 'version_'
    folders = [f for f in folders if re.match(r'version_[0-9]+', f)]
    # get the last folder with the highest number
    last_folder = max(folders, key=lambda f: int(f.split('_')[1]))
    version_number = int(last_folder.split('_')[1])
    return last_folder, version_number

def get_model(hyper_parameters, model_type, check_path, sgm_threshold, num_classes):

    if(model_type=="fcn"):
        print("test su fcn")
        model = FcnSegmentationNet.load_from_checkpoint(check_path, num_classes = num_classes, sgm_threshold = sgm_threshold)
        print("Modello caricato")
    elif (model_type=="deeplab"):
        print("test su deeplab")
        model = DeeplabSegmentationNet.load_from_checkpoint(check_path, num_classes = num_classes, sgm_threshold = sgm_threshold)
        print("Modello caricato")
    elif( model_type=="unet"):
        encoder_name = hyper_parameters["encoder_name"]
        if(encoder_name == "efficientnet-b7"):
            print("test su unet_efficientnet-b7")
            model = unetSegmentationNet.load_from_checkpoint(check_path, num_classes = num_classes, sgm_threshold = sgm_threshold, encoder_name="efficientnet-b7")
            print("Modello caricato")
        elif (encoder_name == "resnet50"):
            print("test su unet_resnet50")
            model = unetSegmentationNet.load_from_checkpoint(check_path, num_classes = num_classes, sgm_threshold = sgm_threshold, encoder_name="resnet50")
            print("Modello caricato")
    else:
        print("Errore  tipo di rete non trattata nel test")
        model = False

    return model