import os
import re
import hydra
import torchvision
import torch
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from sklearn.metrics import mean_squared_error
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

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
        model = FcnSegmentationNet.load_from_checkpoint(check_path, num_classes = num_classes, sgm_threshold = sgm_threshold)
    elif (model_type=="deeplab"):
        model = DeeplabSegmentationNet.load_from_checkpoint(check_path, num_classes = num_classes, sgm_threshold = sgm_threshold)
    elif( model_type=="unet"):
        encoder_name = hyper_parameters["encoder_name"]
        if(encoder_name == "efficientnet-b7"):
            model = unetSegmentationNet.load_from_checkpoint(check_path, num_classes = num_classes, sgm_threshold = sgm_threshold, encoder_name="efficientnet-b7")
        elif (encoder_name == "resnet50"):
            model = unetSegmentationNet.load_from_checkpoint(check_path, num_classes = num_classes, sgm_threshold = sgm_threshold, encoder_name="resnet50")
    else:
        print("Errore tipo di rete non trattata nel test")
        model = False

    return model

def find_path(root_path, version):
    folder_checkpoint = "version_" + str(version)
    path_checkpoint = root_path + "/" + folder_checkpoint + "/checkpoints"

    # check if the forder exists 
    if not os.path.exists(path_checkpoint):
        print(f"Version {version} doesn't exist.")
        return

    files = os.listdir(path_checkpoint)
    print(os.path.join(path_checkpoint, files[0]))
    check_path = os.path.join(path_checkpoint, files[0])

    return check_path

def load_model_from_checkpoint(path, model_type):
    
    checkpoint = torch.load(path)
    print(checkpoint["hyper_parameters"])

    hyper_parameters = checkpoint["hyper_parameters"]

    hp_model = hyper_parameters["model_type"]
    classes = hyper_parameters["classes"] 
    sgm_threshold = hyper_parameters["sgm_threshold"]

    if (hp_model!=model_type):
        print(f"Error in loading {model_type} checkpoint")
        model = False
    else:
        model = get_model(hyper_parameters = hyper_parameters, model_type = model_type, check_path = path, sgm_threshold = sgm_threshold, num_classes=classes)

    return model

def plot_all_results(img, fcn, deeplab, unet_eff, unet_res, ensemble, dec_fun, cat_id):

    #print(img.shape) #torch.Size([2, 3, 448, 448])
    #print(fcn.shape) #torch.Size([2, 1, 448, 448])
    #print(deeplab.shape) #torch.Size([2, 1, 448, 448])
    #print(unet.shape) #torch.Size([2, 1, 448, 448])
    #print(ensemble.shape) #torch.Size([2, 1, 448, 448])
    img = img.to("cpu")
    fcn = fcn.to("cpu")
    deeplab = deeplab.to("cpu")
    unet_eff = unet_eff.to("cpu")
    unet_res = unet_res.to("cpu")
    ensemble = ensemble.to("cpu")

    if fcn.size(1) == 1:
        for i in range(img.size(0)):
            fig, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(1, 6, figsize=(12, 4))
            ax1.imshow(img[i].squeeze().permute(1,2,0))
            ax1.set_title('Real image')
            ax2.imshow(img[i].squeeze().permute(1,2,0), alpha=0.5)
            ax2.imshow(fcn[i].squeeze(0).numpy(), alpha=0.6, cmap='gray')
            ax2.set_title('Fcn prediction')
            ax3.imshow(img[i].squeeze().permute(1,2,0), alpha=0.5)
            ax3.imshow(deeplab[i].squeeze(0).detach().numpy(), alpha=0.6, cmap='gray')
            ax3.set_title('Deeplab prediction')
            ax4.imshow(img[i].squeeze().permute(1,2,0), alpha=0.5)
            ax4.imshow(unet_eff[i].squeeze(0).detach().numpy(), alpha=0.6, cmap='gray')
            ax4.set_title('Unet efficientnet-b7 prediction')
            ax5.imshow(img[i].squeeze().permute(1,2,0), alpha=0.5)
            ax5.imshow(unet_res[i].squeeze(0).detach().numpy(), alpha=0.6, cmap='gray')
            ax5.set_title('Unet Resnet50 prediction')
            ax6.imshow(img[i].squeeze().permute(1,2,0), alpha=0.5)
            ax6.imshow(ensemble[i].squeeze(0).detach().numpy(), alpha=0.6, cmap='gray')
            ax6.set_title(dec_fun)

            directory = "photo_ensemble"
            files = os.listdir(directory)

            # Salva la figura come immagine in un file
            nome_file = os.path.join(directory, f"img_{len(files)}.png")
            plt.savefig(nome_file)

            # Chiudi la figura dopo aver salvato l'immagine
            plt.close(fig)
    else:
        # print(fcn.size(1)) = 4
        for i in range(img.size(0)):
            fig, axs = plt.subplots(fcn.size(1), 6, figsize=(20, 10))
            for j in range(fcn.size(1)):
                axs[j, 0].imshow(img[i].squeeze().permute(1,2,0))
                axs[j, 0].set_title('Real image')
                axs[j, 0].set_xticks([])
                axs[j, 0].set_yticks([])
                axs[j, 1].imshow(img[i].squeeze().permute(1,2,0), alpha=0.5)
                axs[j, 1].imshow(fcn[i, j, :, :].squeeze(0).numpy(), alpha=0.6, cmap='gray')
                axs[j, 1].set_title(f"Fcn prediction class {j}")
                axs[j, 1].set_xticks([])
                axs[j, 1].set_yticks([])
                axs[j, 2].imshow(img[i].squeeze().permute(1,2,0), alpha=0.5)
                axs[j, 2].imshow(deeplab[i, j, :, :].squeeze(0).detach().numpy(), alpha=0.6, cmap='gray')
                axs[j, 2].set_title(f"Deeplab prediction class {j}")
                axs[j, 2].set_xticks([])
                axs[j, 2].set_yticks([])
                axs[j, 3].imshow(img[i].squeeze().permute(1,2,0), alpha=0.5)
                axs[j, 3].imshow(unet_eff[i, j, :, :].squeeze(0).detach().numpy(), alpha=0.6, cmap='gray')
                axs[j, 3].set_title(f"Unet_eff prediction class {j}")
                axs[j, 3].set_xticks([])
                axs[j, 3].set_yticks([])
                axs[j, 4].imshow(img[i].squeeze().permute(1,2,0), alpha=0.5)
                axs[j, 4].imshow(unet_res[i, j, :, :].squeeze(0).detach().numpy(), alpha=0.6, cmap='gray')
                axs[j, 4].set_title(f"Unet_res prediction class {j}")
                axs[j, 4].set_xticks([])
                axs[j, 4].set_yticks([])
                axs[j, 5].imshow(img[i].squeeze().permute(1,2,0), alpha=0.5)
                axs[j, 5].imshow(ensemble[i, j, :, :].squeeze(0).detach().numpy(), alpha=0.6, cmap='gray')
                axs[j, 5].set_title(f"{dec_fun} class {j}")
                axs[j, 5].set_xticks([])
                axs[j, 5].set_yticks([])
                
            fig.suptitle(f"True class: {cat_id[i]}", fontsize=16)
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Ajust the subplots to fit the suptitle
              
            directory = "photo_ensemble"
            files = os.listdir(directory)
            # Salva la figura come immagine in un file
            nome_file = os.path.join(directory, f"img_{len(files)}.png")
            plt.savefig(nome_file)

            # Chiudi la figura dopo aver salvato l'immagine
            plt.close(fig)



