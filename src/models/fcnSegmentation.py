
import hydra
import torch
import numpy as np
import torch.nn as nn
from torchvision import models
from pytorch_lightning import LightningModule
#from torchvision.models.segmentation.deeplabv3 import DeepLabHead


# Define the neural network architecture
class FcnSegmentationNet(LightningModule):
    def __init__(self, num_classes, lr=10e-4):
        super(FcnSegmentationNet, self).__init__()
        self.pretrained_model = models.segmentation.fcn_resnet50(pretrained=True)
        self.pretrained_model.classifier[4] = nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1))
        # add sigmoid activation
        self.last_layer = nn.Sigmoid()
        self.lr = lr
        self.loss = nn.MSELoss()

    def forward(self, x):
        out = self.pretrained_model(x)['out']
        out = self.last_layer(out)
        return out

    def predict_step(self, batch, batch_idx):
        return self(batch[0])

    def training_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        self._common_step(batch, batch_idx, "val") 
        
    def test_step(self, batch, batch_idx):
        self._common_step(batch, batch_idx, "test")
        output = self(batch)
        accuracy = accuracy_score(output, batch['target'])
        self.log('test_accuracy', accuracy)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def _common_step(self, batch, batch_idx, stage):
        img, actual_mask = batch
        x = self.pretrained_model(img)['out']
        x = self.last_layer(x)
        mask_predicted = x
        #mask_predicted = self(x)
        loss = self.loss(mask_predicted, actual_mask)
        self.log(f"{stage}_loss", loss, on_step=True)
        return loss


'''
class DeeplabSegmentationNet(nn.Module):
    def __init__(self, num_classes):
        super(DeeplabSegmentationNet, self).__init__()
        self.pretrained_model = models.segmentation.deeplabv3_resnet101(pretrained=True, progress=True)
        self.pretrained_model.classifier = DeepLabHead(2048, num_classes)

    def forward(self, x):
        out = self.pretrained_model(x)['out']
        return out
'''




if __name__ == "__main__":
    import torchvision.transforms as transforms
    from src.dataset import OralSegmentationDataset
    from matplotlib import pyplot as plt

    dataset = OralSegmentationDataset(
        "data/train.json",
        transform=transforms.Compose([
            transforms.Resize((224, 224), antialias=True),
            transforms.ToTensor()
        ])
    )

    fcn = FcnSegmentationNet(1)

    for i in range(20):
        image, mask = dataset.__getitem__(i)
        # expand batch dimension, add one more dimension corresponding to number of images
        image = image.unsqueeze(0)
        print(image.shape, mask.shape)
        pred = fcn(image)
        plt.imshow(image.squeeze().permute(1,2,0))
        plt.imshow(pred.squeeze().detach().numpy(), alpha=0.4)
        plt.show()