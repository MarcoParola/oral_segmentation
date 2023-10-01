
import hydra
import torch
import numpy as np
import torch.nn as nn
from torchvision import models
from matplotlib import pyplot as plt
from ..dataset import OralSegmentationDataset
from torchvision.models.segmentation.deeplabv3 import DeepLabHead


# Define the neural network architecture
class FcnSegmentationNet(nn.Module):
    def __init__(self, num_classes):
        super(FcnSegmentationNet, self).__init__()
        self.pretrained_model = models.segmentation.fcn_resnet50(pretrained=True)
        self.pretrained_model.classifier[4] = nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x):
        out = self.pretrained_model(x)['out']
        return out


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