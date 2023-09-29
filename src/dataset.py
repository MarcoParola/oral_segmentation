import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import os
import json
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image, ImageDraw


class OralSegmentationDataset(torch.utils.data.Dataset):
    def __init__(self, annonations, transform=None):
        self.annonations = annonations
        self.transform = transform

        with open(annonations, "r") as f:
            self.dataset = json.load(f)
        
        self.images = dict()
        for image in self.dataset["images"]:
            self.images[image["id"]] = image

        
    def __len__(self):
        return len(self.dataset["annotations"])


    def __getitem__(self, idx):
        annotation = self.dataset["annotations"][idx]
        image = self.images[annotation["image_id"]]
        image_path = os.path.join(os.path.dirname(self.annonations), "oral1", image["file_name"])
        image = Image.open(image_path).convert("RGB")

        width, height = image.size
        mask = Image.new('L', (width, height))
        segmentation = annotation["segmentation"]
        for i in range(len(segmentation)):
            ImageDraw.Draw(mask).polygon(segmentation[i], outline=1, fill=1)

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return image, mask


if __name__ == "__main__":
    import torchvision

    dataset = OralSegmentationDataset(
        "data/train.json",
        transform=transforms.Compose([
            transforms.Resize((224, 224), antialias=True),
            transforms.ToTensor()
        ])
    )

    for i in range(20):
        image, mask = dataset.__getitem__(i)
        print(image.shape, mask.shape)
        plt.imshow(image[0])
        plt.imshow(mask[0], alpha=0.2)
        plt.show()
        