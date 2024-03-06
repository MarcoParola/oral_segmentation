import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import os
import json
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image, ImageDraw
import hydra


class MultiClassSegmentationDataset(torch.utils.data.Dataset):
    def __init__(self, annonations, transform=None):
        """
        Args:
            annonations (string): Path to the json annonations file with coco json file.
            transform (callable, optional): Optional transform to be appliedon a sample.
        """
        self.annonations = annonations
        self.transform = transform

        with open(annonations, "r") as f:
            self.dataset = json.load(f)

    def __len__(self):
        return len(self.dataset["images"])


    def __getitem__(self, idx):
        # retrieve image
        image_id = self.dataset["images"][idx]["id"]
        segments = [element["segmentation"] for element in self.dataset["annotations"] if element["image_id"] == image_id]
        #print(f"segments: {len(segments)}")
        category_id = None  # Inizializza la variabile a None

        for element in self.dataset["annotations"]:
            if element["image_id"] == image_id:
                category_id = element["category_id"]
                break
        #print(image_id, category_id)
        
        lengths = [len(segment) for segment in segments[0]]
        segments = [segment for segment in segments[0] if len(segment) == max(lengths)]
        image_path = os.path.join(os.path.dirname(self.annonations), "oral1", self.dataset["images"][idx]["file_name"])
        image = Image.open(image_path).convert("RGB")

        # generate mask
        width, height = image.size
        mask = Image.new('L', (width, height))
        for segment in segments:
            #print(f"segment: {len(segment)}")
            ImageDraw.Draw(mask).polygon(segment, outline=1, fill=1)
        
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)
        
        # scale and repeat mask on all channels
        mask = mask / mask.max()

        # Creare un tensore di zeri con dimensione 4 x H x W
        tensor_masks = torch.zeros((4, mask.shape[1], mask.shape[2]), dtype=torch.float)
        tensor_masks[0] = 1 - mask
        tensor_masks[category_id] = mask

        return image, tensor_masks, category_id
        

@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg):


    print("TRAIN")
    dataset = MultiClassSegmentationDataset(
        "data/train.json",
        transform=transforms.Compose([
            transforms.Resize((cfg.dataset.resize, cfg.dataset.resize), antialias=True),
            transforms.ToTensor()])
    )
    for i in range(dataset.__len__()):
        image, mask = dataset.__getitem__(i)
        print(i, dataset.dataset["images"][i]["file_name"], image.shape, mask.shape, mask.max(), mask.min(), image.max(), image.min())
        plt.imshow(image.permute(1,2,0))
        plt.imshow(mask.permute(1,2,0), cmap='gray', alpha=0.5)
        plt.title(dataset.dataset["images"][i]["file_name"])
        # save image
        # create data/train if not exists
        if not os.path.exists("../outputs/train"):
            os.makedirs("../outputs/train")
        plt.savefig(f"../outputs/train/{i}.png")


    print("VALIDATION")
    dataset = MultiClassSegmentationDataset(
        "data/val.json",
        transform=transforms.Compose([
            transforms.Resize((cfg.dataset.resize, cfg.dataset.resize), antialias=True),
            transforms.ToTensor()])
    )
    for i in range(dataset.__len__()):
        image, mask = dataset.__getitem__(i)
        print(i, dataset.dataset["images"][i]["file_name"], image.shape, mask.shape, mask.max(), mask.min(), image.max(), image.min())
        plt.imshow(image.permute(1,2,0))
        plt.imshow(mask.permute(1,2,0), cmap='gray', alpha=0.5)
        plt.title(dataset.dataset["images"][i]["file_name"])
        # save image
        # create data/train if not exists
        if not os.path.exists("../outputs/val"):
            os.makedirs("../outputs/val")
        plt.savefig(f"../outputs/val/{i}.png")


    print("TEST")    
    dataset = MultiClassSegmentationDataset(
        "data/test.json",
        transform=transforms.Compose([
            transforms.Resize((cfg.dataset.resize, cfg.dataset.resize), antialias=True),
            transforms.ToTensor()])
    )
    for i in range(dataset.__len__()):
        image, mask = dataset.__getitem__(i)
        print(i, dataset.dataset["images"][i]["file_name"], image.shape, mask.shape, mask.max(), mask.min(), image.max(), image.min())
        plt.imshow(image.permute(1,2,0))
        plt.imshow(mask.permute(1,2,0), cmap='gray', alpha=0.5)
        plt.title(dataset.dataset["images"][i]["file_name"])
        # save image
        # create data/train if not exists
        if not os.path.exists("../outputs/test"):
            os.makedirs("../outputs/test")
        plt.savefig(f"../outputs/test/{i}.png")


if __name__ == "__main__":
    import torchvision
    main()