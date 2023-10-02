from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule

from src.dataset import OralSegmentationDataset

class OralClassificationDataModule(LightningDataModule):
    def __init__(self, train, val, test, batch_size=32, train_transform=None, val_transform=None, test_transform=None, transform=None):
        super().__init__()
        if train_transform is None:
            train_transform = transform
        if test_transform is None:
            test_transform = transform
        if val_transform is None:
            val_transform = transform

        self.train_dataset = OralSegmentationDataset(train, transform=train_transform)
        self.val_dataset = OralSegmentationDataset(val, transform=val_transform)
        self.test_dataset = OralSegmentationDataset(test, transform=test_transform)
        self.batch_size = batch_size
        

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)

    def predict_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)
    