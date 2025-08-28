import os
from typing import Optional

import pytorch_lightning as pl
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder


class ImageNetDataset(Dataset):
    def __init__(self, root_dir, split="train", transform=None):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.dataset = ImageFolder(os.path.join(root_dir, split), transform=self.transform)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return {"jpg": self.dataset[idx][0], "cls": self.dataset[idx][1]}


class ImageNetLoader(pl.LightningDataModule):
    def __init__(
        self,
        batch_size: int,
        train: DictConfig = None,
        validation: Optional[DictConfig] = None,
        num_workers: int = 0,
        prefetch_factor: int = 2,
        shuffle: bool = False,
        shuffle_test_loader: bool = False,
        shuffle_val_dataloader: bool = False,
    ):
        super().__init__()

        self.batch_size = batch_size
        self.num_workers = num_workers if num_workers is not None else batch_size * 2
        self.prefetch_factor = prefetch_factor
        self.shuffle = shuffle
        self.shuffle_test_loader = shuffle_test_loader
        self.shuffle_val_dataloader = shuffle_val_dataloader

        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])]
        )
        if train.get("transform", None):
            size = train.get("size", 256)
            transform = transforms.Compose([
                transforms.Resize(size, interpolation=Image.BICUBIC),
                transforms.CenterCrop(size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
            ])

        self.train_dataset = ImageNetDataset(
            root_dir=train.root_dir, split="train", transform=transform
        )
        if validation is not None:
            self.test_dataset = ImageNetDataset(
                root_dir=validation.root_dir, split="val", transform=transform
            )
        else:
            print("Warning: No Validation Datasetdefined, using that one from training")
            self.test_dataset = self.train_dataset

    def prepare_data(self):
        pass

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle_test_loader,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
        )

    def val_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle_val_dataloader,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
        )
