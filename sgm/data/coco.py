import os
from typing import Optional

import numpy as np
import pytorch_lightning as pl
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from torchvision.datasets import CocoCaptions

from sgm.data.augmentations import build_transform


class COCODataset(CocoCaptions):
    def __init__(
        self,
        root_dir,
        split="train",
        transform=None,
        **kwargs,
    ):
        super().__init__(
            root=os.path.join(root_dir, f"{split}2017"),
            annFile=os.path.join(
                root_dir,
                "annotations",
                f"captions_{split}2017.json",
            ),
            transform=transform,
            **kwargs,
        )

    def __getitem__(self, idx):
        image, captions = super().__getitem__(idx)
        if not captions:
            return {"jpg": image, "cls": [""]}
        caption = captions[np.random.randint(len(captions))]
        return {"jpg": image, "cls": [caption]}


class COCOLoader(pl.LightningDataModule):
    def __init__(
        self,
        batch_size: int,
        train: DictConfig = None,
        validation: Optional[DictConfig] = None,
        num_workers: int = 0,
        prefetch_factor: int = 2,
        shuffle: bool = False,
        drop_last: bool = False,
        pin_memory: bool = False,
        persistent_workers: bool = False,
    ):
        super().__init__()

        self.batch_size = batch_size
        self.num_workers = num_workers if num_workers is not None else batch_size * 2
        self.prefetch_factor = prefetch_factor if self.num_workers > 0 else None
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers and self.num_workers > 0

        if train is None:
            raise ValueError("COCOLoader requires a train configuration.")
        train_transform_config = train.get("transform_config") or dict()
        self.train_dataset = COCODataset(
            root_dir=train.root_dir,
            split="train",
            transform=build_transform(**train_transform_config),
        )
        if validation is not None:
            val_transform_config = validation.get("transform_config") or dict()
            self.test_dataset = COCODataset(
                root_dir=validation.root_dir,
                split="val",
                transform=build_transform(**val_transform_config),
            )
        else:
            print("Warning: No Validation Dataset defined, using that one from training")
            self.test_dataset = COCODataset(
                root_dir=train.root_dir,
                split="train",
                transform=build_transform(),
            )

    def prepare_data(self):
        pass

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
            drop_last=False,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
            drop_last=False,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )
