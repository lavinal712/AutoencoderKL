import os
from typing import Optional

import pytorch_lightning as pl
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from torchvision.datasets import UCF101

from sgm.data.augmentations import build_transform


class UCF101Dataset(UCF101):
    def __init__(
        self,
        root_dir,
        split="train",
        transform=None,
        frames_per_clip=17,
        step_between_clips=17,
        **kwargs,
    ):
        super().__init__(
            root=os.path.join(root_dir, "UCF-101"),
            annotation_path=os.path.join(root_dir, "ucfTrainTestlist"),
            frames_per_clip=frames_per_clip,
            step_between_clips=step_between_clips,
            train=(split == "train"),
            transform=transform,
            num_workers=16,
            output_format="TCHW",
            **kwargs,
        )

    def __len__(self):
        return super().__len__()

    def __getitem__(self, idx):
        video, audio, label = super().__getitem__(idx)
        return {"mp4": video.permute(1, 0, 2, 3).contiguous(), "cls": label}


class UCF101Loader(pl.LightningDataModule):
    def __init__(
        self,
        batch_size: int,
        train: DictConfig = None,
        test: Optional[DictConfig] = None,
        num_workers: int = 0,
        prefetch_factor: int = 2,
        shuffle: bool = False,
        drop_last: bool = False,
    ):
        super().__init__()

        self.batch_size = batch_size
        self.num_workers = num_workers if num_workers is not None else batch_size * 2
        self.prefetch_factor = prefetch_factor if self.num_workers > 0 else None
        self.shuffle = shuffle
        self.drop_last = drop_last

        if train is None:
            raise ValueError("UCF101Loader requires a train configuration.")
        self.train_dataset = self.build_dataset(train, split="train")
        self.test_dataset = self.build_dataset(test, split="test")

    @staticmethod
    def build_dataset(config: DictConfig, split: str):
        transform_config = config.get("transform_config") or {}
        return UCF101Dataset(
            root_dir=config.root_dir,
            frames_per_clip=config.get("frames_per_clip", 17),
            step_between_clips=config.get("step_between_clips", 17),
            split=split,
            transform=build_transform(**transform_config),
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
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
            drop_last=False,
        )

    def val_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
            drop_last=False,
        )
