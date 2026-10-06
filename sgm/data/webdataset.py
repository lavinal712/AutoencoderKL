import math
from typing import Optional

import pytorch_lightning as pl
import torch.distributed as dist
import webdataset as wds
from omegaconf import DictConfig

from sgm.data.augmentations import build_transform


class WebDatasetLoader(pl.LightningDataModule):
    def __init__(
        self,
        batch_size: int,
        train: DictConfig,
        test: Optional[DictConfig] = None,
        validation: Optional[DictConfig] = None,
        num_workers: int = 4,
        prefetch_factor: int = 2,
        drop_last: bool = False,
        pin_memory: bool = False,
        persistent_workers: bool = False,
        shard_shuffle: int = 100,
        sample_shuffle: int = 10000,
        epoch_size: Optional[int] = None,
    ):
        super().__init__()

        self.batch_size = batch_size
        self.num_workers = num_workers if num_workers is not None else batch_size * 2
        self.prefetch_factor = prefetch_factor if self.num_workers > 0 else None
        self.drop_last = drop_last
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers and self.num_workers > 0

        self.shard_shuffle = shard_shuffle
        self.sample_shuffle = sample_shuffle
        self.epoch_size = epoch_size

        if train is None:
            raise ValueError("WebDatasetLoader requires a train configuration.")
        self.train_dataset = self.build_dataset(
            train,
            split="train",
            shard_shuffle=shard_shuffle,
            sample_shuffle=sample_shuffle,
            resampled=epoch_size is not None,
        )
        self.test_dataset = self.build_dataset(test, split="test")
        self.val_dataset = self.build_dataset(validation, split="val")

    @staticmethod
    def build_dataset(
        config: DictConfig,
        split: str,
        shard_shuffle: int = 0,
        sample_shuffle: int = 0,
        resampled: bool = False,
    ):
        transform_config = config.get("transform_config") or {}
        transform = build_transform(**transform_config)
        input_key = config.get("input_key", "jpg")
        dataset = wds.WebDataset(
            config.shards,
            resampled=resampled if split == "train" else False,
            shardshuffle=shard_shuffle if split == "train" else False,
            nodesplitter=wds.split_by_node,
            workersplitter=wds.split_by_worker,
        )
        if split == "train" and sample_shuffle > 0:
            dataset = dataset.shuffle(sample_shuffle)

        if input_key == "jpg":
            dataset = (
                dataset.decode("pilrgb")
                .rename(jpg="jpg;jpeg;png;webp", cls="cls", keep=False)
                .map_dict(jpg=transform)
            )
        elif input_key == "mp4":
            dataset = (
                dataset.decode(wds.torch_video)
                .rename(mp4="mp4;avi;mov;webm", cls="cls", keep=False)
                .map_dict(
                    mp4=lambda video: transform(
                        video[0].permute(0, 3, 1, 2)
                    ).permute(1, 0, 2, 3).contiguous()
                )
            )
        else:
            raise ValueError(f"Unsupported WebDataset input key: {input_key}")

        return dataset

    def prepare_data(self):
        pass

    def train_dataloader(self):
        loader = wds.WebLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )
        if self.epoch_size is not None:
            world_size = (
                dist.get_world_size()
                if dist.is_available() and dist.is_initialized()
                else 1
            )
            global_batch_size = self.batch_size * world_size
            num_batches = (
                self.epoch_size // global_batch_size
                if self.drop_last
                else math.ceil(self.epoch_size / global_batch_size)
            )
            loader = loader.with_epoch(num_batches).with_length(num_batches)

        return loader

    def test_dataloader(self):
        return wds.WebLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
            drop_last=False,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )

    def val_dataloader(self):
        return wds.WebLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
            drop_last=False,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )
