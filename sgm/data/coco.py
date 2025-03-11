import json
import os
from typing import Optional

import numpy as np
import pytorch_lightning as pl
from omegaconf import DictConfig
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class COCODataset(Dataset):
    def __init__(self, root_dir, split="train", transform=None):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform

        data_json = os.path.join(root_dir, "annotations", f"captions_{split}2017.json")
        with open(data_json, "r") as json_file:
            self.json_data = json.load(json_file)
            self.img_id_to_filepath = dict()
            self.img_id_to_captions = dict()

        imagedirs = self.json_data["images"]
        self.labels = {"image_ids": list()}
        for imgdir in imagedirs:
            self.img_id_to_filepath[imgdir["id"]] = os.path.join(
                root_dir, f"{split}2017", imgdir["file_name"]
            )
            self.img_id_to_captions[imgdir["id"]] = list()
            self.labels["image_ids"].append(imgdir["id"])

        capdirs = self.json_data["annotations"]
        for capdir in capdirs:
            # there are in average 5 captions per image
            self.img_id_to_captions[capdir["image_id"]].append(np.array(capdir["caption"]))

    def __len__(self):
        return len(self.labels["image_ids"])

    def __getitem__(self, idx):
        img_path = self.img_id_to_filepath[self.labels["image_ids"][idx]]
        image = Image.open(img_path)
        if image.mode != "RGB":
            image = image.convert("RGB")
        image = self.transform(image)
        captions = self.img_id_to_captions[self.labels["image_ids"][idx]]
        # randomly draw one of all available captions per image
        caption = captions[np.random.randint(0, len(captions))]
        return {"jpg": image, "cls": [str(caption)]}


class COCOLoader(pl.LightningDataModule):
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
                transforms.Resize(size),
                transforms.CenterCrop(size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
            ])

        self.train_dataset = COCODataset(
            root_dir=train.root_dir, split="train", transform=transform
        )
        if validation is not None:
            self.test_dataset = COCODataset(
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
