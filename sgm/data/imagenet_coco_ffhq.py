import json
import os
from typing import Optional

import numpy as np
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder


class ImageNetCOCOFFHQDataset(Dataset):
    def __init__(
        self,
        imagenet_dir,
        coco_dir,
        ffhq_dir,
        split="train",
        transform=None,
    ):
        self.imagenet_dir = imagenet_dir
        self.coco_dir = coco_dir
        self.ffhq_dir = ffhq_dir
        self.split = split
        self.transform = transform
        self.imagenet_dataset = ImageFolder(os.path.join(imagenet_dir, split), transform=self.transform)
        self.coco_dataset = self._get_coco_dataset(coco_dir, split)
        self.ffhq_dataset = self._get_ffhq_dataset(ffhq_dir, split)

    def _get_coco_dataset(self, root_dir, split):
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

        dataset = []
        for img_id in self.labels["image_ids"]:
            dataset.append((self.img_id_to_filepath[img_id], img_id))

        return dataset

    def _get_ffhq_dataset(self, root_dir, split):
        data_json = os.path.join(root_dir, "ffhq-dataset-v2.json")
        with open(data_json, "r") as json_file:
            self.metadata = json.load(json_file)
        split_map = {"train": "training", "val": "validation"}

        dataset = []
        for i, data in self.metadata.items():
            category = data["category"]
            if category == split_map[split]:
                image_path = os.path.join(root_dir, data["image"]["file_path"])
                dataset.append((image_path, int(i)))

        return dataset

    def __len__(self):
        return len(self.imagenet_dataset) + len(self.coco_dataset) + len(self.ffhq_dataset)

    def __getitem__(self, idx):
        if idx < len(self.imagenet_dataset):
            return {"jpg": self.imagenet_dataset[idx][0], "cls": self.imagenet_dataset[idx][1]}
        elif idx < len(self.imagenet_dataset) + len(self.coco_dataset):
            idx = idx - len(self.imagenet_dataset)
            img_path = self.coco_dataset[idx][0]
            image = Image.open(img_path)
            if image.mode != "RGB":
                image = image.convert("RGB")
            image = self.transform(image)
            return {"jpg": image, "cls": self.coco_dataset[idx][1]}
        else:
            idx = idx - len(self.imagenet_dataset) - len(self.coco_dataset)
            img_path = self.ffhq_dataset[idx][0]
            image = Image.open(img_path)
            if image.mode != "RGB":
                image = image.convert("RGB")
            image = self.transform(image)
            return {"jpg": image, "cls": self.ffhq_dataset[idx][1]}


class ImageNetCOCOFFHQLoader(pl.LightningDataModule):
    def __init__(
        self,
        batch_size,
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
            if train.get("transform", None) == "center_crop":
                transform = transforms.Compose([
                    transforms.Resize(size),
                    transforms.CenterCrop(size),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
                ])
            elif train.get("transform", None) == "random_crop":
                transform = transforms.Compose([
                    transforms.Resize(size),
                    transforms.RandomCrop(size),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
                ])
            else:
                raise ValueError(f"Invalid transform: {train.get('transform', None)}")

        self.train_dataset = ImageNetCOCOFFHQDataset(
            imagenet_dir=train.imagenet_dir,
            coco_dir=train.coco_dir,
            ffhq_dir=train.ffhq_dir,
            split="train",
            transform=transform,
        )
        if validation is not None:
            self.test_dataset = ImageNetCOCOFFHQDataset(
                imagenet_dir=validation.imagenet_dir,
                coco_dir=validation.coco_dir,
                ffhq_dir=validation.ffhq_dir,
                split="val",
                transform=transform,
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
