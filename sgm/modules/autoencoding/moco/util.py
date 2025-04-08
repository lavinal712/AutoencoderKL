import hashlib
import os
import tarfile

import requests
import torch
import torch.nn as nn
from tqdm import tqdm

URL_MAP = {
    "mocov1_200ep": "https://dl.fbaipublicfiles.com/moco/moco_checkpoints/moco_v1_200ep/moco_v1_200ep_pretrain.pth.tar",
    "mocov2_200ep": "https://dl.fbaipublicfiles.com/moco/moco_checkpoints/moco_v2_200ep/moco_v2_200ep_pretrain.pth.tar",
    "mocov2_800ep": "https://dl.fbaipublicfiles.com/moco/moco_checkpoints/moco_v2_800ep/moco_v2_800ep_pretrain.pth.tar",
    "mocov3_vits": "https://dl.fbaipublicfiles.com/moco-v3/vit-s-300ep/vit-s-300ep.pth.tar",
    "mocov3_vitb": "https://dl.fbaipublicfiles.com/moco-v3/vit-b-300ep/vit-b-300ep.pth.tar",
}

CKPT_MAP = {
    "mocov1_200ep": "moco_v1_200ep_pretrain.pth",
    "mocov2_200ep": "moco_v2_200ep_pretrain.pth",
    "mocov2_800ep": "moco_v2_800ep_pretrain.pth",
    "mocov3_vits": "vit-s-300ep.pth",
    "mocov3_vitb": "vit-b-300ep.pth",
}

MD5_MAP = {
    "mocov1_200ep": "b251726a57be750490c34a7602b59076",
    "mocov2_200ep": "59fd9945645e27d585be89338ce9232e",
    "mocov2_800ep": "a04e12f8b0e44fdcac1fb4e06f33727b",
    "mocov3_vits": "f32f0062e8884e64bcd2044c67bd3e43",
    "mocov3_vitb": "7fe6d104c5ba222fecc6dc838f2dbcf9",
}


def download(url, local_path, chunk_size=1024):
    os.makedirs(os.path.split(local_path)[0], exist_ok=True)
    with requests.get(url, stream=True) as r:
        total_size = int(r.headers.get("content-length", 0))
        with tqdm(total=total_size, unit="B", unit_scale=True) as pbar:
            with open(local_path, "wb") as f:
                for data in r.iter_content(chunk_size=chunk_size):
                    if data:
                        f.write(data)
                        pbar.update(chunk_size)


def md5_hash(path):
    with open(path, "rb") as f:
        content = f.read()
    return hashlib.md5(content).hexdigest()


def get_ckpt_path(name, root, check=False):
    assert name in URL_MAP
    path = os.path.join(root, CKPT_MAP[name])
    if not os.path.exists(path) or (check and not md5_hash(path) == MD5_MAP[name]):
        print("Downloading {} model from {} to {}".format(name, URL_MAP[name], path))
        download(URL_MAP[name], path)
        md5 = md5_hash(path)
        assert md5 == MD5_MAP[name], md5
    return path
