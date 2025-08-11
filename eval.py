import argparse
import glob
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from itertools import chain

import numpy as np
import torch
import torch.distributed as dist
from natsort import natsorted
from omegaconf import OmegaConf
from packaging import version
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from sgm.data.imagenet import ImageNetDataset
from sgm.util import instantiate_from_config
from sgm.modules.autoencoding.lpips.loss.lpips import LPIPS


def get_parser(**parser_kwargs):
    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument(
        "-r",
        "--resume",
        type=str,
        const=True,
        default="",
        nargs="?",
        help="resume from logdir or checkpoint in logdir",
    )
    parser.add_argument(
        "-b",
        "--base",
        nargs="*",
        metavar="base_config.yaml",
        help="paths to base configs. Loaded from left-to-right. "
        "Parameters can be overwritten or added with command-line options of the form `--key value`.",
        default=list(),
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=0,
        help="seed for initialization",
    )
    parser.add_argument(
        "-l",
        "--logdir",
        type=str,
        default="logs",
        help="directory for logging dat shit",
    )
    parser.add_argument(
        "-d",
        "--datadir",
        type=str,
        default="data",
        help="directory for testing data",
    )
    parser.add_argument(
        "-iz",
        "--image_size",
        type=int,
        default=256,
        help="image size for testing data",
    )
    parser.add_argument(
        "-bz",
        "--batch_size",
        type=int,
        default=1,
        help="batch size for sampling data",
    )
    parser.add_argument(
        "-nw",
        "--num_workers",
        type=int,
        default=0,
        help="number of workers for sampling data",
    )
    if version.parse(torch.__version__) >= version.parse("2.0.0"):
        parser.add_argument(
            "--resume_from_checkpoint",
            type=str,
            default=None,
            help="single checkpoint file to resume from",
        )
    return parser


def get_checkpoint_name(logdir):
    ckpt = os.path.join(logdir, "checkpoints", "last**.ckpt")
    ckpt = natsorted(glob.glob(ckpt))
    print('available "last" checkpoints:')
    print(ckpt)
    if len(ckpt) > 1:
        print("got most recent checkpoint")
        ckpt = sorted(ckpt, key=lambda x: os.path.getmtime(x))[-1]
        print(f"Most recent ckpt is {ckpt}")
        with open(os.path.join(logdir, "most_recent_ckpt.txt"), "w") as f:
            f.write(ckpt + "\n")
        try:
            version = int(ckpt.split("/")[-1].split("-v")[-1].split(".")[0])
        except Exception as e:
            print("version confusion but not bad")
            print(e)
            version = 1
        # version = last_version + 1
    else:
        # in this case, we only have one "last.ckpt"
        ckpt = ckpt[0]
        version = 1
    melk_ckpt_name = f"last-v{version}.ckpt"
    print(f"Current melk ckpt name: {melk_ckpt_name}")
    return ckpt, melk_ckpt_name


if __name__ == "__main__":
    parser = get_parser()

    opt, unknown = parser.parse_known_args()

    if not opt.resume and not opt.resume_from_checkpoint:
        raise ValueError(
            "-r/--resume or --resume_from_checkpoint must be specified."
        )
    if opt.resume:
        if not os.path.exists(opt.resume):
            raise ValueError("Cannot find {}".format(opt.resume))
        if os.path.isfile(opt.resume):
            paths = opt.resume.split("/")
            # idx = len(paths)-paths[::-1].index("logs")+1
            # logdir = "/".join(paths[:idx])
            logdir = "/".join(paths[:-2])
            ckpt = opt.resume
            _, melk_ckpt_name = get_checkpoint_name(logdir)
        else:
            assert os.path.isdir(opt.resume), opt.resume
            logdir = opt.resume.rstrip("/")
            ckpt, melk_ckpt_name = get_checkpoint_name(logdir)

        print("#" * 100)
        print(f'Resuming from checkpoint "{ckpt}"')
        print("#" * 100)

        opt.resume_from_checkpoint = ckpt
        base_configs = sorted(glob.glob(os.path.join(logdir, "configs/*.yaml")))
        opt.base = base_configs + opt.base

    logdir = opt.logdir
    os.makedirs(logdir, exist_ok=True)
    os.makedirs(os.path.join(logdir, "gt"), exist_ok=True)
    os.makedirs(os.path.join(logdir, "rec"), exist_ok=True)

    # Setup PyTorch:
    assert torch.cuda.is_available(), "Sampling with DDP requires at least one GPU. sample.py supports CPU-only usage"
    torch.set_grad_enabled(False)

    # Setup DDP:
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = opt.seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    configs = [OmegaConf.load(cfg) for cfg in opt.base]
    cli = OmegaConf.from_dotlist(unknown)
    config = OmegaConf.merge(*configs, cli)

    # model
    model = instantiate_from_config(config.model)
    model.apply_ckpt(opt.resume_from_checkpoint)
    model.to(device)
    model.eval()

    perceptual_model = LPIPS().eval()
    perceptual_model.to(device)

    # data
    transform = transforms.Compose([
        transforms.Resize(opt.image_size, interpolation=Image.BICUBIC),
        transforms.CenterCrop(opt.image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
    ])
    dataset = ImageNetDataset(opt.datadir, split="val", transform=transform)
    sampler = DistributedSampler(
        dataset,
        num_replicas=dist.get_world_size(),
        rank=rank,
        shuffle=False,
        seed=opt.seed,
        drop_last=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=False,
        sampler=sampler,
        num_workers=opt.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    print(f"Dataset contains {len(dataset):,} images ({opt.datadir})")   

    eval_steps = 0
    psnr_list = []
    ssim_list = []
    lpips_list = []
    for batch in tqdm(loader):
        x = batch["jpg"].to(device)
        gt = x.detach().cpu().permute(0, 2, 3, 1).numpy()
        gt = ((gt + 1.0) / 2.0).clip(0.0, 1.0)

        with torch.no_grad():
            z = model.encode(x)
            x_hat = model.decode(z)
            lpips = perceptual_model(x, x_hat)
        x_hat = x_hat.detach().cpu().permute(0, 2, 3, 1).numpy()
        x_hat = ((x_hat + 1.0) / 2.0).clip(0.0, 1.0)

        index_list = []
        gt_img_list = []
        x_hat_img_list = []
        for i, (_gt, _x_hat) in enumerate(zip(gt, x_hat)):
            # metrics
            psnr = peak_signal_noise_ratio(_gt, _x_hat, data_range=1.0)
            ssim = structural_similarity(_gt, _x_hat, multichannel=True, channel_axis=-1, data_range=1.0)
            psnr_list.append(psnr)
            ssim_list.append(ssim)
            lpips_list.append(lpips[i].item())

            # save images
            index = i * dist.get_world_size() + rank + eval_steps * opt.batch_size * dist.get_world_size()
            _gt = (_gt * 255.0).astype(np.uint8)
            _x_hat = (_x_hat * 255.0).astype(np.uint8)
            gt_img = Image.fromarray(_gt)
            x_hat_img = Image.fromarray(_x_hat)
            index_list.append(index)
            gt_img_list.append(gt_img)
            x_hat_img_list.append(x_hat_img)
        with ThreadPoolExecutor(max_workers=max(32, os.cpu_count() * 3)) as executor:
            for index, gt_img, x_hat_img in zip(index_list, gt_img_list, x_hat_img_list):
                executor.submit(gt_img.save, os.path.join(logdir, "gt", f"{index:06d}.png"))
                executor.submit(x_hat_img.save, os.path.join(logdir, "rec", f"{index:06d}.png"))

        eval_steps += 1

    world_size = dist.get_world_size()
    gather_psnr_list = [None for _ in range(world_size)]
    gather_ssim_list = [None for _ in range(world_size)]
    gather_lpips_list = [None for _ in range(world_size)]
    dist.all_gather_object(gather_psnr_list, psnr_list)
    dist.all_gather_object(gather_ssim_list, ssim_list)
    dist.all_gather_object(gather_lpips_list, lpips_list)

    if rank == 0:
        # PSNR, SSIM, LPIPS
        psnr_list = list(chain(*gather_psnr_list))
        ssim_list = list(chain(*gather_ssim_list))
        lpips_list = list(chain(*gather_lpips_list))

        # rFID
        command = f"python -m pytorch_fid {os.path.join(logdir, 'gt')} {os.path.join(logdir, 'rec')} --device cuda:{rank}"
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        rfid = float(result.stdout.split(" ")[-1])

        np.savez(os.path.join(logdir, "results.npz"), psnr=np.array(psnr_list), ssim=np.array(ssim_list), lpips=np.array(lpips_list))
        with open(os.path.join(logdir, "results.txt"), "w") as f:
            f.write(f"PSNR: {np.mean(psnr_list)} ± {np.std(psnr_list)}\n")
            f.write(f"SSIM: {np.mean(ssim_list)} ± {np.std(ssim_list)}\n")
            f.write(f"LPIPS: {np.mean(lpips_list)} ± {np.std(lpips_list)}\n")
            f.write(f"rFID: {rfid}\n")

    dist.barrier()
    dist.destroy_process_group()
