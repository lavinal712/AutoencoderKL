import argparse
import glob
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from itertools import chain

import numpy as np
import torch
import torch.distributed as dist
from diffusers import AutoencoderKL
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
        "--use_hf",
        action="store_true",
        default=False,
        help="whether to use huggingface model",
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


def create_npz_from_sample_folder(sample_dir, num=50_000):
    """
    Builds a single .npz file from a folder of .png samples.
    """
    samples = []
    for i in tqdm(range(num), desc="Building .npz file from samples"):
        sample_pil = Image.open(f"{sample_dir}/{i:06d}.png")
        sample_np = np.asarray(sample_pil).astype(np.uint8)
        samples.append(sample_np)
    samples = np.stack(samples)
    assert samples.shape == (num, samples.shape[1], samples.shape[2], 3)
    npz_path = f"{sample_dir}.npz"
    np.savez(npz_path, arr_0=samples)
    print(f"Saved .npz file to {npz_path} [shape={samples.shape}].")
    return npz_path


if __name__ == "__main__":
    parser = get_parser()

    opt, unknown = parser.parse_known_args()

    if not opt.resume and not opt.resume_from_checkpoint:
        raise ValueError(
            "-r/--resume or --resume_from_checkpoint must be specified."
        )
    if opt.resume and not opt.use_hf:
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

    os.makedirs(opt.logdir, exist_ok=True)
    os.makedirs(os.path.join(opt.logdir, "inputs"), exist_ok=True)
    os.makedirs(os.path.join(opt.logdir, "reconstructions"), exist_ok=True)

    # Setup PyTorch:
    assert torch.cuda.is_available(), "Sampling with DDP requires at least one GPU."
    torch.set_grad_enabled(False)

    # Setup DDP:
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = rank % torch.cuda.device_count()
    seed = opt.seed * world_size + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={world_size}.")

    # model
    if not opt.use_hf:
        configs = [OmegaConf.load(cfg) for cfg in opt.base]
        cli = OmegaConf.from_dotlist(unknown)
        config = OmegaConf.merge(*configs, cli)
        model = instantiate_from_config(config.model)
        model.apply_ckpt(opt.resume_from_checkpoint)
    else:
        try:
            model = AutoencoderKL.from_pretrained(opt.resume)
        except:
            model = AutoencoderKL.from_pretrained(opt.resume, subfolder="vae")
    model.to(device)
    model.eval()

    perceptual_model = LPIPS().eval()
    perceptual_model.to(device)

    # data
    transform = transforms.Compose([
        transforms.Resize(opt.image_size),
        transforms.CenterCrop(opt.image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
    ])
    dataset = ImageNetDataset(opt.datadir, split="val", transform=transform)
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False,
        seed=seed,
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
    if rank == 0:
        print(f"Dataset contains {len(dataset):,} images ({opt.datadir})")

    psnr_list = []
    ssim_list = []
    lpips_list = []
    for step, batch in tqdm(enumerate(loader), total=len(loader), disable=rank != 0):
        x = batch["jpg"].to(device)
        inputs = x.detach().cpu().permute(0, 2, 3, 1).numpy()
        inputs = ((inputs + 1.0) / 2.0).clip(0.0, 1.0)

        with torch.no_grad():
            if not opt.use_hf:
                z = model.encode(x)
                x_hat = model.decode(z)
            else:
                z = model.encode(x).latent_dist.sample()
                x_hat = model.decode(z).sample
            lpips = perceptual_model(x, x_hat)
        reconstructions = x_hat.detach().cpu().permute(0, 2, 3, 1).numpy()
        reconstructions = ((reconstructions + 1.0) / 2.0).clip(0.0, 1.0)

        index_list = []
        input_image_list = []
        reconstruction_image_list = []
        for i, (_input, reconstruction) in enumerate(zip(inputs, reconstructions)):
            # metrics
            psnr = peak_signal_noise_ratio(_input, reconstruction, data_range=1.0)
            ssim = structural_similarity(_input, reconstruction, channel_axis=-1, data_range=1.0)
            psnr_list.append(psnr)
            ssim_list.append(ssim)
            lpips_list.append(lpips[i].item())

            # images
            index = step * opt.batch_size * world_size + i * world_size + rank
            _input = (_input * 255.0).astype(np.uint8)
            input_image = Image.fromarray(_input)
            reconstruction = (reconstruction * 255.0).astype(np.uint8)
            reconstruction_image = Image.fromarray(reconstruction)
            index_list.append(index)
            input_image_list.append(input_image)
            reconstruction_image_list.append(reconstruction_image)

        # save images
        with ThreadPoolExecutor(max_workers=max(32, os.cpu_count() * 3)) as executor:
            for index, input_image, reconstruction_image in zip(index_list, input_image_list, reconstruction_image_list):
                executor.submit(input_image.save, os.path.join(opt.logdir, "inputs", f"{index:06d}.png"))
                executor.submit(reconstruction_image.save, os.path.join(opt.logdir, "reconstructions", f"{index:06d}.png"))

    # gather
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
        # create_npz_from_sample_folder(os.path.join(opt.logdir, "inputs"))
        # create_npz_from_sample_folder(os.path.join(opt.logdir, "reconstructions"))
        command = f"python -m pytorch_fid {os.path.join(opt.logdir, 'inputs')} {os.path.join(opt.logdir, 'reconstructions')} --device cuda:{rank}"
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        rfid = float(result.stdout.split(" ")[-1])

        print(f"PSNR: {np.mean(psnr_list)} ± {np.std(psnr_list)}")
        print(f"SSIM: {np.mean(ssim_list)} ± {np.std(ssim_list)}")
        print(f"LPIPS: {np.mean(lpips_list)} ± {np.std(lpips_list)}")
        print(f"rFID: {rfid}")

    dist.barrier()
    dist.destroy_process_group()
