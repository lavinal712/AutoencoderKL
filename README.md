# AutoencoderKL

## About The Project

There are many great training scripts for VAE on Github. However, some repositories are not maintained and some are not updated to the latest version of PyTorch. Therefore, I decided to create this repository to provide a simple and easy-to-use training script for VAE by Lightning. Beside, the code is easy to transfer to other projects for time saving.

- Support training and finetuning both [Stable Diffusion](https://github.com/CompVis/stable-diffusion) VAE and [Flux](https://github.com/black-forest-labs/flux) VAE.
- Support evaluating reconstruction quality (FID, PSNR, SSIM, LPIPS).
- A practical guidance of training VAE.
- Easy to modify the code for your own research.

## Visualization

This is the visualization of AutoencoderKL. From left to right, there are the original image, the reconstructed image and the difference between them. From top to bottom, there are the results of SD VAE, SDXL VAE and FLUX VAE.

Image source: [https://www.bilibili.com/opus/762402574076739817](https://www.bilibili.com/opus/762402574076739817)

![baka](assets/visualization.png)

<!-- GETTING STARTED -->
## Getting Started

To get a local copy up and running follow these simple example steps.

### Installation

```bash
git clone https://github.com/lavinal712/AutoencoderKL.git
cd AutoencoderKL
conda create -n autoencoderkl python=3.10 -y
conda activate autoencoderkl
pip install -r requirements.txt
```

### Training

To start training, you need to prepare a config file. You can refer to the config files in the `configs` folder.

If you want to train on your own dataset, you should write your own data loader in `sgm/data` and modify the parameters in the config file.

Finetuning a VAE model is simple. You just need to specify the `ckpt_path` and `trainable_ae_params` in the config file. To keep the latent space of the original model, it is recommended to set decoder to be trainable.

Then, you can start training by running the following command.

```bash
NUM_GPUS=4
NUM_NODES=1

torchrun --nproc_per_node=${NUM_GPUS} --nnodes=${NUM_NODES} main.py \
    --base configs/autoencoder_kl_32x32x4.yaml \
    --train \
    --logdir logs/autoencoder_kl_32x32x4 \
    --scale_lr True \
    --wandb False \
```

### Evaluation

We provide a script to evaluate the reconstruction quality of the trained model. `--resume` provides a convenient way to load the checkpoint from the log directory.

We introduce multi-GPU and multi-thread method for faster evaluation.

The default dataset is ImageNet. You can change the dataset by modifying the `--datadir` in the command line and the evaluation script.

```bash
NUM_GPUS=4
NUM_NODES=1

torchrun --nproc_per_node=${NUM_GPUS} --nnodes=${NUM_NODES} eval.py \
    --resume logs/autoencoder_kl_32x32x4 \
    --base configs/autoencoder_kl_32x32x4.yaml \
    --logdir eval/autoencoder_kl_32x32x4 \
    --datadir /path/to/ImageNet \
    --image_size 256 \
    --batch_size 16 \
    --num_workers 16 \
```

Here are the evaluation results on ImageNet.

| Model         | rFID  | PSNR   | SSIM  | LPIPS |
| ------------- | ----- | ------ | ----- | ----- |
| sd-vae-ft-mse | 0.692 | 26.910 | 0.772 | 0.130 |
| sdxl-vae      | 0.665 | 27.376 | 0.794 | 0.122 |
| flux-vae      | 0.165 | 32.871 | 0.924 | 0.045 |

### Converting to diffusers

[huggingface/diffusers](https://github.com/huggingface/diffusers) is a library for diffusion models. It provides a script [convert_vae_pt_to_diffusers.py
](https://github.com/huggingface/diffusers/blob/main/scripts/convert_vae_pt_to_diffusers.py) to convert a PyTorch Lightning model to a diffusers model.

Currently, the script is not updated for all kinds of VAE models, just for SD VAE.

```bash
python convert_vae_pt_to_diffusers.py \
    --vae_path logs/autoencoder_kl_32x32x4/checkpoints/last.ckpt \
    --dump_path autoencoder_kl_32x32x4 \
```

## Guidance

Here are some guidance for training VAE. If there are any mistakes, please let me know.

- Learning rate: In LDM repository [CompVis/latent-diffusion](https://github.com/CompVis/latent-diffusion), the base learning rate is set to 4.5e-6 in the config file. However, the batch size is 12, accumulated gradient is 2 and `scale_lr` is set to `True`. Therefore, the effective learning rate is 4.5e-6 * 12 * 2 * 1 = 1.08e-4. It is better to set the learning rate from 1.0e-4 to 1.0e-5. In finetuning stage, it can be smaller than the first stage.
  - `scale_lr`: It is better to set `scale_lr` to `False` when training on a large dataset.
- Discriminator: You should open the discriminator in the end of the training, when the VAE has good reconstruction performance. In default, `disc_start` is set to 50001.
- Perceptual loss: LPIPS is a good metric for evaluating the quality of the reconstructed images. Some models use other perceptual loss functions to gain better performance, such as [sypsyp97/convnext_perceptual_loss](https://github.com/sypsyp97/convnext_perceptual_loss).

## Acknowledgments

Thanks for the following repositories. Without their code, this project would not be possible.

- [Stability-AI/generative-models](https://github.com/Stability-AI/generative-models). We heavily borrow the code from this repository, just modifing a few parameters for our concept.
- [CompVis/latent-diffusion](https://github.com/CompVis/latent-diffusion). We follow the hyperparameter settings of this repository in config files.
