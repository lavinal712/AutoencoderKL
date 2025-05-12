# ViTVAE

For a long time, VAEs have mostly relied on CNN architectures. Some projects, like [ViT-VQGAN](https://github.com/thuanz123/enhancing-transformers), [ViTok](https://github.com/philippe-eecs/vitok), have experimented with Transformers. Now, thanks to various acceleration techniques, training Transformer-based VAEs has become more feasible, and scaling them to larger model sizes is more accessible. This branch offers a 2D version of the [MAGI-1 VAE](https://github.com/SandAI-org/MAGI-1/tree/main/inference/model/vae).

## Getting Started

### Installation

```
git clone https://github.com/lavinal712/AutoencoderKL.git -b dc-ae
cd AutoencoderKL
conda create -n autoencoderkl python=3.10 -y
conda activate autoencoderkl
pip install -r requirements.txt
pip install --no-cache-dir --no-build-isolation flash-attn==2.7.0.post2
```

### Training

```bash
torchrun --nproc_per_node=4 --nnodes=1 main.py \
    --base configs/magi-1_2d.yaml \
    --train \
    --scale_lr False \
    --wandb True \
```

## TODO

- [ ] Support [ViT-VQGAN](https://github.com/thuanz123/enhancing-transformers).
- [ ] Support [ViTok](https://github.com/philippe-eecs/vitok).

## Acknowledgments

- [MAGI-1](https://github.com/SandAI-org/MAGI-1)
- [ViT-VQGAN](https://github.com/thuanz123/enhancing-transformers)
- [ViTok](https://github.com/philippe-eecs/vitok)
