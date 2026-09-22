from typing import Any, Mapping, Optional, Sequence, Tuple, Union

from omegaconf import OmegaConf
from torchvision import transforms

from sgm.util import instantiate_from_config

DEFAULT_MEAN: Tuple[float, float, float] = (0.5, 0.5, 0.5)
DEFAULT_STD: Tuple[float, float, float] = (0.5, 0.5, 0.5)

Size = Union[int, Sequence[int]]
TransformConfig = Mapping[str, Any]


def instantiate_transform(config: TransformConfig):
    if OmegaConf.is_config(config):
        config = OmegaConf.to_container(config, resolve=True)
    if not isinstance(config, Mapping) or "target" not in config:
        raise ValueError("Each transform must contain a 'target'.")
    unknown = set(config) - {"target", "params"}
    if unknown:
        raise ValueError(f"Unknown transform fields: {sorted(unknown)}")

    params = dict(config.get("params", {}))
    if "transforms" in params:
        params["transforms"] = [
            instantiate_transform(config) for config in params["transforms"]
        ]

    op = instantiate_from_config({"target": config["target"], "params": params})
    if not callable(op):
        raise TypeError(f"Transform '{config['target']}' is not callable.")

    return op


def build_transform(
    size: Optional[Size] = 256,
    mean: Sequence[float] = DEFAULT_MEAN,
    std: Sequence[float] = DEFAULT_STD,
    *,
    image_transforms: Optional[Sequence[TransformConfig]] = None,
    tensor_transforms: Optional[Sequence[TransformConfig]] = None,
) -> transforms.Compose:
    if image_transforms is None:
        image_ops = []
        if size is not None:
            resize_size = size if isinstance(size, int) else list(size)
            image_ops = [
                transforms.Resize(resize_size, antialias=True),
                transforms.CenterCrop(resize_size),
            ]
    else:
        image_ops = [instantiate_transform(config) for config in image_transforms]

    tensor_ops = (
        []
        if tensor_transforms is None
        else [instantiate_transform(config) for config in tensor_transforms]
    )

    return transforms.Compose([
        *image_ops,
        transforms.ToTensor(),
        *tensor_ops,
        transforms.Normalize(mean=list(mean), std=list(std)),
    ])
