import contextlib
import logging
import os
import shutil
from typing import Optional

import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback

from .util import default

logpy = logging.getLogger(__name__)


class SaveDiffusersCallback(Callback):
    def __init__(
        self,
        save_dir: Optional[str] = None,
        every_n_epochs: int = 1,
        every_n_train_steps: int = 0,
        export_on_train_end: bool = True,
        use_ema: bool = False,
        safe_serialization: bool = True,
    ):
        super().__init__()
        assert (
            every_n_epochs or every_n_train_steps or export_on_train_end
        )
        self.save_dir = default(save_dir, "diffusers")
        self.every_n_epochs = every_n_epochs
        self.every_n_train_steps = every_n_train_steps
        self.export_on_train_end = export_on_train_end
        self.use_ema = use_ema
        self.safe_serialization = safe_serialization
    
    def export(self, trainer, pl_module, name: str) -> None:
        if not trainer.is_global_zero:
            return
        if not hasattr(pl_module, "save_pretrained"):
            logpy.warning(
                "%s has no `save_pretrained` method.", type(pl_module).__name__
            )
            return

        logdir = getattr(trainer, "logdir", None) or trainer.default_root_dir
        save_dir = os.path.join(logdir, self.save_dir, name)
        tmp_dir = save_dir + ".tmp"
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir)
        os.makedirs(tmp_dir, exist_ok=True)

        if self.use_ema and getattr(pl_module, "use_ema", False):
            ema_context = pl_module.ema_scope("Saving EMA weights")
        else:
            ema_context = contextlib.nullcontext()

        try:
            with ema_context:
                pl_module.save_pretrained(
                    tmp_dir, safe_serialization=self.safe_serialization
                )
        except Exception:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            raise

        if os.path.exists(save_dir):
            shutil.rmtree(save_dir)
        os.rename(tmp_dir, save_dir)

        logpy.info(f"Saved diffusers model to {save_dir}")

    def on_train_epoch_end(self, trainer, pl_module) -> None:
        if not self.every_n_epochs:
            return
        if (trainer.current_epoch + 1) % self.every_n_epochs:
            return
        self.export(trainer, pl_module, f"epoch={trainer.current_epoch:06d}")

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        if not self.every_n_train_steps:
            return
        if trainer.global_step % self.every_n_train_steps:
            return
        self.export(trainer, pl_module, f"step={trainer.global_step:09d}")

    def on_train_end(self, trainer, pl_module) -> None:
        if not self.export_on_train_end:
            return
        self.export(trainer, pl_module, "last")
