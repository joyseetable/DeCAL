"""
Unified training entry point for GROUT-MAIL (Hydra-based, like train_new.py).

Usage:
    # Use default config (configs/train_pihm_flickr30k.yaml)
    CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 train.py

    # Use a different config
    CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 train.py --config-name=train_mail_flickr30k

    # Override specific values
    CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 train.py seed=123 model.mail_rank=16
"""
import os
import hydra
import torch
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
from pytorch_lightning.strategies import DDPStrategy
from typing import List

from src.datasets import create_datamodule
from src.trainers import build_trainer


@hydra.main(config_path="configs", config_name="train_mail_pure_flickr30k", version_base="1.2")
def main(cfg: DictConfig) -> None:
    if os.environ.get("LOCAL_RANK", "0") == "0":
        print("--- Config ---")
        print(OmegaConf.to_yaml(cfg))
        print("--------------")

    pl.seed_everything(cfg.seed, workers=True)

    # Build components
    datamodule = create_datamodule(cfg)
    trainer_type = cfg.model.get("type", "mail_pure")
    system = build_trainer(trainer_type, cfg)

    # Resume from checkpoint
    load_ckpt = cfg.get("resume_from_checkpoint", None)
    if load_ckpt:
        ckpt = torch.load(load_ckpt, map_location="cpu")
        system.load_state_dict(ckpt["state_dict"], strict=False)

    # Callbacks
    callbacks: List[pl.Callback] = []
    if hasattr(cfg, "checkpoint_callback"):
        callbacks.append(ModelCheckpoint(save_on_train_epoch_end=False, **cfg.checkpoint_callback))

    # Loggers
    save_dir = cfg.trainer.get("default_root_dir", "./outputs")
    loggers = [
        TensorBoardLogger(save_dir=save_dir, name=cfg.project_name,
                          version=cfg.dataset.get("name", "default")),
        CSVLogger(save_dir=save_dir, name=cfg.project_name,
                  version=cfg.dataset.get("name", "default")),
    ]

    # Trainer
    trainer = pl.Trainer(
        max_epochs=cfg.trainer.get("max_epochs", 30),
        logger=loggers,
        callbacks=callbacks,
        strategy=DDPStrategy(find_unused_parameters=True),
        accelerator="gpu",
        devices="auto",
        gradient_clip_algorithm="norm",
        val_check_interval=cfg.trainer.get("val_check_interval", 1.0),
    )

    if cfg.get("test_only", False):
        trainer.test(system, datamodule=datamodule, ckpt_path=load_ckpt)
    else:
        trainer.fit(system, datamodule=datamodule)

    if os.environ.get("LOCAL_RANK", "0") == "0":
        print("----- Run Finished -----")


if __name__ == '__main__':
    main()
