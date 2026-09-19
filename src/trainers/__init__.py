"""
Trainer registry for GROUT-MAIL.
Supports pluggable trainer backends via a lightweight registry pattern.

Usage:
    from src.trainers import build_trainer
    trainer = build_trainer("mail_pure", config)
"""
from typing import Dict, Type
import pytorch_lightning as pl

TRAINER_REGISTRY: Dict[str, Type[pl.LightningModule]] = {}


def register_trainer(name: str):
    """Decorator to register a trainer class in the registry."""
    def _register(cls):
        TRAINER_REGISTRY[name] = cls
        return cls
    return _register


# Map model.type (from config) → trainer name
TYPE_TO_TRAINER = {
    "ours": "mail_pure",
    "mail_pure": "mail_pure",
}


def build_trainer(identifier: str, config) -> pl.LightningModule:
    """Factory function to build a trainer by name or model type."""
    name = TYPE_TO_TRAINER.get(identifier, identifier)
    if name not in TRAINER_REGISTRY:
        raise ValueError(
            f"Unknown trainer: '{identifier}'. Available types: {list(TYPE_TO_TRAINER.keys())}. "
            f"Registered trainers: {list(TRAINER_REGISTRY.keys())}"
        )
    return TRAINER_REGISTRY[name](config)


# Import trainers to trigger registration
from src.trainers.mail_pure_trainer import PureMAILTrainer  # noqa: F401
