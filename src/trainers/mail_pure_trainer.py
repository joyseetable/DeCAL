"""
Pure MAIL trainer — AL (affine) layers only.

Builds PureMAILModel (no bridge / projectors / gates / teacher); all
training / validation / eval / optimizer logic is inherited from DistillSystem.
"""
from src.trainers import register_trainer
from src.systems.retrieval_system import DistillSystem


@register_trainer("mail_pure")
class PureMAILTrainer(DistillSystem):
    """Pure AL-layer MAIL retrieval trainer."""
    pass
