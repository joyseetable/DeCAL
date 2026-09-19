import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
import os
from datetime import datetime
import torch

class TextLoggingCallback(Callback):
    def __init__(self, log_dir="text_logs"):
        super().__init__()
        self.log_dir = log_dir
        self.log_file = None

    def setup(self, trainer, pl_module, stage: str):
        """在 DDP 启动后，由 Rank 0 创建日志文件"""
        if trainer.is_global_zero:
            # 使用 trainer.log_dir，确保所有日志都在同一个地方
            root_dir = trainer.log_dir or "."
            self.log_dir = os.path.join(root_dir, self.log_dir)
            os.makedirs(self.log_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            self.log_file = os.path.join(self.log_dir, f"training_results_{timestamp}.txt")
            with open(self.log_file, 'a') as f:
                f.write(f"Training started at {timestamp}\n\n")
    
    def on_validation_epoch_end(self, trainer, pl_module):
        """在每个验证 epoch 结束后，由 Rank 0 写入指标"""
        if trainer.is_global_zero and self.log_file:
            # 从 Trainer 的全局“成绩单”上读取数值
            metrics = trainer.callback_metrics
            epoch = pl_module.current_epoch
            
            # 安全地获取指标
            rsum_tensor = metrics.get("val/RSUM")
            if rsum_tensor is None: return

            rsum = rsum_tensor.item()
            i2t_r1 = metrics.get("val/recall/i2t_r1", torch.tensor(-1)).item()
            i2t_r5 = metrics.get("val/recall/i2t_r5", torch.tensor(-1)).item()
            i2t_r10 = metrics.get("val/recall/i2t_r10", torch.tensor(-1)).item()
            t2i_r1 = metrics.get("val/recall/t2i_r1", torch.tensor(-1)).item()
            t2i_r5 = metrics.get("val/recall/t2i_r5", torch.tensor(-1)).item()
            t2i_r10 = metrics.get("val/recall/t2i_r10", torch.tensor(-1)).item()

            log_message = (
                f"--- Epoch {epoch} Validation Results ---\n"
                f"  RSUM: {rsum:.4f}\n"
                f"  I2T: R@1 {i2t_r1:.2f}, R@5 {i2t_r5:.2f}, R@10 {i2t_r10:.2f}\n"
                f"  T2I: R@1 {t2i_r1:.2f}, R@5 {t2i_r5:.2f}, R@10 {t2i_r10:.2f}\n\n"
            )
            
            with open(self.log_file, 'a') as f:
                f.write(log_message)