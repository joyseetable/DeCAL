# src/datamodules/global_feature_datamodule.py

import os
from typing import Optional
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import torch
import hydra
import clip
import torch.distributed as dist
from torch.utils.data import DistributedSampler

# -----------------------------------------------------------
# 修改点 1: 导入上一轮修改后的 RSICDDataset 类
# 注意：假设你上一步的文件名为 dataset.py，如果是其他名字请修改 from .xxx
# -----------------------------------------------------------
from .RSICD_dataset import RSICDDataset, create_collate_fn

class RSICDDatamodule(pl.LightningDataModule):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.image_transform = None
        self.tokenizer = clip.tokenize
        self._collate_fn = None
        
        if dist.is_initialized() and dist.get_rank() == 0:
            print("初始化 RSICD CLIP 数据模块")

    def prepare_data(self):
        # RSICD 数据通常是预先下载好的，此处留空或用于解压
        pass

    def setup(self, stage: Optional[str] = None):
        """
        初始化模型相关的转换和 Dataset
        """
        # 1. 初始化 CLIP 的预处理函数 (Image Transform)
        if self.image_transform is None:
            clip_model_name = self.config.model.get('clip_model_name', 'ViT-B/32')
            # 指定 device='cpu' 是为了防止多进程加载数据时在子进程初始化 CUDA 上下文
            _, self.image_transform = clip.load(
                clip_model_name, 
                device="cpu", 
                jit=False
            )
        
        # 2. 创建 collate 函数 (保持原有的批处理逻辑)
        if self._collate_fn is None:
            self._collate_fn = create_collate_fn(self.tokenizer)
        
        # 3. 创建数据集
        # RSICDDataset 内部会根据 split 参数 ('train', 'val', 'test') 
        # 从同一个 JSON 文件中筛选数据，所以这里只需传入对应的 split 字符串
        
        if stage == 'fit' or stage is None:
            self.train_dataset = RSICDDataset(
                config=self.config, 
                split='train',
                clip_preprocess=self.image_transform,
                clip_tokenize=self.tokenizer
            )
            
            self.val_dataset = RSICDDataset(
                config=self.config, 
                split='test',
                clip_preprocess=self.image_transform,
                clip_tokenize=self.tokenizer
            )
            
            if not dist.is_initialized() or dist.get_rank() == 0:
                print(f"[Setup] 训练集: {len(self.train_dataset)}, 验证集: {len(self.val_dataset)}")
            
        if stage == 'test' or stage is None:
            # 指定 split='test' 即可，Dataset 类会自动处理筛选
            self.test_dataset = RSICDDataset(
                config=self.config, 
                split='test',
                clip_preprocess=self.image_transform,
                clip_tokenize=self.tokenizer
            )
            if not dist.is_initialized() or dist.get_rank() == 0:
                print(f"[Setup] 测试集: {len(self.test_dataset)}")

    def train_dataloader(self) -> DataLoader:
        sampler = None
        if dist.is_initialized():
            sampler = DistributedSampler(self.train_dataset, shuffle=True)
        
        return DataLoader(
            self.train_dataset, 
            batch_size=self.config.dataset.batch_size, 
            # 如果有 sampler (DDP模式)，shuffle 必须为 False
            shuffle=(sampler is None),
            num_workers=self.config.dataset.num_workers, 
            collate_fn=self._collate_fn, 
            pin_memory=True,
            sampler=sampler,
            drop_last=True  # 训练时丢弃不完整 batch 以保持 tensor 维度一致
        )

    def val_dataloader(self) -> DataLoader:
        sampler = None
        if dist.is_initialized():
            # 保持你原有的逻辑：
            # 让每张卡都验证完整数据集 (num_replicas=1, rank=0)。
            # RSICD 数据集较小 (~1万张图)，这样做不会爆内存，
            # 且能避免在 validation_step 中进行复杂的 all_gather 操作。
            sampler = DistributedSampler(
                self.val_dataset, 
                num_replicas=1, 
                rank=0,         
                shuffle=False
            )
            
        return DataLoader(
            self.val_dataset, 
            batch_size=self.config.dataset.batch_size, 
            shuffle=False,
            num_workers=self.config.dataset.num_workers, 
            collate_fn=self._collate_fn, 
            pin_memory=True,
            sampler=sampler,
            drop_last=False # 验证集必须保留所有数据
        )

    def test_dataloader(self) -> DataLoader:
        sampler = None
        if dist.is_initialized():
            # 同验证集逻辑，确保测试指标覆盖所有样本
            sampler = DistributedSampler(
                self.test_dataset, 
                num_replicas=1, 
                rank=0,         
                shuffle=False   
            )
            
        return DataLoader(
            self.test_dataset, 
            batch_size=self.config.dataset.batch_size,
            shuffle=False, 
            num_workers=self.config.dataset.num_workers, 
            collate_fn=self._collate_fn, 
            pin_memory=True,
            sampler=sampler,
            drop_last=False
        )