# src/datamodules/global_feature_datamodule.py

import os
from typing import Optional
import pytorch_lightning as pl
from torch.utils.data import DataLoader, DistributedSampler, Subset
import torch
import hydra
import clip
import torch.distributed as dist
import numpy as np

# 导入新的 Dataset 类
from .flickr30k_dataset import PrototypeGuidedCLIPDataset, create_collate_fn

class PrototypeGuidedCLIPDataModule(pl.LightningDataModule):
    """
    为原型引导CLIP模型优化的DataModule
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # 延迟初始化
        self.image_transform = None
        self.tokenizer = clip.tokenize
        self._collate_fn = None
        
        # 打印初始化信息 (安全检查dist是否可用)
        if dist.is_available() and dist.is_initialized() and dist.get_rank() == 0:
            print("初始化原型引导CLIP数据模块")

    def prepare_data(self):
        pass

    def setup(self, stage: Optional[str] = None):
        """初始化转换和Dataset"""
        # 1. 初始化CLIP预处理 (CPU加载，避免占用GPU显存)
        if self.image_transform is None:
            clip_model_name = self.config.model.get('clip_model_name', 'ViT-B/32')
            _, self.image_transform = clip.load(
                clip_model_name, 
                device="cpu", 
                jit=False
            )
        
        # 2. 创建collate函数
        if self._collate_fn is None:
            self._collate_fn = create_collate_fn(self.tokenizer)
        
        # 3. 创建数据集
        if stage == 'fit' or stage is None:
            full_train_dataset = PrototypeGuidedCLIPDataset(
                config=self.config, 
                split='train',
                clip_preprocess=self.image_transform,
                clip_tokenize=self.tokenizer,
            )
            train_ratio = self.config.dataset.get('train_ratio', 1.0) 
            
            if train_ratio < 1.0:
                total_size = len(full_train_dataset)
                subset_size = int(total_size * train_ratio)
                rng = np.random.RandomState(42) 
                indices = rng.permutation(total_size)
                selected_indices = indices[:subset_size]
                
                # 使用 Subset 包装
                self.train_dataset = Subset(full_train_dataset, selected_indices)
                
                # 打印日志
                if not dist.is_initialized() or dist.get_rank() == 0:
                    print(f"✂️ [Data Scaling] 启用数据缩放: 比例={train_ratio}")
                    print(f"   原始数量: {total_size} -> 采样后: {len(self.train_dataset)}")
            else:
                self.train_dataset = full_train_dataset
                if not dist.is_initialized() or dist.get_rank() == 0:
                     print(f"✅ [Data Scaling] 使用全量数据: {len(self.train_dataset)}")

            self.val_dataset = PrototypeGuidedCLIPDataset(
                config=self.config, 
                split='val',
                clip_preprocess=self.image_transform,
                clip_tokenize=self.tokenizer,
            )
            
            # 安全打印
            if dist.is_available() and dist.is_initialized() and dist.get_rank() == 0:
                print(f"训练集: {len(self.train_dataset)} | 验证集: {len(self.val_dataset)}")
            elif not dist.is_initialized():
                 print(f"训练集: {len(self.train_dataset)} | 验证集: {len(self.val_dataset)}")

        if stage == 'test' or stage is None:
            self.test_dataset = PrototypeGuidedCLIPDataset(
                config=self.config, 
                split='test',
                clip_preprocess=self.image_transform,
                clip_tokenize=self.tokenizer,
            )

    def train_dataloader(self) -> DataLoader:
        """
        训练加载器：使用 DistributedSampler 以支持 DDP
        """
        sampler = None
        # 检查是否使用了分布式训练
        if dist.is_available() and dist.is_initialized():
            sampler = DistributedSampler(self.train_dataset, shuffle=True)
        
        return DataLoader(
            self.train_dataset, 
            batch_size=self.config.dataset.batch_size, 
            # 如果有sampler，shuffle必须为False（由sampler处理）
            # 如果没有sampler（单卡），则需要Shuffle
            shuffle=(sampler is None),
            num_workers=self.config.dataset.num_workers, 
            collate_fn=self._collate_fn, 
            pin_memory=True,
            sampler=sampler,
            drop_last=True 
        )

    def val_dataloader(self) -> DataLoader:
        """
        验证加载器：
        关键修正：配合 System 中的 rank_0 验证逻辑。
        我们需要 Rank 0 看到完整且顺序一致的数据集，不能让 PL 自动切分数据。
        """
        sampler = None
        if dist.is_available() and dist.is_initialized():
            # 这里的trick是：告诉Sampler只有1个副本，这样它就不会切分数据
            # 且指定rank=0，确保随机数种子一致（虽然shuffle=False其实无所谓）
            # 这样所有卡都会拿到完整数据，但我们在 System 里通过 `if rank!=0 return` 让其他卡空转
            sampler = DistributedSampler(
                self.val_dataset, 
                num_replicas=1, # 关键：伪装成单卡
                rank=0,         # 关键：伪装成主卡
                shuffle=False   # 关键：保持顺序以配合 [::5] 切片
            )
            
        return DataLoader(
            self.val_dataset, 
            batch_size=self.config.dataset.batch_size, 
            shuffle=False, # 验证集绝对不能打乱
            num_workers=self.config.dataset.num_workers, 
            collate_fn=self._collate_fn, 
            pin_memory=True,
            sampler=sampler,
            drop_last=False
        )

    def test_dataloader(self) -> DataLoader:
        """测试加载器：逻辑同验证集"""
        sampler = None
        if dist.is_available() and dist.is_initialized():
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

# 兼容性别名
class GlobalFeatureDataModule(PrototypeGuidedCLIPDataModule):
    pass