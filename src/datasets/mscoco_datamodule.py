# src/datamodules/global_feature_datamodule.py

import os
from typing import Optional
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import torch
import hydra
import clip
import torch.distributed as dist
from torch.utils.data import DistributedSampler, Subset
import numpy as np

# -----------------------------------------------------------
# 修改点 1: 确保导入的是你上一轮修改过的那个文件中的 Dataset 类
# 假设你上一轮的代码保存在 src/datamodules/flickr30k_dataset2.py 中
# -----------------------------------------------------------
from .mscoco_dataset import MSCOCODataset, create_collate_fn

class MSCOCODatamodule(pl.LightningDataModule):
    """
    适配扁平化 COCO 数据集的 DataModule
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.image_transform = None
        self.tokenizer = clip.tokenize
        self._collate_fn = None
        
        if dist.is_initialized() and dist.get_rank() == 0:
            print("初始化原型引导CLIP数据模块 (COCO Flat Version)")

    def prepare_data(self):
        pass

    def setup(self, stage: Optional[str] = None):
        """
        初始化模型相关的转换和Dataset
        """
        # 1. 初始化CLIP的预处理函数
        if self.image_transform is None:
            clip_model_name = self.config.model.get('clip_model_name', 'ViT-B/32')
            # 指定 device='cpu' 是为了防止多进程加载数据时在子进程初始化 CUDA
            _, self.image_transform = clip.load(
                clip_model_name, 
                device="cpu", 
                jit=False
            )
        
        # 2. 创建collate函数
        if self._collate_fn is None:
            self._collate_fn = create_collate_fn(self.tokenizer)
        
        # 3. 创建数据集
        # 注意：这里的 split 参数会直接影响 Dataset 读取 config 中的哪个 key
        # 例如 split='train' -> 读取 config.dataset.train_annotations_path
        
        if stage == 'fit' or stage is None:
            full_train_dataset = MSCOCODataset(
                config=self.config, 
                split='train',
                clip_preprocess=self.image_transform,
                clip_tokenize=self.tokenizer
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
            self.val_dataset = MSCOCODataset(
                config=self.config, 
                split='val',
                clip_preprocess=self.image_transform,
                clip_tokenize=self.tokenizer
            )
            
            if not dist.is_initialized() or dist.get_rank() == 0:
                print(f"训练集大小: {len(self.train_dataset)}, 验证集大小: {len(self.val_dataset)}")
            
        if stage == 'test' or stage is None:
            # 如果你的 config 中没有 test_annotations_path，通常测试也是用验证集
            # 这里假设 config 中定义了 test_annotations_path，或者你可以复用 'val'
            test_split = 'test' if 'test_annotations_path' in self.config.dataset else 'val'
            
            self.test_dataset = MSCOCODataset(
                config=self.config, 
                split=test_split,
                clip_preprocess=self.image_transform,
                clip_tokenize=self.tokenizer
            )
            if not dist.is_initialized() or dist.get_rank() == 0:
                print(f"测试集大小: {len(self.test_dataset)} (Split: {test_split})")

    def train_dataloader(self) -> DataLoader:
        sampler = None
        if dist.is_initialized():
            sampler = DistributedSampler(self.train_dataset, shuffle=True)
        
        return DataLoader(
            self.train_dataset, 
            batch_size=self.config.dataset.batch_size, 
            # 如果有 sampler，shuffle 必须为 False
            shuffle=(sampler is None),
            num_workers=self.config.dataset.num_workers, 
            collate_fn=self._collate_fn, 
            pin_memory=True,
            sampler=sampler,
            drop_last=True  # 训练时丢弃最后一个不完整的batch是OK的
        )

    def val_dataloader(self) -> DataLoader:
        sampler = None
        if dist.is_initialized():
            # Trick: 伪装成单卡模式，让所有卡都验证完整数据集
            # 只有当你需要在 validation_step 中手动处理多卡聚合，或者
            # 你在 LightningModule 中通过 if rank==0 控制打印时才这么做。
            # 如果你的计算资源有限，建议让 DistributedSampler 正常切分 (num_replicas=None)。
            sampler = DistributedSampler(
                self.val_dataset, 
                num_replicas=1, # 全量数据给每张卡
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
            # -----------------------------------------------------------
            # 修改点 2: 验证集 drop_last 必须为 False，否则指标计算不准
            # -----------------------------------------------------------
            drop_last=False 
        )

    def test_dataloader(self) -> DataLoader:
        sampler = None
        if dist.is_initialized():
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
            # -----------------------------------------------------------
            # 修改点 3: 测试集 drop_last 必须为 False
            # -----------------------------------------------------------
            drop_last=False
        )