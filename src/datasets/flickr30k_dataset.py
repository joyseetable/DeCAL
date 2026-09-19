import os
import json
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
import hydra
from tqdm import tqdm
import clip
import torch.distributed as dist

class PrototypeGuidedCLIPDataset(Dataset):
    """
    为原型引导CLIP模型优化的数据集
    """
    def __init__(self, config, split: str, clip_preprocess, clip_tokenize):
        super().__init__()
        self.config = config
        self.split = split
        self.clip_preprocess = clip_preprocess
        self.clip_tokenize = clip_tokenize
        
        # 1. 加载标注文件 
        # [新增: 动态兼容原版 .json 和增强版 .jsonl]
        annotations_path = hydra.utils.to_absolute_path(self.config.dataset.get(f"{split}_annotations_path"))
        
        self.dataset_images = []
        if annotations_path.endswith('.jsonl'):
            # 读取我们离线生成的增强 jsonl
            with open(annotations_path, 'r', encoding='utf-8') as f:
                for line in f:
                    self.dataset_images.append(json.loads(line.strip()))
        else:
            # 兼容读取原始 Karpathy json (供 Val/Test 使用)
            with open(annotations_path, 'r', encoding='utf-8') as f:
                self.dataset_images = json.load(f)['images']
                
        self.image_dir_base = self.config.dataset.get("image_dir", "./data/flickr30k/images")
        
        # 2. 建立(image_idx, caption_idx)索引列表
        self.ids = []
        for i, d in enumerate(self.dataset_images):
            # [新增: 兼容新旧格式的 Caption 数量读取]
            num_captions = len(d.get('original_captions', d.get('sentences', [])))
            self.ids.extend([(i, j) for j in range(num_captions)])
        
        # 3. 统计信息
        if not dist.is_initialized() or dist.get_rank() == 0:
            print(f"数据集: {split}, 图像总数: {len(self.dataset_images)}, 总图文对数: {len(self.ids)}")
        
        # 4. 缓存所有图像文件名
        self.filenames = [img.get('filename', '') for img in self.dataset_images]
        
    
    def __len__(self) -> int:
        return len(self.ids)
    
    def __getitem__(self, index: int) -> dict:
        image_idx, caption_idx = self.ids[index]
        image_info = self.dataset_images[image_idx]
        filename = image_info.get('filename', '')
        
        # [新增: 兼容新旧格式的文本读取]
        if 'original_captions' in image_info:
            caption = image_info['original_captions'][caption_idx]
        else:
            caption = image_info['sentences'][caption_idx]['raw']
        
        # 图像路径
        image_path = os.path.join(self.image_dir_base, filename)
        
        # [新增: 提取 PIHM-Next 增强数据 (仅在增强集存在时生效)]
        triplet_entities = []
        hard_negatives = []
        if 'triplets' in image_info:
            # 将该图像所有三元组的 entity 提取出来作为局部查询 Query
            triplet_entities = [t['entity'] for t in image_info['triplets'] if 'entity' in t]
            # triplet_entities = image_info.get('triplets', [])
            # 容错：如果 LLM 没提取出实体，给个全局占位符防报错
            if len(triplet_entities) == 0:
                triplet_entities = ["object"]
                
        if 'hard_negatives' in image_info:
            # 将所有难负样本句子提取成列表
            hard_negatives = [v for k, v in image_info['hard_negatives'].items() if isinstance(v, str) and len(v) > 0]

        try:
            image_pil = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"无法加载图像 {image_path}: {e}")
            return {
                "image": None,
                "caption": None,
                "filename": filename,
                "image_path": image_path,
                "triplet_entities": [],
                "hard_negatives": []
            }
        
        # 图像预处理
        image_tensor = self.clip_preprocess(image_pil)
        
        # 返回数据字典
        return {
            "image": image_tensor,          
            "caption": caption,             
            "filename": filename,           
            "image_path": image_path,       
            "image_idx": image_idx,         
            "caption_idx": caption_idx,     
            # [新增: 将增强数据传入 collate_fn]
            "triplet_entities": triplet_entities, 
            "hard_negatives": hard_negatives      
        }


def create_collate_fn(clip_tokenize):
    """
    创建数据集的collate函数
    """
    def collate_fn(batch):
        # 过滤掉无效样本
        batch = [b for b in batch if b["image"] is not None and b["caption"] is not None]
        
        if len(batch) == 0:
            return None
        
        # 提取图像
        images = torch.stack([b["image"] for b in batch])
        
        # 提取文本并批量tokenize (全局正样本)
        captions = [b["caption"] for b in batch]
        text_tokens = clip_tokenize(captions, truncate=True)
        
        filenames = [b["filename"] for b in batch]
        image_paths = [b["image_path"] for b in batch]
        
        # [新增: 收集实体列表和负样本列表]
        # 注意：因为每张图的实体和负样本数量不等长，所以这里以 List[List[str]] 的形式原样返回。
        # 这样不会破坏原有的 Tensor 结构，留给后续模型 Forward 或 Loss 时再去动态 Tokenize。
        triplet_entities_list = [b["triplet_entities"] for b in batch]
        hard_negatives_list = [b["hard_negatives"] for b in batch]
        
        return {
            "images": images,                
            "text_tokens": text_tokens,      
            "filenames": filenames,          
            "image_paths": image_paths,      
            "captions": captions,            
            # [新增: 挂载到最终的 batch 字典中]
            "triplet_entities": triplet_entities_list,
            "hard_negatives": hard_negatives_list
        }

    return collate_fn


def create_data_loader(config, split: str, batch_size: int, 
                      clip_model_name: str = 'ViT-B/32', 
                      num_workers: int = 4,
                      shuffle: bool = None):
    """
    创建数据加载器
    """
    # 加载CLIP获取 transform (不需要加载权重，但CLIP API比较死板)
    # 建议: device='cpu' 避免占用显存
    # jit=False 保持一致性
    _, clip_preprocess = clip.load(clip_model_name, device='cpu', jit=False)
    
    # 创建tokenize函数
    tokenizer = clip.tokenize
    
    # 创建数据集
    dataset = PrototypeGuidedCLIPDataset(
        config=config,
        split=split,
        clip_preprocess=clip_preprocess,
        clip_tokenize=tokenizer,
    )
    
    # 设置shuffle: 训练打乱，验证/测试不打乱 (关键!)
    if shuffle is None:
        shuffle = (split == 'train')
    
    # 创建collate函数
    collate_fn = create_collate_fn(tokenizer)
    
    # 创建数据加载器
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True, # 建议开启，加速 Host 到 Device 传输
        drop_last=(split == 'train')  # 训练时丢弃最后一个不完整的batch，稳定BatchNorm(如果有)
    )
    
    return dataloader