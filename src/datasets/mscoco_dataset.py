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

class MSCOCODataset(Dataset):
    """
    适配大模型增强后的 JSONL 数据集类 (内存级自动展平版 590k)
    """
    def __init__(self, config, split: str, clip_preprocess, clip_tokenize):
        super().__init__()
        self.config = config
        self.split = split
        self.clip_preprocess = clip_preprocess
        self.clip_tokenize = clip_tokenize
        
        annotations_path = hydra.utils.to_absolute_path(self.config.dataset.get(f"{split}_annotations_path"))
        
        if not os.path.exists(annotations_path):
             raise FileNotFoundError(f"标注文件未找到: {annotations_path}")

        print(f"正在加载标注文件: {annotations_path} ...")
        self.annotations = []
        
        # ==========================================
        # 🚀 核心修改：在内存中暴力展平数据！
        # 把 118k 的图像级数据，重新炸开成 590k 的图文对数据
        # ==========================================
        with open(annotations_path, 'r', encoding='utf-8') as f:
            if annotations_path.endswith('.jsonl'):
                for line in f:
                    if not line.strip(): continue
                    data = json.loads(line)
                    
                    # 提取该图片共享的实体列表 (提取一次，造福 5 句话)
                    triplets = data.get('triplets', [])
                    entities = [t.get('entity') for t in triplets if t.get('entity')]
                    if len(entities) == 0:
                        entities = ["object"]

                    # 把打包的 5 句话炸开，独立变成 5 条样本
                    captions = data.get('original_captions', [data.get('caption', '')])
                    for idx, cap in enumerate(captions):
                        # 如果有增强版的 anchor，可以选择只替换第一句话
                        final_caption = data.get('augmented_anchor', cap) if idx == 0 else cap
                        
                        self.annotations.append({
                            'filename': data.get('filename', ''),
                            'image_id': data.get('image_id', data.get('imageid', 0)),
                            'caption': final_caption,
                            'entities': entities  # 共享大模型提取的实体
                        })
            else:
                # 兼容原始的 val_flattened.json (已经是展平的)
                raw_data = json.load(f)
                for item in raw_data:
                    self.annotations.append({
                        'filename': item.get('filename', ''),
                        'image_id': item.get('image_id', item.get('imageid', 0)),
                        'caption': item.get('caption', ''),
                        'entities': ["object"] # 验证集没有三元组，兜底
                    })

        if split == "train":
            self.image_dir_base = self.config.dataset.get("image_dir_base_train", "/data/clc/data/mscoco/train2017")
        else:
            self.image_dir_base = self.config.dataset.get("image_dir_base_val", "/data/clc/data/mscoco/val2017")

        if not dist.is_initialized() or dist.get_rank() == 0:
            print(f"数据集: {split}, [展平后] 实际加载样本数: {len(self.annotations)}")
    
    def __len__(self) -> int:
        return len(self.annotations)
    
    def __getitem__(self, index: int) -> dict:
        # 因为在 __init__ 里已经展平，这里的逻辑回归到最纯粹极简的状态
        ann = self.annotations[index]
        
        filename = ann['filename']
        image_id = ann['image_id']
        caption = ann['caption']
        entities = ann['entities']
            
        image_path = os.path.join(self.image_dir_base, filename)
        
        try:
            image_pil = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"无法加载图像 {image_path}: {e}")
            return {
                "image": None,
                "caption": caption,
                "filename": filename,
                "image_path": image_path,
                "entities": entities
            }
        
        image_tensor = self.clip_preprocess(image_pil)
        
        return {
            "image": image_tensor,          
            "caption": caption,             
            "filename": filename,           
            "image_path": image_path,       
            "image_id": image_id,           
            "triplet_entities": entities,           
            "index": index                  
        }

def create_collate_fn(clip_tokenize):
    """
    创建数据集的collate函数
    """
    def collate_fn(batch):
        # 1. 过滤掉加载失败的样本
        batch = [b for b in batch if b["image"] is not None]
        
        if len(batch) == 0:
            return None
        
        # 2. 提取图像并堆叠
        images = torch.stack([b["image"] for b in batch])
        
        # 3. 提取文本并批量tokenize (针对全局文本)
        captions = [b["caption"] for b in batch]
        text_tokens = clip_tokenize(captions, truncate=True)
        
        # 4. 提取元数据
        filenames = [b["filename"] for b in batch]
        image_paths = [b["image_path"] for b in batch]
        
        image_ids = [b["image_id"] for b in batch]
        image_ids_tensor = torch.tensor(image_ids, dtype=torch.long)
        
        # 🚀 5. 提取实体列表打包 (不在这里 tokenize，因为你的模型内会处理)
        triplet_entities = [b["triplet_entities"] for b in batch]
        
        return {
            "images": images,                
            "text_tokens": text_tokens,      
            "filenames": filenames,          
            "image_paths": image_paths,      
            "captions": captions,            
            "image_id": image_ids_tensor,    
            "triplet_entities": triplet_entities  # 🚀 直接透传给模型 forward
        }
    
    return collate_fn

def create_data_loader(config, split: str, batch_size: int, 
                      clip_model_name: str = 'ViT-B/32', 
                      num_workers: int = 4,
                      shuffle: bool = None):
    
    _, clip_preprocess = clip.load(clip_model_name, device='cpu')
    tokenizer = clip.tokenize
    
    dataset = MSCOCODataset(
        config=config,
        split=split,
        clip_preprocess=clip_preprocess,
        clip_tokenize=tokenizer,
    )
    
    if shuffle is None:
        shuffle = (split == 'train')
    
    collate_fn = create_collate_fn(tokenizer)
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=(split == 'train')
    )
    
    return dataloader