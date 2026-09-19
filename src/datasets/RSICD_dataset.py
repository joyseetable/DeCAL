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

class RSICDDataset(Dataset):
    """
    适配 RSICD 数据集 (Official Standard Split)
    
    RSICD JSON 标准结构通常为:
    {
      "images": [
        {
          "filename": "airport_1.jpg",
          "imgid": 0,
          "sentences": [{"raw": "desc1..."}, {"raw": "desc2..."}],
          "split": "train"  <-- 依据此字段划分
        },
        ...
      ]
    }
    我们将在此类内部将其转换为扁平化格式以适配原有的训练逻辑。
    """
    def __init__(self, config, split: str, clip_preprocess, clip_tokenize):
        super().__init__()
        self.config = config
        self.split = split
        self.clip_preprocess = clip_preprocess
        self.clip_tokenize = clip_tokenize
        
        # ---------------------------------------------------------------------
        # 修改点 1: 加载 RSICD 统一的大 JSON 文件
        # ---------------------------------------------------------------------
        # 假设 config 中配置了一个指向 dataset_rsicd.json 的路径
        # 你需要在 hydra 配置里将 key 改为对应的 json 路径
        # 例如: annotations_path: /path/to/dataset_rsicd.json
        annotations_path = hydra.utils.to_absolute_path(
            self.config.dataset.get("annotations_path", "/data/clc/data/RSICD/dataset_rsicd.json")
        )
        
        if not os.path.exists(annotations_path):
             raise FileNotFoundError(f"RSICD 标注文件未找到: {annotations_path}")

        if not dist.is_initialized() or dist.get_rank() == 0:
            print(f"正在加载 RSICD 标注文件: {annotations_path} ...")
            
        with open(annotations_path, 'r', encoding='utf-8') as f:
            full_data = json.load(f)
            
        # 兼容处理：有些 JSON 根节点直接是 list，有些是 dict 包含 'images'
        images_data = full_data.get('images', full_data) if isinstance(full_data, dict) else full_data

        # ---------------------------------------------------------------------
        # 修改点 2: 依据 split 筛选并展平数据 (Flattening)
        # ---------------------------------------------------------------------
        # RSICD 中: split 取值为 'train', 'val', 'test'
        target_split = split 
        # 有些数据集标注可能用 'restval' 代替 'val'，这里做个简单映射，如果 RSICD 标准版通常就是 'val'
        if split == 'val': 
            target_split = 'val' 
        
        self.annotations = []
        
        for img_item in images_data:
            # 1. 检查 split 是否匹配
            if img_item.get('split') != target_split:
                continue
            
            # 2. 获取基本信息
            filename = img_item['filename']
            img_id = img_item.get('imgid', 0) # RSICD通常有imgid
            
            # 3. 遍历该图片下的所有句子，展平为 (图, 句) 对
            for sent in img_item['sentences']:
                raw_caption = sent['raw']
                
                # 构建与原有 COCO 格式兼容的字典
                self.annotations.append({
                    "filename": filename,
                    "caption": raw_caption,
                    "imageid": img_id
                })

        # ---------------------------------------------------------------------
        # 修改点 3: 图片路径设置
        # ---------------------------------------------------------------------
        # RSICD 图片通常都在一个大文件夹里，不分 train/val 文件夹
        self.image_dir_base = self.config.dataset.get("image_dir", "/data/clc/data/RSICD/RSICD_images")

        # 4. 统计信息
        if not dist.is_initialized() or dist.get_rank() == 0:
            print(f"数据集: {split}, 筛选后样本数: {len(self.annotations)} (来自 {len(set(x['filename'] for x in self.annotations))} 张图片)")
    
    def __len__(self) -> int:
        return len(self.annotations)
    
    def __getitem__(self, index: int) -> dict:
        # 直接获取图文对数据
        ann = self.annotations[index]
        
        filename = ann['filename']
        caption = ann['caption']
        image_id = ann.get('imageid', 0)
        
        # 拼接图片完整路径
        # image_path = os.path.join(self.image_dir_base, filename)
        base_path = os.path.join(self.image_dir_base, filename)
        
        # 1. 如果原文件名路径存在，直接用
        if os.path.exists(base_path):
            image_path = base_path
        else:
            # 2. 如果不存在，尝试替换后缀名查找
            # 获取无后缀的文件名 (例如: "image_01.jpg" -> "image_01")
            file_stem = os.path.splitext(filename)[0]
            
            # 常见的备选后缀
            potential_paths = [
                os.path.join(self.image_dir_base, file_stem + ".tif"),
                os.path.join(self.image_dir_base, file_stem + ".jpg"),
                os.path.join(self.image_dir_base, file_stem + ".png")
            ]
            
            image_path = base_path # 默认回退到原始路径
            for p in potential_paths:
                if os.path.exists(p):
                    image_path = p
                    break
        try:
            # 加入 .convert("RGB") 以兼容可能存在的单通道或 TIF 图像
            image_pil = Image.open(image_path).convert("RGB")
        except Exception as e:
            # 打印错误但不要中断训练，返回None由collate_fn过滤
            print(f"无法加载图像 {image_path}: {e}")
            return {
                "image": None,
                "caption": "",
                "filename": filename,
                "image_path": image_path,
                "image_id": image_id,
                "index": index
            }
        
        # 1. 图像预处理 (CLIP Transform)
        image_tensor = self.clip_preprocess(image_pil)
        
        # 2. 返回数据 (保持原有格式完全不变)
        return {
            "image": image_tensor,          # [3, 224, 224]
            "caption": caption,             # 原始文本字符串
            "filename": filename,           # 文件名
            "image_path": image_path,       # 完整路径
            "image_id": image_id,           # 图片ID
            "index": index                  # 数据索引
        }

# Collate Function 保持不变
def create_collate_fn(clip_tokenize):
    def collate_fn(batch):
        # 1. 过滤掉加载失败的样本 (image is None)
        batch = [b for b in batch if b["image"] is not None and b["caption"] is not None]
        
        if len(batch) == 0:
            return None
        
        # 2. 提取图像并堆叠
        images = torch.stack([b["image"] for b in batch])
        
        # 3. 提取文本并批量tokenize
        captions = [b["caption"] for b in batch]
        text_tokens = clip_tokenize(captions, truncate=True)
        
        # 4. 提取元数据
        filenames = [b["filename"] for b in batch]
        image_paths = [b["image_path"] for b in batch]
        
        # 提取 image_id 并转换为 Tensor
        image_ids = [b["image_id"] for b in batch]
        image_ids_tensor = torch.tensor(image_ids, dtype=torch.long)
        
        return {
            "images": images,                # [batch_size, 3, 224, 224]
            "text_tokens": text_tokens,      # [batch_size, seq_len]
            "filenames": filenames,          # list of strings
            "image_paths": image_paths,      # list of strings
            "captions": captions,            # list of strings
            "image_id": image_ids_tensor     # [batch_size] (Tensor)
        }
    
    return collate_fn

# DataLoader 创建函数保持不变
def create_data_loader(config, split: str, batch_size: int, 
                      clip_model_name: str = 'ViT-B/32', 
                      num_workers: int = 4,
                      shuffle: bool = None):
    
    _, clip_preprocess = clip.load(clip_model_name, device='cpu')
    tokenizer = clip.tokenize
    
    dataset = RSICDDataset(
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