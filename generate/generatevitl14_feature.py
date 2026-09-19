import os
import json
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import clip
from tqdm import tqdm

# --- 配置参数 ---
# !!! 请根据你的实际路径修改以下变量 !!!
FLICKR30K_ANNOTATIONS_PATH = "/data/clc/APSE-IPIK/NEW_APSEIPIK/DATA/flickr30k/annotations/train.json"
FLICKR30K_IMAGE_ROOT = "/data/clc/APSE-IPIK/NEW_APSEIPIK/DATA/flickr30k/images"
OUTPUT_NPY_PATH = "teacher_feats_vitl14.npy"
TEACHER_MODEL_NAME = "ViT-L/14"
BATCH_SIZE = 64
NUM_WORKERS = 8
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# --------------------

class ImageFilenameDataset(Dataset):
    """
    一个简单的 Dataset，用于加载图片和返回对应的文件名，
    以便于将特征映射回原始图片。
    """
    def __init__(self, image_filenames, image_root, preprocess):
        self.image_filenames = image_filenames
        self.image_root = image_root
        self.preprocess = preprocess

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        filename = self.image_filenames[idx]
        img_path = os.path.join(self.image_root, filename)
        
        try:
            # 1. 加载和预处理图片
            image = Image.open(img_path).convert("RGB")
            image_tensor = self.preprocess(image)
        except Exception as e:
            print(f"Warning: Skipping {filename} due to error: {e}")
            # 返回None，让 collate_fn 过滤掉
            return None 

        # 2. 返回预处理后的 Tensor 和文件名
        return image_tensor, filename

def collate_fn(batch):
    """自定义 Collate 函数，用于过滤掉无效样本 (None)"""
    batch = [item for item in batch if item is not None]
    if not batch:
        return None, None
    
    images = torch.stack([item[0] for item in batch])
    filenames = [item[1] for item in batch]
    
    return images, filenames

def get_image_list_from_json(annotations_path):
    """从 Flickr30k JSON 文件中提取所有图片文件名"""
    print(f"--- 1. Reading image list from {annotations_path} ---")
    try:
        with open(annotations_path, 'r') as f:
            data = json.load(f)
        
        # 确保JSON结构正确
        if 'images' in data:
            filenames = [img_info['filename'] for img_info in data['images']]
            print(f"Found {len(filenames)} unique images.")
            return filenames
        else:
            raise ValueError("JSON structure missing 'images' key.")
            
    except FileNotFoundError:
        print(f"Error: Annotation file not found at {annotations_path}")
        return []
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in {annotations_path}")
        return []


def extract_features():
    """执行特征提取的主函数"""
    
    # 1. 加载 CLIP Teacher 模型
    print(f"--- 2. Loading CLIP Teacher Model ({TEACHER_MODEL_NAME}) to {DEVICE} ---")
    model, preprocess = clip.load(TEACHER_MODEL_NAME, device=DEVICE)
    model.eval()
    
    # 2. 获取图片文件列表
    image_filenames = get_image_list_from_json(FLICKR30K_ANNOTATIONS_PATH)
    if not image_filenames:
        print("Feature extraction aborted.")
        return

    # 3. 初始化 DataLoader
    dataset = ImageFilenameDataset(image_filenames, FLICKR30K_IMAGE_ROOT, preprocess)
    dataloader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=NUM_WORKERS, 
        collate_fn=collate_fn,
        pin_memory=True if DEVICE == 'cuda' else False
    )

    # 4. 执行特征提取
    print("--- 3. Starting feature extraction ---")
    all_features = {}
    
    with torch.no_grad():
        for images, filenames in tqdm(dataloader, desc="Extracting ViT-L/14 Features"):
            if images is None: continue # 跳过空批次
            
            images = images.to(DEVICE)
            
            # 使用 encode_image 提取特征
            features = model.encode_image(images)
            
            # 标准 CLIP 实践：特征 L2 归一化
            features /= features.norm(dim=-1, keepdim=True)
            
            # 将特征从 GPU 移到 CPU 并转为 NumPy 数组
            features_np = features.cpu().numpy()
            
            # 将特征存储到字典中
            for filename, feature in zip(filenames, features_np):
                all_features[filename] = feature

    # 5. 保存结果
    print(f"--- 4. Saving {len(all_features)} features to {OUTPUT_NPY_PATH} ---")
    # 使用 allow_pickle=True 来保存包含复杂对象（如字典）的 NumPy 文件
    np.save(OUTPUT_NPY_PATH, all_features, allow_pickle=True)
    print("Extraction complete. Feature file ready for distillation.")

if __name__ == '__main__':
    extract_features()