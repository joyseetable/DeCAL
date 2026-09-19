import sys
import os
import torch
import torch.nn.functional as F
from PIL import Image
import hydra
from omegaconf import DictConfig, OmegaConf
import json
import clip
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# 假设你的系统和模型定义在这个路径
from src.systems.retrieval_system import DistillSystem

def load_rsitmd_from_json(json_file, image_root, split='test'):
    """
    从 RSITMD 的 JSON 文件加载数据
    RSITMD 的 json 通常包含所有数据，通过 'split' 字段区分 train/val/test
    """
    print(f"正在加载 RSITMD 数据集，目标 split: {split} ...")
    with open(json_file, 'r') as f:
        data = json.load(f)["images"]
    
    dataset = []
    
    for item in data:
        # 过滤 split (train, val, test)
        if item.get('split') != split:
            continue
            
        filename = item.get('filename', '')
        image_path = os.path.join(image_root, filename)
        
        sentences = item.get('sentences', [])
        # 提取所有描述的原始文本
        captions = [sentence['raw'] for sentence in sentences]
        
        # 检查图片是否存在以及描述数量
        if os.path.exists(image_path):
            if len(captions) >= 5:
                dataset.append((image_path, captions[:5]))  # 确保取前5个描述
            else:
                # RSITMD 有些图片描述可能少于5个，这里做个填充或跳过，通常RSITMD都是5个
                print(f"警告: 图像 {filename} 描述不足5个，已跳过")
        else:
            # RSITMD 图片通常是 .tif 格式，如果路径不对可能需要检查后缀
            print(f"跳过图像 {filename}: 文件不存在 (路径: {image_path})")
    
    print(f"从 {json_file} ({split}集) 加载了 {len(dataset)} 个样本")
    return dataset

def get_text_features(model, dataset, device, batch_size=128):
    """获取文本特征 (逻辑保持不变)"""
    feature_list = []
    
    # 收集所有文本
    all_captions = []
    for img_path, captions in dataset:
        all_captions.extend(captions[:5]) 
    
    print(f"总文本数量: {len(all_captions)}")
    
    with torch.no_grad():
        for i in range(0, len(all_captions), batch_size):
            end_idx = min(i + batch_size, len(all_captions))
            text_batch = all_captions[i:end_idx]
            
            text_tokens = clip.tokenize(text_batch, truncate=True).to(device)
            text_features = model.encode_text(text_tokens)
            
            text_features = text_features.float().cpu()
            feature_list.append(text_features)
            
    text_features = torch.cat(feature_list, dim=0)
    return text_features

def get_image_features(model, preprocess, dataset, device, batch_size=128):
    """获取图像特征 (逻辑保持不变)"""
    img_feature_list = []
    
    with torch.no_grad():
        # 这里为了演示方便，没有用DataLoader，直接逐个处理可能会慢，
        # 既然你之前的脚本是逐个处理的，这里保持一致，但建议加上 batch 处理以加速
        batch_imgs = []
        
        for i, (img_path, captions) in enumerate(tqdm(dataset, desc="提取图像特征")):
            try:
                # RSITMD 可能是 TIFF 格式，convert('RGB') 很重要
                image = Image.open(img_path).convert('RGB')
                image_input = preprocess(image)
                batch_imgs.append(image_input)
                
                # 当凑够一个 batch 或者 是最后一个时进行推理
                if len(batch_imgs) == batch_size or i == len(dataset) - 1:
                    img_tensor = torch.stack(batch_imgs).to(device)
                    img_feature = model.encode_image(img_tensor)
                    img_feature = img_feature.float().cpu()
                    img_feature_list.append(img_feature)
                    batch_imgs = [] # 清空 batch
                    
            except Exception as e:
                print(f"处理图像 {img_path} 时出错: {e}")
                continue

    img_features = torch.cat(img_feature_list, dim=0)
    return img_features

def evaluate_metrics(similarity, num_images):
    """统一计算 R@1, R@5, R@10"""
    # Image -> Text
    i2t_r1, i2t_r5, i2t_r10 = 0, 0, 0
    for i in range(num_images):
        pred = similarity[i]
        # 对应的5个文本索引范围
        true_indices = set(range(i * 5, (i + 1) * 5))
        
        # 获取排序后的索引
        indices = pred.argsort(descending=True)
        
        # Check R@1
        if indices[0].item() in true_indices: i2t_r1 += 1
        
        # Check R@5
        # 只要前5个预测里有一个属于该图片的文本即可
        top5 = indices[:5].tolist()
        if any(idx in true_indices for idx in top5): i2t_r5 += 1
            
        # Check R@10
        top10 = indices[:10].tolist()
        if any(idx in true_indices for idx in top10): i2t_r10 += 1

    i2t_r1 /= num_images
    i2t_r5 /= num_images
    i2t_r10 /= num_images

    # Text -> Image
    t2i_r1, t2i_r5, t2i_r10 = 0, 0, 0
    similarity_t2i = similarity.T
    num_texts = num_images * 5
    
    for i in range(num_texts):
        pred = similarity_t2i[i]
        true_img_idx = i // 5
        
        indices = pred.argsort(descending=True)
        
        if indices[0].item() == true_img_idx: t2i_r1 += 1
        if true_img_idx in indices[:5].tolist(): t2i_r5 += 1
        if true_img_idx in indices[:10].tolist(): t2i_r10 += 1

    t2i_r1 /= num_texts
    t2i_r5 /= num_texts
    t2i_r10 /= num_texts

    return (i2t_r1*100, i2t_r5*100, i2t_r10*100), (t2i_r1*100, t2i_r5*100, t2i_r10*100)

# 保持你的 OmegaConf Resolver
def get_center_path(clip_name):
    # 基础路径
    base_dir = "/data/clc/APSE-IPIK/NEW_APSEIPIK/offline/clustering_results"
    if "ViT-B" in clip_name:
        return "/data/clc/APSE-IPIK/NEW_APSEIPIK/offline/RSI_clustering_results/cluster_centers_vitb32_RSI_32.npy"
        # return "/data/clc/APSE-IPIK/NEW_APSEIPIK/offline/clustering_results/cluster_centers_160.npy"
    elif "ViT-L" in clip_name:
        return "/data/clc/APSE-IPIK/NEW_APSEIPIK/offline/clustering_results/cluster_centers_vitl14_160.npy"
    else:
        return "/data/clc/APSE-IPIK/NEW_APSEIPIK/offline/clustering_results/cluster_centers_160.npy"

# 注册 resolver
if not OmegaConf.has_resolver("select_path"):
    OmegaConf.register_new_resolver("select_path", get_center_path)

@hydra.main(config_path="../configs", config_name="eval", version_base=None)
def evaluate_rsitmd(config: DictConfig):
    print("=== RSITMD 遥感数据集评估 (Test Split) ===")
    print(OmegaConf.to_yaml(config))
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")
    
    # ================= 配置路径 =================
    # 请修改为 RSITMD 的实际路径
    # RSITMD 的 json 通常叫 dataset_RSITMD.json
    rsitmd_json_file = '/data/clc/data/RSIMTD/dataset_RSITMD.json' 
    rsitmd_image_root = '/data/clc/data/RSIMTD/images'
    
   
    CKPT_PATH = config.get('checkpoint_path', None)
    if not CKPT_PATH:
        raise SystemExit("请通过命令行传入 checkpoint: checkpoint_path=/path/to/best.ckpt")
    # ===========================================

    if not os.path.exists(rsitmd_json_file):
        print(f"错误: 找不到 json 文件: {rsitmd_json_file}")
        # 尝试一些常见的默认路径作为 fallback
        possible_paths = [
            './data/RSITMD/dataset_RSITMD.json',
            '../data/RSITMD/dataset_RSITMD.json'
        ]
        for p in possible_paths:
            if os.path.exists(p):
                rsitmd_json_file = p
                print(f"找到替代路径: {rsitmd_json_file}")
                break

    # 加载模型
    print(f"加载模型: {CKPT_PATH}")
    from eval_utils import load_model_from_checkpoint
    model = load_model_from_checkpoint(CKPT_PATH, device)
    model.eval()
    _, preprocess = clip.load(config.model.original_clip_name, device=device)
    print("模型加载成功")
    
    # 加载数据 (只加载 Test 集)
    dataset = load_rsitmd_from_json(rsitmd_json_file, rsitmd_image_root, split='test')
    
    if len(dataset) == 0:
        print("错误: 数据集为空，请检查路径或 JSON split 字段")
        return

    # 提取特征
    print("开始提取文本特征...")
    text_features = get_text_features(model, dataset, device)
    
    print("开始提取图像特征...")
    image_features = get_image_features(model, preprocess, dataset, device)
    
    # 归一化
    image_features = F.normalize(image_features, p=2, dim=-1)
    text_features = F.normalize(text_features, p=2, dim=-1)
    
    # 计算相似度
    print(f"计算相似度矩阵: {image_features.shape} x {text_features.shape}.T")
    similarity = image_features @ text_features.T
    
    # 评估
    num_images = len(dataset)
    (i2t_r1, i2t_r5, i2t_r10), (t2i_r1, t2i_r5, t2i_r10) = evaluate_metrics(similarity, num_images)
    
    rsum = i2t_r1 + i2t_r5 + i2t_r10 + t2i_r1 + t2i_r5 + t2i_r10
    
    # 打印结果
    print("\n" + "="*50)
    print(f"RSITMD Test Set Evaluation (Images: {num_images})")
    print("="*50)
    print(f"Image-to-Text:")
    print(f"R@1:  {i2t_r1:.2f}")
    print(f"R@5:  {i2t_r5:.2f}")
    print(f"R@10: {i2t_r10:.2f}")
    print("-" * 20)
    print(f"Text-to-Image:")
    print(f"R@1:  {t2i_r1:.2f}")
    print(f"R@5:  {t2i_r5:.2f}")
    print(f"R@10: {t2i_r10:.2f}")
    print("-" * 20)
    print(f"RSUM: {rsum:.2f}")
    print("="*50)
    
    # 保存结果
    results = {
        'dataset': 'RSITMD',
        'split': 'test',
        'i2t': {'r1': i2t_r1, 'r5': i2t_r5, 'r10': i2t_r10},
        't2i': {'r1': t2i_r1, 'r5': t2i_r5, 'r10': t2i_r10},
        'rsum': rsum
    }
    
    output_file = f"eval_rsitmd_RSUM={rsum:.2f}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"结果已保存: {output_file}")

if __name__ == "__main__":
    evaluate_rsitmd()