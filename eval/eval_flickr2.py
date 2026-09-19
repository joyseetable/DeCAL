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
from src.systems.retrieval_system import DistillSystem
from eval_utils import register_resolver, load_model_from_checkpoint
register_resolver()

def load_flickr30k_from_json(json_file, image_root, max_images=1000):
    """
    从JSON文件加载Flickr30K验证集，限制最大图片数量
    """
    with open(json_file, 'r') as f:
        data = json.load(f)["images"]
    
    dataset = []
    count = 0
    
    for item in data:
        if count >= max_images:
            break
            
        # 根据您提供的JSON结构
        filename = item.get('filename', '')
        image_path = os.path.join(image_root, filename)
        sentences = item.get('sentences', [])
        
        # 提取所有描述的原始文本
        captions = [sentence['raw'] for sentence in sentences]
        
        if os.path.exists(image_path) and len(captions) >= 5:
            dataset.append((image_path, captions[:5]))  # 确保有5个描述
            count += 1
        else:
            print(f"跳过图像 {filename}: 文件不存在或描述不足5个")
    
    print(f"从 {json_file} 加载了 {len(dataset)} 个样本 (限制: {max_images})")
    return dataset

def get_text_features(model, dataset, device, batch_size=64):
    """获取文本特征 - 使用增强文本编码器"""
    feature_list = []
    
    # 收集所有文本
    all_captions = []
    for img_path, captions in dataset:
        all_captions.extend(captions[:5])  # 每个图像取5个描述
    
    print(f"总文本数量: {len(all_captions)} (应该是 {len(dataset) * 5})")
    
    # 分批处理避免OOM
    with torch.no_grad():
        for i in range(0, len(all_captions), batch_size):
            end_idx = min(i + batch_size, len(all_captions))
            text_batch = all_captions[i:end_idx]
            
            # 使用CLIP的tokenizer
            text_tokens = clip.tokenize(text_batch, truncate=True).to(device)
            
            # 使用增强文本编码器
            text_features = model.encode_text(text_tokens)
            
            
            # 确保数据类型一致并移到CPU
            text_features = text_features.float().cpu()
            feature_list.append(text_features)
            
            if (i // batch_size) % 10 == 0:
                print(f"已处理 {i}/{len(all_captions)} 个文本")
    
    text_features = torch.cat(feature_list, dim=0)
    return text_features

def get_image_features(model, preprocess, dataset, device, max_images=1000):
    """获取图像特征 - 使用模型的图像编码器，限制图片数量"""
    img_feature_list = []
    
    # 确保不超过最大图片数量
    dataset = dataset[:max_images]
    
    with torch.no_grad():
        for i, (img_path, captions) in enumerate(tqdm(dataset, desc="提取图像特征")):
            try:
                image = Image.open(img_path).convert('RGB')
                image_input = preprocess(image).unsqueeze(0).to(device)
                
                # 使用模型的encode_image方法
                img_feature = model.encode_image(image_input)
                
                
                # 确保数据类型一致并移到CPU
                img_feature = img_feature.float().cpu()
                img_feature_list.append(img_feature)
                
                # 清理GPU内存
                if (i + 1) % 50 == 0:
                    torch.cuda.empty_cache()
                    
            except Exception as e:
                print(f"处理图像 {img_path} 时出错: {e}")
                continue

    img_features = torch.cat(img_feature_list, dim=0)
    return img_features

def evaluate_i2t(similarity, num_images):
    """评估Image-to-Text检索"""
    pred_true_r1 = 0
    pred_true_r5 = 0 
    pred_true_r10 = 0
    
    for i in range(num_images):
        pred = similarity[i]
        
        # R@1
        top1_indices = pred.argsort(descending=True)[:1]
        for j in range(5):
            true_index = 5 * i + j
            if true_index in top1_indices:
                pred_true_r1 += 1
                break
        
        # R@5
        top5_indices = pred.argsort(descending=True)[:5]
        for j in range(5):
            true_index = 5 * i + j
            if true_index in top5_indices:
                pred_true_r5 += 1
                break
        
        # R@10
        top10_indices = pred.argsort(descending=True)[:10]
        for j in range(5):
            true_index = 5 * i + j
            if true_index in top10_indices:
                pred_true_r10 += 1
                break
    
    i2t_r1 = pred_true_r1 / num_images
    i2t_r5 = pred_true_r5 / num_images
    i2t_r10 = pred_true_r10 / num_images
    
    print(f"Image→Text R@1: {i2t_r1:.4f}")
    print(f"Image→Text R@5: {i2t_r5:.4f}")
    print(f"Image→Text R@10: {i2t_r10:.4f}")
    
    return i2t_r1*100, i2t_r5*100, i2t_r10*100

def evaluate_t2i(similarity, num_images):
    """评估Text-to-Image检索"""
    similarity_t2i = similarity.T
    num_texts = num_images * 5
    
    pred_true_r1 = 0
    pred_true_r5 = 0
    pred_true_r10 = 0
    
    for i in range(num_texts):
        pred = similarity_t2i[i]
        true_index = i // 5  # 对应的图像索引
        
        # R@1
        top1_indices = pred.argsort(descending=True)[:1]
        if true_index in top1_indices:
            pred_true_r1 += 1
        
        # R@5  
        top5_indices = pred.argsort(descending=True)[:5]
        if true_index in top5_indices:
            pred_true_r5 += 1
        
        # R@10
        top10_indices = pred.argsort(descending=True)[:10]
        if true_index in top10_indices:
            pred_true_r10 += 1
    
    t2i_r1 = pred_true_r1 / num_texts
    t2i_r5 = pred_true_r5 / num_texts
    t2i_r10 = pred_true_r10 / num_texts
    
    print(f"Text→Image R@1: {t2i_r1:.4f}")
    print(f"Text→Image R@5: {t2i_r5:.4f}")
    print(f"Text→Image R@10: {t2i_r10:.4f}")
    
    return t2i_r1*100, t2i_r5*100, t2i_r10*100


@hydra.main(config_path="../configs", config_name="eval", version_base=None)
def evaluate_flickr30k_final(config: DictConfig):
    """
    使用官方分割的Flickr30K最终评估 - 限制1000张图片
    """
    print("=== Flickr30K 官方验证集评估 (限制1000张图片) ===")
    print(OmegaConf.to_yaml(config))
    
    # 设备配置
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")
    
    # 最大图片数量
    MAX_IMAGES = 1000
    
    # Load model using shared utility
    checkpoint_path = config.get('checkpoint_path', None)
    if not checkpoint_path:
        raise SystemExit("请通过命令行传入 checkpoint: checkpoint_path=/path/to/best.ckpt")
    model = load_model_from_checkpoint(checkpoint_path, device)

    # 使用CLIP的预处理
    _, preprocess = clip.load(config.model.original_clip_name, device=device)

    # 加载Flickr30K数据集
    json_file = config.dataset.get('json_file', '/data/clc/data/flickr30k/annotations/test.json')
    image_root = config.dataset.get('image_root', '/data/clc/data/flickr30k/images')
    
    # 如果路径不存在，尝试自动查找
    if not os.path.exists(json_file):
        possible_json_paths = [
            './data/flickr30k/val.json',
            '/data/flickr30k/val.json',
            '/data/clc/Long-CLIP/data/flickr30k/val.json'
        ]
        for path in possible_json_paths:
            if os.path.exists(path):
                json_file = path
                break
    
    if not os.path.exists(image_root):
        possible_image_paths = [
            './data/flickr30k-images',
            '/data/flickr30k-images',
            '/data/clc/Long-CLIP/data/flickr30k-images'
        ]
        for path in possible_image_paths:
            if os.path.exists(path):
                image_root = path
                break
    
    print(f"JSON文件: {json_file}")
    print(f"图像路径: {image_root}")
    
    # 加载数据集 - 限制1000张图片
    dataset = load_flickr30k_from_json(json_file, image_root, max_images=MAX_IMAGES)
    
    if len(dataset) == 0:
        print("错误: 没有加载到任何数据!")
        return
    
    # 确保数据集不超过1000张
    if len(dataset) > MAX_IMAGES:
        dataset = dataset[:MAX_IMAGES]
        print(f"警告: 数据集超过{MAX_IMAGES}张，已截断为前{MAX_IMAGES}张")
    
    print(f"最终使用的图片数量: {len(dataset)}")
    
    # 提取特征
    print("开始提取文本特征...")
    text_features = get_text_features(model, dataset, device)

    print("开始提取图像特征...")
    image_features = get_image_features(model, preprocess, dataset, device, max_images=MAX_IMAGES)
    
    print(f"特征维度: 文本 {text_features.shape}, 图像 {image_features.shape}")
    print(f"特征数据类型: 文本 {text_features.dtype}, 图像 {image_features.dtype}")
    
    # 验证特征数量是否正确
    expected_text_features = len(dataset) * 5
    if text_features.shape[0] != expected_text_features:
        print(f"警告: 文本特征数量不匹配! 期望: {expected_text_features}, 实际: {text_features.shape[0]}")
    
    if image_features.shape[0] != len(dataset):
        print(f"警告: 图像特征数量不匹配! 期望: {len(dataset)}, 实际: {image_features.shape[0]}")
    
    # 确保数据类型一致
    image_features = image_features.float()
    text_features = text_features.float()
    
    # 归一化
    image_features = F.normalize(image_features, p=2, dim=-1)
    text_features = F.normalize(text_features, p=2, dim=-1)
    
    # 计算相似度矩阵
    similarity = image_features @ text_features.T
    print(f"相似度矩阵维度: {similarity.shape}")
    
    # 评估指标
    print("\n" + "="*50)
    print("评估结果")
    print("="*50)
    
    num_images = len(dataset)
    
    # Image→Text 检索
    i2t_r1, i2t_r5, i2t_r10 = evaluate_i2t(similarity, num_images) 
    
    # Text→Image 检索
    t2i_r1, t2i_r5, t2i_r10 = evaluate_t2i(similarity, num_images)
    
    # 计算RSUM
    rsum = i2t_r1 + i2t_r5 + i2t_r10 + t2i_r1 + t2i_r5 + t2i_r10
    
    # 存储结果
    results = {
        'i2t_r1': float(i2t_r1),
        'i2t_r5': float(i2t_r5), 
        'i2t_r10': float(i2t_r10),
        't2i_r1': float(t2i_r1),
        't2i_r5': float(t2i_r5),
        't2i_r10': float(t2i_r10),
        'rsum': float(rsum),
        'num_images': num_images,
        'num_texts': num_images * 5
    }
    
    print(f"\nRSUM: {rsum:.4f}")
    print(f"使用的图片数量: {num_images}")
    print(f"使用的文本数量: {num_images * 5}")
    
    # 保存结果
    output_file = f"val_at_flickr30k_RSUM={rsum:.4f}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n结果已保存到: {output_file}")
    
    return results

if __name__ == "__main__":
    evaluate_flickr30k_final()