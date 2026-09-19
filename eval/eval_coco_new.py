import sys
import os
import time
import torch
import torch.nn.functional as F
from torchvision.datasets import CocoCaptions
import clip
import hydra
from omegaconf import DictConfig, OmegaConf
import json
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.systems.retrieval_system import DistillSystem

# Use shared eval utilities
from eval_utils import register_resolver, load_model_from_checkpoint, clean_state_dict
register_resolver()
@hydra.main(config_path="../configs", config_name="eval", version_base=None)
def evaluate_final_model(config: DictConfig):
    """
    最终模型评估 - 直接使用训练好的模型进行特征提取和检索评估
    """
    print("=== 最终模型评估 ===")
    print(OmegaConf.to_yaml(config))
   
    # 设备配置
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")
    
    # Load model using shared utility
    checkpoint_path = config.get('checkpoint_path', None)
    if not checkpoint_path:
        raise SystemExit("请通过命令行传入 checkpoint: checkpoint_path=/path/to/best.ckpt")
    model = load_model_from_checkpoint(checkpoint_path, device)

    # 使用CLIP的预处理
    clip_model, preprocess = clip.load(config.model.original_clip_name, device=device)
    
    # 加载COCO数据集
    coco_root = config.dataset.get('root', '/data/clc/Long-CLIP/data/coco/val2017')
    coco_annfile = config.dataset.get('annFile', '/data/clc/Long-CLIP/data/coco/annotations/captions_val2017.json')
    
    coco = CocoCaptions(root=coco_root, annFile=coco_annfile)
    
    num_samples = 5000  # 评估的样本数量
    
    print(f"\n=== 开始评估，使用 {num_samples} 个样本 ===")
    
    image_features = []
    text_features = []
    # === [新增] 初始化统计变量 ===
    total_infer_time = 0.0  # 总推理时间
    total_infer_count = 0   # 推理次数 (1次encode算1次)
    torch.cuda.reset_peak_memory_stats() # 重置显存峰值记录
    torch.cuda.empty_cache()
    # ==========================
    with torch.no_grad():
        
        for i, (image, captions) in enumerate(tqdm(coco, desc="提取特征")):
            if i >= num_samples:
                break
            
            # 图像特征提取 - 使用模型的encode_image方法
            image_input = preprocess(image).unsqueeze(0).to(device)
            # [新增] 图像推理计时
            if i > 10: torch.cuda.synchronize() # 跳过Warmup
            start_t = time.time()
            
            image_feature = model.encode_image(image_input)

            if i > 10:
                torch.cuda.synchronize()
                total_infer_time += time.time() - start_t
                total_infer_count += 1

            image_features.append(image_feature.float().cpu())
            
            # 文本特征提取 - 使用enhance_text方法获取增强文本特征
            captions = captions[:5]  # 每个图像取前5个描述
            for caption in captions:
                text_input = clip.tokenize([caption]).to(device)
                # [新增] 文本推理计时
                if i > 10:
                    torch.cuda.synchronize()
                start_t = time.time()
                text_feature = model.encode_text(text_input)
                if i > 10:
                    torch.cuda.synchronize()
                    total_infer_time += time.time() - start_t
                    total_infer_count += 1
                text_features.append(text_feature.float().cpu())
    # === [新增] 计算并打印性能指标 ===
    peak_memory = torch.cuda.max_memory_allocated() / 1024 / 1024 # MB
    avg_latency = (total_infer_time / total_infer_count) * 1000 if total_infer_count > 0 else 0 # ms
    
    print("\n" + "="*40)
    print("🚀 性能分析 (Complexity Analysis)")
    print("="*40)
    print(f"显存占用 (Peak Memory): {peak_memory:.2f} MB")
    print(f"平均延迟 (Avg Latency): {avg_latency:.2f} ms / sample")
    print("="*40 + "\n")
    # ================================
    # 合并特征
    image_features = torch.cat(image_features, dim=0)
    text_features = torch.cat(text_features, dim=0)
    image_features = image_features.float()
    text_features = text_features.float()
    
    print(f"特征维度: 图像 {image_features.shape}, 文本 {text_features.shape}")
    
    # 归一化
    image_features = F.normalize(image_features, p=2, dim=-1)
    text_features = F.normalize(text_features, p=2, dim=-1)
    
    # 计算相似度矩阵
    similarity = image_features @ text_features.T
    print(f"相似度矩阵维度: {similarity.shape}")
    
    # 评估指标
    print("计算评估指标...")
    
    # Image-to-Text 检索
    i2t_r1, i2t_r5, i2t_r10 = evaluate_i2t(similarity, num_samples) 
    
    # Text-to-Image 检索  
    t2i_r1, t2i_r5, t2i_r10 = evaluate_t2i(similarity, num_samples)
    
    # 计算RSUM
    rsum = i2t_r1 + i2t_r5 + i2t_r10 + t2i_r1 + t2i_r5 + t2i_r10
    
    # 存储结果
    results = {
        'i2t_r1': i2t_r1,
        'i2t_r5': i2t_r5, 
        'i2t_r10': i2t_r10,
        't2i_r1': t2i_r1,
        't2i_r5': t2i_r5,
        't2i_r10': t2i_r10,
        'rsum': rsum
    }
    
    # 打印结果
    print("\n" + "="*70)
    print("最终模型评估结果")
    print("="*70)
    
    print(f"图像到文本检索:")
    print(f"  R@1:  {i2t_r1:.4f}")
    print(f"  R@5:  {i2t_r5:.4f}") 
    print(f"  R@10: {i2t_r10:.4f}")
    print(f"文本到图像检索:")
    print(f"  R@1:  {t2i_r1:.4f}")
    print(f"  R@5:  {t2i_r5:.4f}")
    print(f"  R@10: {t2i_r10:.4f}")
    print(f"RSUM:   {rsum:.4f}")
    
    # 保存结果
    output_file = f"val_at_mscoco_RSUM={rsum:.4f}.json"
    with open(output_file, 'w') as f:
        json.dump({k: float(v) for k, v in results.items()}, f, indent=2)
    
    print(f"\n结果已保存到: {output_file}")
    return results

def evaluate_i2t(similarity, num_samples):
    """评估Image-to-Text检索"""
    pred_true_r1 = 0
    pred_true_r5 = 0 
    pred_true_r10 = 0
    
    for i in range(num_samples):
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
    
    return pred_true_r1*100/num_samples, pred_true_r5*100/num_samples, pred_true_r10*100/num_samples

def evaluate_t2i(similarity, num_samples):
    """评估Text-to-Image检索"""
    similarity_t2i = similarity.T
    
    pred_true_r1 = 0
    pred_true_r5 = 0
    pred_true_r10 = 0
    
    for i in range(num_samples * 5):
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
    
    return pred_true_r1*100/(num_samples*5), pred_true_r5*100/(num_samples*5), pred_true_r10*100/(num_samples*5)

if __name__ == "__main__":
    evaluate_final_model()