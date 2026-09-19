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
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.systems.retrieval_system import DistillSystem

# ==========================================
# 核心修改 1: 加载数据时必须带上 imgid
# ==========================================
def load_rsicd_with_ids(json_file, image_root, max_images=1000):
    with open(json_file, 'r') as f:
        # 兼容不同 JSON 格式
        content = json.load(f)
        data = content.get("images", content)
    
    dataset = []
    count = 0
    
    # 强制使用 'val' 以对齐训练时的验证集
    target_split = 'test' 
    print(f"正在加载 {target_split} 集...")

    for idx, item in enumerate(data):
        if count >= max_images:
            break
            
        split = item.get('split', '')
        if split != target_split:
            continue

        filename = item.get('filename', '')
        # 【关键】获取 Image ID，如果没有就用 index 代替，但在 RSICD 中必须用 imgid
        img_id = item.get('imgid', idx)
        
        # 自动处理 .tif / .jpg 后缀问题
        base_path = os.path.join(image_root, filename)
        final_path = base_path
        if not os.path.exists(base_path):
            file_stem = os.path.splitext(filename)[0]
            for ext in ['.tif', '.jpg', '.png']:
                p = os.path.join(image_root, file_stem + ext)
                if os.path.exists(p):
                    final_path = p
                    break
        
        sentences = item.get('sentences', [])
        captions = [sentence['raw'] for sentence in sentences]
        
        if os.path.exists(final_path) and len(captions) >= 5:
            # 返回 (路径, 文本列表, 图片ID)
            dataset.append((final_path, captions[:5], img_id))
            count += 1
        else:
            # 只有在文件真的不存在时才跳过
            pass 
            
    print(f"从 {json_file} 加载了 {len(dataset)} 个样本 (Split: {target_split})")
    return dataset

# ==========================================
# 核心修改 2: 移植训练代码中的 ID 匹配逻辑
# ==========================================
def calculate_metrics_by_id(image_features, text_features, image_ids, device):
    """
    完全复刻训练代码的逻辑：
    1. 提取 Unique Images (去重 Gallery)
    2. 使用 Mask 矩阵计算 Recall (允许乱序，允许 ID 重复)
    """
    print("\n[开始计算指标 - 使用 ID 匹配模式]")
    
    # 1. 提取唯一图片 (Gallery Reduction)
    # 训练代码逻辑：unique_ids, unique_indices = np.unique(image_ids_np, return_index=True)
    image_ids_np = np.array(image_ids)
    unique_ids, unique_indices = np.unique(image_ids_np, return_index=True)
    
    # 按照 unique_indices 提取对应的图像特征
    unique_image_features = image_features[unique_indices].to(device)
    unique_ids_tensor = torch.tensor(unique_ids, device=device)
    
    # 文本特征不需要去重，保持原样
    all_text_features = text_features.to(device)
    
    # 原始所有图片的 ID (对应每一行文本所属的图片ID)
    # 注意：text_features 是 [N*5, Dim]，我们需要构造一个 [N*5] 的 ID 列表
    # 目前传入的 image_ids 是 [N] (对应 dataset 长度)
    # 所以我们需要把 image_ids 扩充 5 倍
    
    # 构造文本对应的真实 Image ID
    # 假设 dataset 里第 i 个元素的 id 是 X，那么 text_features 里第 5*i 到 5*i+4 的 ID 都是 X
    ids_list_expanded = []
    for img_id in image_ids:
        ids_list_expanded.extend([img_id] * 5)
    all_text_ids_tensor = torch.tensor(ids_list_expanded, device=device)
    
    print(f"Unique Images (Gallery): {len(unique_ids)}")
    print(f"Total Queries (Text): {len(all_text_ids_tensor)}")

    # 2. 计算相似度 [Text_N, Image_M_Unique]
    # 归一化 (防止外部忘记归一化)
    unique_image_features = F.normalize(unique_image_features, dim=-1)
    all_text_features = F.normalize(all_text_features, dim=-1)
    
    logits_per_text = torch.matmul(all_text_features, unique_image_features.t())
    logits_per_image = logits_per_text.t()

    # 3. 生成 Mask
    # ground_truth[i][j] = True 表示 Text[i] 的 ID 等于 UniqueImage[j] 的 ID
    # [Text_N, 1] == [1, Image_M]
    ground_truth_mask = (all_text_ids_tensor.unsqueeze(1) == unique_ids_tensor.unsqueeze(0))

    # 4. 计算 Recall 辅助函数
    def calc_r_k(logits, mask, prefix):
        # 取前 10
        _, top_indices = logits.topk(10, dim=1)
        # 根据 Mask 提取结果
        extracted_corrects = torch.gather(mask, 1, top_indices)
        
        res = {}
        for k in [1, 5, 10]:
            corrects_at_k = extracted_corrects[:, :k].any(dim=1)
            score = corrects_at_k.float().mean().item() * 100
            print(f"{prefix} R@{k}: {score:.2f}")
            res[f'{prefix}_r{k}'] = score
        return res
    print("--- Image to Text ---")
    i2t_res = calc_r_k(logits_per_image, ground_truth_mask.t(), "Image->Text")
    
    print("--- Text to Image ---")
    t2i_res = calc_r_k(logits_per_text, ground_truth_mask, "Text->Image")
    
    
    
    return {**t2i_res, **i2t_res}

# 辅助: 提取特征的 Loop (微调以收集 ID)
def extract_all_features(model, preprocess, dataset, device):
    img_feats = []
    txt_feats = []
    img_ids = []
    
    print("正在提取特征...")
    # Batch size
    bs = 64
    
    # 拆分数据
    paths = [x[0] for x in dataset]
    captions_group = [x[1] for x in dataset] # List of lists
    ids = [x[2] for x in dataset]
    
    # 1. 图像特征
    with torch.no_grad():
        for i in tqdm(range(0, len(paths), bs), desc="Images"):
            batch_paths = paths[i:i+bs]
            batch_imgs = []
            for p in batch_paths:
                try:
                    img = Image.open(p).convert("RGB")
                    batch_imgs.append(preprocess(img))
                except:
                    batch_imgs.append(torch.zeros(3, 224, 224))
            
            if batch_imgs:
                img_input = torch.stack(batch_imgs).to(device)
                feat = model.encode_image(img_input)
                img_feats.append(feat.cpu())

            if i == 0:
                feat_sum = feat[0].sum().item()
                feat_head = feat[0, :5].cpu().numpy().tolist()
                print(f"\n[指纹-离线态] Sum: {feat_sum:.6f} | Head: {feat_head}")
    
    # 2. 文本特征
    # 展平所有文本
    flat_captions = [c for group in captions_group for c in group]
    
    with torch.no_grad():
        for i in tqdm(range(0, len(flat_captions), bs), desc="Texts"):
            batch_txts = flat_captions[i:i+bs]
            tokens = clip.tokenize(batch_txts, truncate=True).to(device)
            feat = model.encode_text(tokens)
            txt_feats.append(feat.cpu())

    all_img_feats = torch.cat(img_feats, dim=0)
    all_txt_feats = torch.cat(txt_feats, dim=0)
    
    return all_img_feats, all_txt_feats, ids

# Hydra Resolver
def get_center_path(clip_name):
    # 这里请填入你确定的路径，或者保持原样
    USERSI = True
    if "ViT-B" in clip_name:
        if USERSI:
            return "/data/clc/APSE-IPIK/NEW_APSEIPIK/offline/RSI_clustering_results/cluster_centers_vitb32_RSI_32.npy"
    return "/data/clc/APSE-IPIK/NEW_APSEIPIK/offline/clustering_results/cluster_centers_160.npy"

OmegaConf.register_new_resolver("select_path", get_center_path)

@hydra.main(config_path="../configs", config_name="eval", version_base=None)
def main(config: DictConfig):
    print("=== RSICD 离线对齐测试 (ID Mode) ===")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. 加载模型
    checkpoint_path = config.get('checkpoint_path', None)
    if not checkpoint_path:
        raise SystemExit("请通过命令行传入 checkpoint: checkpoint_path=/path/to/best.ckpt")
    print(f"Checkpoint: {checkpoint_path}")

    from eval_utils import load_model_from_checkpoint
    model = load_model_from_checkpoint(checkpoint_path, device)
    model.eval()

    # CLIP Preprocess
    clipmodel, preprocess = clip.load(config.model.original_clip_name, device=device)
    
    # 2. 加载数据 (带 ID)
    json_file = '/data/clc/data/RSICD/annocations/dataset_rsicd.json'
    image_root = '/data/clc/data/RSICD/images'
    dataset = load_rsicd_with_ids(json_file, image_root, max_images=30000)
    
    # 3. 提取特征
    img_feats, txt_feats, img_ids = extract_all_features(model, preprocess, dataset, device)
    
    print(f"Img Feats: {img_feats.shape}")
    print(f"Txt Feats: {txt_feats.shape}")
    print(f"IDs count: {len(img_ids)}")
    
    # 4. 计算指标 (使用训练代码的逻辑)
    results = calculate_metrics_by_id(img_feats, txt_feats, img_ids, device)
    
    rsum = sum(results.values())
    print(f"\n[Final Result] RSUM: {rsum:.2f}")

if __name__ == "__main__":
    main()