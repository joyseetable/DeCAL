import numpy as np
import torch

def compute_recalls_from_similarity(similarity_matrix: np.ndarray, captions_per_image: int = 5):
    """
    输入:
      similarity_matrix: numpy array, shape (N_images, N_texts)  (N_texts == N_images * captions_per_image)
      captions_per_image: 每张图片对应的文本数量（Flickr30k 为 5）

    输出:
      recalls: dict 包含 i2t_r1/i2t_r5/i2t_r10/t2i_r1/t2i_r5/t2i_r10（百分比，0-100）
    """
    assert isinstance(similarity_matrix, np.ndarray), "similarity_matrix must be numpy.ndarray"
    n_images, n_texts = similarity_matrix.shape
    assert n_texts == n_images * captions_per_image, (
        f"文本数量不匹配: n_texts={n_texts}, expected {n_images}*{captions_per_image}={n_images*captions_per_image}"
    )

    # ---------- Image -> Text (I2T) ----------
    # 对每张图像，得到对所有文本的排序（从大到小）
    # 为效率：用 argpartition 获得 top-k，再精确排序
    def i2t_recalls(sim, ks=(1,5,10)):
        i2t = {}
        # 获取每一行的索引，按相似度从大到小
        ranks = np.argsort(-sim, axis=1)  # shape (n_images, n_texts)
        # 对每张图像 i，正确的文本索引范围是 [i*cpi, (i+1)*cpi)
        for k in ks:
            hit = 0
            for i in range(n_images):
                correct_start = i * captions_per_image
                correct_end = (i+1) * captions_per_image
                topk = ranks[i, :k]
                # 如果 top-k 中任意一个是正确文本则算命中
                if np.any((topk >= correct_start) & (topk < correct_end)):
                    hit += 1
            i2t[f"i2t_r{k}"] = 100.0 * hit / n_images
        return i2t

    # ---------- Text -> Image (T2I) ----------
    # 对每个文本（总共 n_texts），判断它对应的 image 在所有图像中的排名
    def t2i_recalls(sim, ks=(1,5,10)):
        t2i = {}
        # 为每个文本 j，我们要知道它对应的是哪张图像：
        # image_idx = j // captions_per_image
        # 先计算文本到图像的相似度矩阵 = sim.T (shape n_texts x n_images)
        sim_t = sim.T  # (n_texts, n_images)
        ranks_t = np.argsort(-sim_t, axis=1)  # 对每个文本，图像按相似度降序排列
        # 对每个 k 统计命中率
        for k in ks:
            hit = 0
            for j in range(n_texts):
                image_idx = j // captions_per_image
                topk = ranks_t[j, :k]
                if image_idx in topk:
                    hit += 1
            t2i[f"t2i_r{k}"] = 100.0 * hit / n_texts
        return t2i

    ks = (1,5,10)
    i2t = i2t_recalls(similarity_matrix, ks)
    t2i = t2i_recalls(similarity_matrix, ks)

    recalls = {}
    recalls.update(i2t)
    recalls.update(t2i)
    return recalls


# ----------------- 集成到 Lightning 的 on_validation_epoch_end 的示例 -----------------
def on_validation_epoch_end_full_eval(self, captions_per_image=5):
    """
    替换/调用于 OriginalPaperLitSystem.on_validation_epoch_end
    要求: self.validation_step_outputs 是一个 list，每个元素:
        {"image_features": tensor(B, D), "text_features": tensor(B*cpi, D)}
    且 dataloader 保证 shuffle=False, 每张图的 5 条 caption 在全局文本序列中按图像顺序排列。
    """
    if not hasattr(self, "validation_step_outputs") or len(self.validation_step_outputs) == 0:
        print("No validation outputs collected.")
        return {}

    # 拼接所有 batch（on_validation_epoch_end 前已经把每 batch 的 features .cpu() 存好了）
    all_image_feats = torch.cat([x['image_features'] for x in self.validation_step_outputs], dim=0)
    all_text_feats  = torch.cat([x['text_features'] for x in self.validation_step_outputs], dim=0)

    # debug prints (shape)
    print(f"[DEBUG] all_image_feats.shape = {all_image_feats.shape}")
    print(f"[DEBUG] all_text_feats.shape  = {all_text_feats.shape}")

    # 断言数量关系
    n_images = all_image_feats.shape[0]
    n_texts  = all_text_feats.shape[0]
    assert n_texts == n_images * captions_per_image, (
        f"文本/图像数量不匹配: n_texts={n_texts}, n_images={n_images}, expected n_texts == n_images*{captions_per_image}"
    )

    # 归一化（以防某些路径没做）
    all_image_feats = torch.nn.functional.normalize(all_image_feats, dim=-1)
    all_text_feats  = torch.nn.functional.normalize(all_text_feats, dim=-1)

    # 计算相似度（在 CPU 上 numpy 计算）
    # 如果数据量较大也可用 GPU: similarity = all_image_feats.to(device) @ all_text_feats.to(device).T
    sim = (all_image_feats @ all_text_feats.T).cpu().numpy()  # shape (n_images, n_texts)

    # sanity-check: sim 值应在 [-1, 1]（因为是余弦相似度）
    print(f"[DEBUG] sim.min={sim.min():.6f}, sim.max={sim.max():.6f}, mean={sim.mean():.6f}")

    # 计算 recalls
    recalls = compute_recalls_from_similarity(sim, captions_per_image)

    # 打印并 log
    stage_name = "val"
    print(f"\n--- {stage_name.upper()} RECALLS ---")
    print(f"I2T: R@1 {recalls['i2t_r1']:.2f}, R@5 {recalls['i2t_r5']:.2f}, R@10 {recalls['i2t_r10']:.2f}")
    print(f"T2I: R@1 {recalls['t2i_r1']:.2f}, R@5 {recalls['t2i_r5']:.2f}, R@10 {recalls['t2i_r10']:.2f}")

    # Lightning logging if available
    if hasattr(self, "log"):
        for k, v in recalls.items():
            self.log(f"{stage_name}/recall/{k}", v, prog_bar=(k in ["i2t_r1", "t2i_r1"]), sync_dist=True)
        self.log(f"{stage_name}/RSUM", sum(recalls.values()), prog_bar=True, sync_dist=True)

    # 清空缓存的 outputs（以免下一 epoch 重复）
    self.validation_step_outputs.clear()
    return recalls
