"""
Shared evaluation utilities for GROUT-MAIL.
Consolidates model loading, feature extraction, and metric computation
that was duplicated across eval_coco_new.py, eval_RSICD.py, eval_RSITMD.py, eval_flickr2.py.
"""
import os
import sys
import torch
import torch.nn.functional as F
import clip
import numpy as np
from PIL import Image
from tqdm import tqdm
from omegaconf import OmegaConf

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def get_center_path(clip_name: str) -> str:
    """OmegaConf resolver: select prototype center path based on CLIP model name."""
    if "ViT-B" in clip_name:
        return "/data/clc/APSE-IPIK/NEW_APSEIPIK/offline/clustering_results/cluster_centers_160.npy"
    elif "ViT-L" in clip_name:
        return "/data/clc/APSE-IPIK/NEW_APSEIPIK/offline/clustering_results/cluster_centers_vitl14_160.npy"
    return "/data/clc/APSE-IPIK/NEW_APSEIPIK/offline/clustering_results/cluster_centers_160.npy"


def register_resolver():
    """Register the select_path OmegaConf resolver if not already registered."""
    if not OmegaConf.has_resolver("select_path"):
        OmegaConf.register_new_resolver("select_path", get_center_path)


def clean_state_dict(state_dict: dict) -> dict:
    """Remove model./module. prefix from state dict keys."""
    return {k.replace("model.", "").replace("module.", ""): v for k, v in state_dict.items()}


def load_model_from_checkpoint(checkpoint_path: str, device: str):
    """Load a model directly from a Lightning checkpoint, bypassing PL's
    load_from_checkpoint (which breaks when __init__ requires a complex config).

    Reads the saved hyper_parameters.config, reconstructs the correct model
    class, and loads the state dict manually.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # ── Reconstruct config from saved hyperparameters ──
    hparams = checkpoint.get('hyper_parameters', {})
    # Two possible formats depending on how save_hyperparameters was called:
    #   1. save_hyperparameters(config)        → hparams IS the config
    #   2. save_hyperparameters({'config': c})  → hparams['config'] is the config
    if isinstance(hparams, dict) and 'config' in hparams:
        config = hparams['config']
        if isinstance(config, dict):
            config = OmegaConf.create(config)
    elif hasattr(hparams, 'model') or (isinstance(hparams, dict) and 'model' in hparams):
        # hparams itself is the config (OmegaConf DictConfig or dict)
        config = OmegaConf.create(hparams) if isinstance(hparams, dict) else hparams
    else:
        raise ValueError(
            f"Cannot find config in checkpoint hyper_parameters. "
            f"hparams type: {type(hparams).__name__}, "
            f"keys: {list(hparams.keys()) if isinstance(hparams, dict) else 'N/A'}"
        )

    # ── Build the correct model class ──
    from src.models.mail_pure import PureMAILModel
    model = PureMAILModel(config)

    # ── Load weights ──
    state_dict = checkpoint.get('state_dict', checkpoint)
    state_dict = clean_state_dict(state_dict)
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    return model


def extract_image_features(model, preprocess, image_paths, device, batch_size=64):
    """Extract normalized image features for a list of image paths."""
    features = []
    with torch.no_grad():
        for i in tqdm(range(0, len(image_paths), batch_size), desc="Image features"):
            batch_paths = image_paths[i:i + batch_size]
            imgs = []
            for p in batch_paths:
                try:
                    imgs.append(preprocess(Image.open(p).convert('RGB')))
                except Exception:
                    imgs.append(torch.zeros(3, 224, 224))
            if imgs:
                img_tensor = torch.stack(imgs).to(device)
                features.append(model.encode_image(img_tensor).cpu())
    return F.normalize(torch.cat(features, dim=0), dim=-1)


def extract_text_features(model, captions, device, batch_size=64):
    """Extract normalized text features for a list of caption strings."""
    features = []
    with torch.no_grad():
        for i in tqdm(range(0, len(captions), batch_size), desc="Text features"):
            batch = captions[i:i + batch_size]
            tokens = clip.tokenize(batch, truncate=True).to(device)
            features.append(model.encode_text(tokens).cpu())
    return F.normalize(torch.cat(features, dim=0), dim=-1)


def evaluate_i2t(similarity, num_images, captions_per_image=5):
    """Image-to-Text recall metrics. Returns R@1, R@5, R@10 as percentages."""
    correct = {1: 0, 5: 0, 10: 0}
    for i in range(num_images):
        pred = similarity[i]
        for k in [1, 5, 10]:
            topk = set(pred.argsort(descending=True)[:k].tolist())
            gt = set(range(i * captions_per_image, (i + 1) * captions_per_image))
            if topk & gt:
                correct[k] += 1
    return (
        correct[1] * 100 / num_images,
        correct[5] * 100 / num_images,
        correct[10] * 100 / num_images,
    )


def evaluate_t2i(similarity, num_images, captions_per_image=5):
    """Text-to-Image recall metrics. Returns R@1, R@5, R@10 as percentages."""
    similarity_t = similarity.T
    num_texts = num_images * captions_per_image
    correct = {1: 0, 5: 0, 10: 0}
    for i in range(num_texts):
        pred = similarity_t[i]
        gt_idx = i // captions_per_image
        for k in [1, 5, 10]:
            topk = set(pred.argsort(descending=True)[:k].tolist())
            if gt_idx in topk:
                correct[k] += 1
    return (
        correct[1] * 100 / num_texts,
        correct[5] * 100 / num_texts,
        correct[10] * 100 / num_texts,
    )


def compute_rsum(i2t_scores, t2i_scores):
    """Compute RSUM = sum of all 6 recall metrics."""
    return sum(i2t_scores) + sum(t2i_scores)


def print_results(i2t_scores, t2i_scores, num_images, dataset_name=""):
    """Pretty-print evaluation results."""
    prefix = f" [{dataset_name}]" if dataset_name else ""
    print(f"\n{'='*50}")
    print(f"Evaluation Results{prefix} (Images: {num_images})")
    print(f"{'='*50}")
    print(f"Image-to-Text: R@1={i2t_scores[0]:.2f}  R@5={i2t_scores[1]:.2f}  R@10={i2t_scores[2]:.2f}")
    print(f"Text-to-Image: R@1={t2i_scores[0]:.2f}  R@5={t2i_scores[1]:.2f}  R@10={t2i_scores[2]:.2f}")
    rsum = compute_rsum(i2t_scores, t2i_scores)
    print(f"RSUM: {rsum:.2f}")
    print(f"{'='*50}")
    return rsum


def save_results(i2t_scores, t2i_scores, num_images, output_prefix="eval"):
    """Save evaluation results as JSON."""
    import json
    rsum = compute_rsum(i2t_scores, t2i_scores)
    results = {
        'i2t_r1': float(i2t_scores[0]), 'i2t_r5': float(i2t_scores[1]), 'i2t_r10': float(i2t_scores[2]),
        't2i_r1': float(t2i_scores[0]), 't2i_r5': float(t2i_scores[1]), 't2i_r10': float(t2i_scores[2]),
        'rsum': float(rsum), 'num_images': num_images,
    }
    output_file = f"{output_prefix}_RSUM={rsum:.2f}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {output_file}")
    return results
