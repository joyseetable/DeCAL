import torch
import torch.nn.functional as F
from typing import Dict
import numpy as np


def info_nce_loss(image_features: torch.Tensor, text_features: torch.Tensor, logit_scale: torch.Tensor) -> torch.Tensor:
    """
    Compute InfoNCE/CLIP contrastive loss.
    
    Args:
        image_features: Normalized image embeddings
        text_features: Normalized text embeddings  
        logit_scale: Learnable temperature parameter
        
    Returns:
        Contrastive loss value
    """
    # Cosine similarity matrix
    logits_per_image = logit_scale * image_features @ text_features.t()
    logits_per_text = logits_per_image.t()
    
    # Symmetric cross entropy loss
    labels = torch.arange(len(logits_per_image), device=image_features.device)
    loss_i = F.cross_entropy(logits_per_image, labels)
    loss_t = F.cross_entropy(logits_per_text, labels)
    
    return (loss_i + loss_t) / 2


def triplet_loss(anchor: torch.Tensor, positive: torch.Tensor, margin: float) -> torch.Tensor:
    """
    Compute triplet loss for retrieval.
    
    Args:
        anchor: Anchor embeddings (image features)
        positive: Positive embeddings (matching text features)
        margin: Margin for triplet loss
        
    Returns:
        Triplet loss value
    """
    batch_size = anchor.size(0)
    
    # Compute similarity matrices
    sim_ap = torch.matmul(anchor, positive.t())  # anchor-positive similarities
    mask = torch.eye(batch_size, device=anchor.device, dtype=torch.bool)
    negative_similarities = sim_ap.masked_fill(mask, -1e9)
    hardest_negative, _ = torch.max(negative_similarities, dim=1)
    # For each anchor, hardest negative is the most similar non-matching positive
    # sim_an = sim_ap.clone()
    # # Mask out positives (diagonal)
    # sim_an.fill_diagonal_(-1e9)
    
    # # Hardest negative for each anchor
    # hardest_negative, _ = torch.max(sim_an, dim=1)
    positive_similarities = torch.diag(sim_ap)
    # Positive similarities (diagonal)
    # positive_similarities = torch.diag(sim_ap)
    
    # Triplet loss
    loss = F.relu(margin + hardest_negative - positive_similarities)
    return loss.mean()


def calculate_recalls(similarity_matrix: np.ndarray) -> Dict[str, float]:
    """
    Calculate retrieval recall metrics from similarity matrix.
    This function is kept for backward compatibility.
    
    Args:
        similarity_matrix: Precomputed similarity matrix
        
    Returns:
        Dictionary containing recall metrics
    """
    n_images = similarity_matrix.shape[0]
    n_captions = similarity_matrix.shape[1]
    
    # Image-to-Text retrieval
    i2t_ranks = np.zeros(n_images)
    for i in range(n_images):
        inds = np.argsort(similarity_matrix[i])[::-1]
        rank = n_captions
        for j in range(5):  # Assuming 5 captions per image
            caption_idx = i * 5 + j
            if caption_idx < n_captions:
                tmp = np.where(inds == caption_idx)[0][0]
                if tmp < rank:
                    rank = tmp
        i2t_ranks[i] = rank
    
    # Text-to-Image retrieval
    t2i_ranks = np.zeros(n_captions)
    for i in range(n_captions):
        inds = np.argsort(similarity_matrix[:, i])[::-1]
        image_idx = i // 5  # Each image has 5 captions
        rank = np.where(inds == image_idx)[0][0]
        t2i_ranks[i] = rank
    
    # Calculate recall metrics
    i2t_r1 = 100.0 * np.sum(i2t_ranks < 1) / len(i2t_ranks)
    i2t_r5 = 100.0 * np.sum(i2t_ranks < 5) / len(i2t_ranks)
    i2t_r10 = 100.0 * np.sum(i2t_ranks < 10) / len(i2t_ranks)
    
    t2i_r1 = 100.0 * np.sum(t2i_ranks < 1) / len(t2i_ranks)
    t2i_r5 = 100.0 * np.sum(t2i_ranks < 5) / len(t2i_ranks)
    t2i_r10 = 100.0 * np.sum(t2i_ranks < 10) / len(t2i_ranks)
    
    rsum = i2t_r1 + i2t_r5 + i2t_r10 + t2i_r1 + t2i_r5 + t2i_r10
    
    return {
        'i2t_r1': i2t_r1, 'i2t_r5': i2t_r5, 'i2t_r10': i2t_r10,
        't2i_r1': t2i_r1, 't2i_r5': t2i_r5, 't2i_r10': t2i_r10,
        'rsum': rsum
    }


def prompt_consistency_loss(prompt_similarities: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    """
    Compute consistency loss for prompt selection to encourage diverse prompt usage.
    
    Args:
        prompt_similarities: Similarity matrix between queries and prompts
        temperature: Temperature for softmax
        
    Returns:
        Consistency loss value
    """
    batch_size, pool_size = prompt_similarities.shape
    
    # Apply softmax to get probability distribution
    probs = F.softmax(prompt_similarities / temperature, dim=1)
    
    # Encourage diverse prompt selection by maximizing entropy
    entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
    
    # We want to maximize entropy (encourage diverse selection), so we minimize negative entropy
    return -torch.mean(entropy)



