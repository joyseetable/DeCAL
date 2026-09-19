import torch
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any
from omegaconf import DictConfig
import pytorch_lightning as pl


class RetrievalEvaluator:
    """
    Comprehensive evaluation module for retrieval tasks.
    Handles similarity computation, recall metrics, and logging.
    """
    
    def __init__(self, config: DictConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize metric storage
        self.metrics_history = {
            'i2t_r1': [], 'i2t_r5': [], 'i2t_r10': [],
            't2i_r1': [], 't2i_r5': [], 't2i_r10': [],
            'rsum': []
        }
    
    def compute_similarity_matrix(self, image_features: np.ndarray, text_features: np.ndarray) -> np.ndarray:
        """
        Compute similarity matrix between image and text embeddings.
        
        Args:
            image_features: Normalized image embeddings [n_images, embedding_dim]
            text_features: Normalized text embeddings [n_texts, embedding_dim]
            
        Returns:
            Similarity matrix [n_images, n_texts]
        """
        return image_features @ text_features.T
    
    def calculate_recalls(self, similarity_matrix: np.ndarray) -> Dict[str, float]:
        """
        Calculate retrieval recall metrics from similarity matrix.
        
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
            # Find best rank among the 5 captions for this image
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
            # Find the rank of the correct image
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
        
        recalls = {
            'i2t_r1': i2t_r1, 'i2t_r5': i2t_r5, 'i2t_r10': i2t_r10,
            't2i_r1': t2i_r1, 't2i_r5': t2i_r5, 't2i_r10': t2i_r10,
            'rsum': rsum
        }
        
        # Update metrics history
        for key, value in recalls.items():
            if key in self.metrics_history:
                self.metrics_history[key].append(value)
        
        return recalls
    
    def evaluate_embeddings(self, image_features: torch.Tensor, text_features: torch.Tensor) -> Dict[str, float]:
        """
        Evaluate retrieval performance from embeddings.
        
        Args:
            image_features: Image embeddings tensor
            text_features: Text embeddings tensor
            
        Returns:
            Dictionary of recall metrics
        """
        # Convert to numpy and ensure they're normalized
        img_feats_np = image_features.cpu().numpy()
        txt_feats_np = text_features.cpu().numpy()
        
        # Normalize features (in case they aren't already)
        img_feats_np = img_feats_np / np.linalg.norm(img_feats_np, axis=1, keepdims=True)
        txt_feats_np = txt_feats_np / np.linalg.norm(txt_feats_np, axis=1, keepdims=True)
        
        # Compute similarity matrix
        sim_matrix = self.compute_similarity_matrix(img_feats_np, txt_feats_np)
        
        # Calculate recalls
        return self.calculate_recalls(sim_matrix)
    
    def log_metrics(self, recalls: Dict[str, float], pl_module: pl.LightningModule, 
                   stage: str = "val", print_results: bool = True):
        """
        Log metrics to PyTorch Lightning and optionally print to console.
        
        Args:
            recalls: Dictionary of recall metrics
            pl_module: Lightning module for logging
            stage: Stage name (val, test, val_zeroshot)
            print_results: Whether to print results to console
        """
        if print_results:
            print(f"\n--- {stage.upper()} RECALLS ---")
            print(f"I2T: R@1 {recalls['i2t_r1']:.2f}, R@5 {recalls['i2t_r5']:.2f}, R@10 {recalls['i2t_r10']:.2f}")
            print(f"T2I: R@1 {recalls['t2i_r1']:.2f}, R@5 {recalls['t2i_r5']:.2f}, R@10 {recalls['t2i_r10']:.2f}")
        
        # Log individual metrics
        for metric_name, value in recalls.items():
            pl_module.log(f'{stage}/{metric_name}', value, 
                         prog_bar=(metric_name in ['i2t_r1', 't2i_r1', 'rsum']), 
                         sync_dist=True)
    
    def get_best_metrics(self) -> Dict[str, float]:
        """Get the best metrics from history based on RSUM"""
        if not self.metrics_history['rsum']:
            return {}
        
        best_idx = np.argmax(self.metrics_history['rsum'])
        best_metrics = {}
        
        for metric in self.metrics_history:
            if self.metrics_history[metric]:
                best_metrics[f'best_{metric}'] = self.metrics_history[metric][best_idx]
        
        return best_metrics
    
    def reset_metrics(self):
        """Reset metrics history"""
        for key in self.metrics_history:
            self.metrics_history[key] = []


class PromptAnalysisEvaluator:
    """
    Specialized evaluator for analyzing prompt selection behavior in VPT models.
    """
    
    def __init__(self, config: DictConfig):
        self.config = config
        self.prompt_selection_history = []
    
    def analyze_prompt_selection(self, prompt_outputs: Dict[str, Any], batch_idx: int) -> Dict[str, Any]:
        """
        Analyze prompt selection patterns from the prompt module outputs.
        
        Args:
            prompt_outputs: Output from the Prompt module forward pass
            batch_idx: Current batch index
            
        Returns:
            Dictionary with prompt selection analysis
        """
        if 'prompt_idx' not in prompt_outputs:
            return {}
        
        prompt_indices = prompt_outputs['prompt_idx'].cpu().numpy()
        similarities = prompt_outputs['similarity'].cpu().numpy()
        
        analysis = {
            'batch_idx': batch_idx,
            'prompt_indices': prompt_indices,
            'similarities': similarities,
            'top_prompts_frequency': np.bincount(prompt_indices.flatten(), minlength=self.config.model.pool_size),
            'avg_similarity': np.mean(similarities),
            'prompt_diversity': len(np.unique(prompt_indices)) / self.config.model.top_k
        }
        
        self.prompt_selection_history.append(analysis)
        return analysis
    
    def get_prompt_usage_summary(self) -> Dict[str, Any]:
        """Get summary statistics of prompt usage"""
        if not self.prompt_selection_history:
            return {}
        
        all_indices = np.concatenate([item['prompt_indices'].flatten() for item in self.prompt_selection_history])
        
        return {
            'total_selections': len(all_indices),
            'unique_prompts_used': len(np.unique(all_indices)),
            'most_frequent_prompt': np.argmax(np.bincount(all_indices)),
            'prompt_usage_distribution': np.bincount(all_indices, minlength=self.config.model.pool_size),
            'avg_prompt_diversity': np.mean([item['prompt_diversity'] for item in self.prompt_selection_history])
        }