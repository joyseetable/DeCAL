import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from omegaconf import DictConfig
import clip
import logging
import os
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np

class DistillSystem(pl.LightningModule):
    def __init__(self, config: DictConfig):
        super().__init__()
        self.save_hyperparameters(config)
        self.config = config

        from src.models.mail_pure import PureMAILModel
        self.model = PureMAILModel(config)

        self._setup_human_readable_logging()

        self.helper_models = {}
        if not config.get("test_only", False):
            zeroshot_model, _ = clip.load(config.model.original_clip_name, device="cpu", jit=False)
            self.helper_models['zeroshot'] = zeroshot_model.eval().float()

        self.validation_outputs = []
        self.is_sanity_check = True
        self.baseline_validated = False
        self._print_trainable_parameters()
    def _print_trainable_parameters(self):
        if self.global_rank != 0:
            return

        total_params = 0
        trainable_params = 0
        for name, param in self.model.named_parameters():
            total_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()

        print(f"\n{'='*40}")
        print(f"Trainable Parameters: {trainable_params:,} / {total_params:,} ({trainable_params/total_params:.2%})")
        print(f"{'='*40}\n")
    def _setup_human_readable_logging(self):
        
        self.logger_human = logging.getLogger("human_readable")
        self.logger_human.setLevel(logging.INFO)
        
        # 避免重复添加处理器
        if self.logger_human.handlers:
            return

        # 1. 基础格式器
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        node_rank = int(os.environ.get("NODE_RANK", "0"))
        
        if local_rank == 0 and node_rank == 0:
            # 只有主进程才有资格生成时间戳和创建文件
            log_dir = self.config.get("log_dir", "./logs")
            os.makedirs(log_dir, exist_ok=True)
            
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = os.path.join(log_dir, f"training_{timestamp}.log")
            
            # 文件处理器
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.INFO)
            file_handler.setFormatter(formatter)
            self.logger_human.addHandler(file_handler)
            
            # 控制台处理器 (主进程负责打印到屏幕)
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            console_handler.setFormatter(formatter)
            self.logger_human.addHandler(console_handler)
            
            self.logger_human.info(f"🚀 Training started at {timestamp}")
            self.logger_human.info(f"📂 Log file strictly pinned to Rank 0: {log_file}")
        else:
            # 其他 7 个从属进程：配置一个 NullHandler 防止报错，但不输出任何东西
            self.logger_human.addHandler(logging.NullHandler())
            # 强制屏蔽从属进程的日志级别
            self.logger_human.setLevel(logging.CRITICAL)
    def setup(self, stage: str):
        if 'zeroshot' in self.helper_models:
            self.helper_models['zeroshot'] = self.helper_models['zeroshot'].to(self.device)

    def forward(self, batch):
        return self.model(batch)

    def on_train_epoch_start(self):
        if self.global_rank == 0:
            self.logger_human.info(f"Starting training Epoch {self.current_epoch}")

    def on_train_batch_start(self, batch, batch_idx):
        if self.global_rank != 0:
            return
        if batch_idx == 0:
            gpu_memory = torch.cuda.memory_allocated() / 1024**3
            self.logger_human.info(f"GPU memory: {gpu_memory:.2f} GB")
            current_lr = self.trainer.optimizers[0].param_groups[0]['lr']
            self.logger_human.info(f"Current LR: {current_lr:.2e}")
        elif batch_idx % 100 == 0:
            self.logger_human.info(f"Epoch {self.current_epoch} - batch {batch_idx}")

    def on_train_epoch_end(self):
        if self.global_rank != 0:
            return
        train_loss = self.trainer.callback_metrics.get('train/total_loss')
        if train_loss is not None:
            self.logger_human.info(f"Epoch {self.current_epoch} - Train Loss: {train_loss.item():.4f}")

    def training_step(self, batch, batch_idx):
        if batch is None:
            if self.global_rank == 0:
                self.logger_human.warning(f"Batch {batch_idx} is None. Returning dummy loss.")
            return torch.tensor(0.0, device=self.device, requires_grad=True)

        outputs = self(batch)
        total_loss = outputs["loss"]
        loss_global = outputs.get("loss_global", torch.tensor(0.0, device=self.device))
        loss_ot = outputs.get("loss_ot", torch.tensor(0.0, device=self.device))
        loss_balance = outputs.get("loss_balance", torch.tensor(0.0, device=self.device))

        self.log('train/total_loss', total_loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('train/loss_global', loss_global, on_step=True, on_epoch=True, sync_dist=True)
        self.log('train/loss_ot', loss_ot, on_step=True, on_epoch=True, sync_dist=True)
        self.log('train/loss_balance', loss_balance, on_step=True, on_epoch=True, sync_dist=True)

        if torch.isnan(total_loss):
            if self.global_rank == 0:
                self.logger_human.warning(f"Loss is NaN at batch {batch_idx}. Returning dummy loss.")
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        return total_loss 
    
    def configure_optimizers(self):
        # 1. 参数分组逻辑
        decay_params = []
        no_decay_params = []
        prompt_params=[]
        # 这里的白名单包含你希望“不衰减”的参数名字关键字
        # prompt_embeddings, class_embedding, pos_embed 等通常不衰减
        # bias 和 LayerNorm (ln_, bn_) 通常也不衰减
        no_decay_keywords = ['bias', 'LayerNorm', 'ln_', 'bn_', 'embed', 'centers']

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            
            # 检查参数名是否包含不衰减的关键字
            if any(key in name for key in no_decay_keywords):
                no_decay_params.append(param)
            elif 'prompts' in name:
                prompt_params.append(param)
            else:
                # 剩下的主要是 Adapter 的 Linear.weight
                decay_params.append(param)

        # 2. 定义参数组
        optim_groups = [
            {
                "params": decay_params, 
                "weight_decay": self.config.optimizer.get("weight_decay", 0.01) # Adapter 使用 0.01
            },
            {
                "params": no_decay_params, 
                "weight_decay": 0.0  # Prompt 和 Bias 不衰减
            },
            {
                "params": prompt_params,
                "weight_decay": 0.0001  # Prompt 不衰减
            }
        ]

        # 3. 创建优化器
        optimizer = torch.optim.AdamW(
            optim_groups,
            lr=self.config.optimizer.get("lr", 1e-4)
        )

        max_epochs = self.config.trainer.max_epochs
        scheduler = CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=1e-6)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            }
        }
        


    def on_validation_epoch_start(self):
        self.is_sanity_check = self.trainer.sanity_checking
        self.validation_outputs.clear()

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            if batch is None:
                return {}
            if 'images' not in batch or 'text_tokens' not in batch:
                return {}

            images = batch['images'].to(self.device)
            text_tokens = batch['text_tokens'].to(self.device)

            if 'image_id' in batch:
                image_ids = batch['image_id']
            elif 'imageid' in batch:
                image_ids = batch['imageid']
            else:
                image_ids = torch.full((len(images),), -1, device=self.device)

            if not isinstance(image_ids, torch.Tensor):
                image_ids = torch.tensor(image_ids)
            image_ids = image_ids.to(self.device)

            if self.is_sanity_check and 'zeroshot' in self.helper_models:
                zeroshot_model = self.helper_models['zeroshot']
                image_features = zeroshot_model.encode_image(images)
                text_features = zeroshot_model.encode_text(text_tokens)
            else:
                image_features = self.model.encode_image(images)
                text_features = self.model.encode_text(text_tokens)

            image_features = F.normalize(image_features, dim=-1)
            text_features = F.normalize(text_features, dim=-1)

        self.validation_outputs.append({
            "image_features": image_features.cpu(),
            "text_features": text_features.cpu(),
            "image_ids": image_ids.cpu(),
        })
        return {}

    def on_validation_epoch_end(self):
        if self.trainer.is_global_zero and self.validation_outputs:
            self._compute_validation_metrics_simple()
        else:
            for metric in ['RSUM', 'i2t_r1', 'i2t_r5', 'i2t_r10', 't2i_r1', 't2i_r5', 't2i_r10']:
                self.log(f'val/{metric}', 0.0, sync_dist=True, on_epoch=True)

        self.validation_outputs.clear()

        if torch.distributed.is_initialized():
            torch.distributed.barrier()

    def _compute_validation_metrics_simple(self):
        """兼容 Flickr30k (顺序) 和 COCO (ID) 的验证指标计算"""
        if not self.validation_outputs:
            return
            
        # 1. 聚合数据
        all_image_features = torch.cat([output["image_features"] for output in self.validation_outputs]).to(self.device)
        all_text_features = torch.cat([output["text_features"] for output in self.validation_outputs]).to(self.device)
        all_image_ids = torch.cat([output["image_ids"] for output in self.validation_outputs]).to(self.device)
        
        # 2. 智能判断使用哪种计算逻辑
        # 检查是否包含有效的 Image ID (即不全为 -1)
        has_valid_ids = (all_image_ids >= 0).any()
        
        if has_valid_ids:
            # === 方案 A: COCO 模式 (基于 ID 匹配) ===
            if self.global_rank == 0:
                print(f"检测到有效 image_id，使用 ID 匹配模式进行评估 (适合 COCO)...")
            recalls, num_unique_images = self._calculate_recalls_by_id(
                all_image_features, all_text_features, all_image_ids
            )
        else:
            # === 方案 B: Flickr30k 旧模式 (基于顺序切片) ===
            if self.global_rank == 0:
                print(f"未检测到 image_id，使用顺序切片模式进行评估 (适合 Flickr30k)...")
            recalls, num_unique_images = self._calculate_recalls_sequential(
                all_image_features, all_text_features
            )

        if not recalls:
            return

        rsum = sum([recalls['i2t_r1'], recalls['i2t_r5'], recalls['i2t_r10'],
                    recalls['t2i_r1'], recalls['t2i_r5'], recalls['t2i_r10']])

        self.log(f'val/RSUM', rsum, prog_bar=True, sync_dist=True, on_epoch=True)
        self.log(f'val/i2t_r1', recalls['i2t_r1'], sync_dist=True, on_epoch=True)
        self.log(f'val/i2t_r5', recalls['i2t_r5'], sync_dist=True, on_epoch=True)
        self.log(f'val/i2t_r10', recalls['i2t_r10'], sync_dist=True, on_epoch=True)
        self.log(f'val/t2i_r1', recalls['t2i_r1'], sync_dist=True, on_epoch=True)
        self.log(f'val/t2i_r5', recalls['t2i_r5'], sync_dist=True, on_epoch=True)
        self.log(f'val/t2i_r10', recalls['t2i_r10'], sync_dist=True, on_epoch=True)

        if self.global_rank == 0:
            self.logger_human.info(f"Epoch {self.current_epoch} - Unique images: {num_unique_images} - Results:")
            self.logger_human.info(f"  RSUM: {rsum:.2f}")
            self.logger_human.info(f"  I2T R@1/5/10: {recalls['i2t_r1']:.2f}/{recalls['i2t_r5']:.2f}/{recalls['i2t_r10']:.2f}")
            self.logger_human.info(f"  T2I R@1/5/10: {recalls['t2i_r1']:.2f}/{recalls['t2i_r5']:.2f}/{recalls['t2i_r10']:.2f}")

    def _calculate_recalls_by_id(self, all_image_features, all_text_features, all_image_ids):
        """基于 image_id 的精确匹配 (COCO Style)"""
        # 转为 numpy
        image_ids_np = all_image_ids.cpu().numpy()
        
        # 1. 提取唯一图片
        unique_ids, unique_indices = np.unique(image_ids_np, return_index=True)
        num_unique_images = len(unique_ids)
        
        if num_unique_images == 0:
            return {}, 0

        unique_image_features = all_image_features[unique_indices]
        
        # 2. 计算相似度
        # [Text_N, Image_M]
        logits_per_text = torch.matmul(all_text_features, unique_image_features.t())
        logits_per_image = logits_per_text.t()

        # 3. 生成 Mask
        unique_ids_tensor = torch.tensor(unique_ids, device=self.device)
        # ground_truth[i][j] = True 表示 Text[i] 属于 Image[j]
        ground_truth_mask = (all_image_ids.unsqueeze(1) == unique_ids_tensor.unsqueeze(0))

        # 4. 计算指标
        t2i_results = self._compute_recall_from_logits(logits_per_text, ground_truth_mask)
        i2t_results = self._compute_recall_from_logits(logits_per_image, ground_truth_mask.t())

        return {**t2i_results, **i2t_results}, num_unique_images
    def _calculate_recalls_sequential(self, all_image_features, all_text_features):
        """基于固定顺序 (1图5文) 的匹配 (Flickr30k Old Style)"""
        captions_per_image = 5
        total_samples = all_image_features.shape[0]
        
        # 确保样本数能被 5 整除，否则切掉多余的
        num_images = total_samples // captions_per_image
        if total_samples % captions_per_image != 0:
            limit = num_images * captions_per_image
            all_image_features = all_image_features[:limit]
            all_text_features = all_text_features[:limit]
            
        if num_images == 0:
            return {}, 0
            
        # 1. 提取唯一图片特征 (每隔5个取1个)
        unique_image_features = all_image_features[::captions_per_image]
        
        # 2. 计算相似度 [Text(N), Image(N/5)]
        logits_per_text = torch.matmul(all_text_features, unique_image_features.t())
        logits_per_image = logits_per_text.t()
        
        # 3. 生成 Mask (基于索引关系)
        # 第 i 个文本 对应 第 i // 5 张图
        # 我们手动构造一个 Mask，效果等同于 ID 匹配
        device = logits_per_text.device
        
        # text_indices: [0, 1, 2, ..., N-1] -> target_img_idx: [0, 0, 0, 0, 0, 1, 1, ...]
        text_indices = torch.arange(total_samples, device=device)
        target_img_indices = text_indices // captions_per_image
        
        # gallery_indices: [0, 1, ..., num_images-1]
        gallery_indices = torch.arange(num_images, device=device)
        
        # Mask: [N, num_images]
        ground_truth_mask = (target_img_indices.unsqueeze(1) == gallery_indices.unsqueeze(0))
        
        # 4. 复用同一个计算函数
        t2i_results = self._compute_recall_from_logits(logits_per_text, ground_truth_mask)
        i2t_results = self._compute_recall_from_logits(logits_per_image, ground_truth_mask.t())
        
        return {**t2i_results, **i2t_results}, num_images
    def _compute_recall_from_logits(self, logits, ground_truth_mask):
        """Vectorized recall computation. Returns r@1, r@5, r@10 as percentages."""
        max_k = min(10, logits.shape[1])  # handle small gallery (e.g. DDP sanity check)
        if max_k < 1:
            return {}
        _, top_indices = logits.topk(max_k, dim=1)
        extracted_corrects = torch.gather(ground_truth_mask, 1, top_indices)

        results = {}
        prefix = 'i2t_' if logits.shape[0] < logits.shape[1] else 't2i_'
        for k in [1, 5, 10]:
            if k <= max_k:
                score = extracted_corrects[:, :k].any(dim=1).float().mean().item() * 100
            else:
                score = 0.0
            results[f'{prefix}r{k}'] = score

        return results
    
