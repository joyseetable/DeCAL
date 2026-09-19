"""
Pure MAIL retrieval model — AL (affine) layers ONLY.

Injects learnable affine transforms  x -> x * a + b  into every CLIP
transformer block (attention + MLP paths, vision and text branches),
with a frozen CLIP backbone. Nothing else:

  - NO low-rank bridge (proj_down / proj_up)
  - NO alignment projectors (vis_proj / txt_proj)
  - NO spatial gate modules
  - NO teacher / distillation

Loss: standard global InfoNCE (with optional differentiable all-gather).

This is the "minimal AL-only" variant. The affine transform is defined
locally here (AffineProjection) so we do not depend on the bridge /
projector plumbing inside src.models.mail_layers.MAILProjection.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch.distributed as dist

from src.clip_mail import clip as mail_clip
from src.models.mail_variants import CoupledAffineProjection, SharedScalarProjection


# ── Standard differentiable all-gather (for global InfoNCE) ──
class GatherLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        if not dist.is_initialized():
            return x
        output = [torch.empty_like(x) for _ in range(dist.get_world_size())]
        dist.all_gather(output, x)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        if not dist.is_initialized():
            return grads[0]
        all_gradients = torch.stack(grads)
        dist.all_reduce(all_gradients)
        return all_gradients[dist.get_rank()]


def diff_all_gather(x):
    if not dist.is_initialized():
        return x
    return torch.cat(GatherLayer.apply(x), dim=0)


class AffineProjection(nn.Module):
    """
    Single AL projection layer: x -> x * a + b, one (a, b) per branch.

    a initialized to 1, b initialized to 0 -> identity at start.
    """
    def __init__(self, visual_dim, text_dim):
        super().__init__()
        self.visual_a = nn.Parameter(torch.ones(visual_dim))
        self.visual_b = nn.Parameter(torch.zeros(visual_dim))
        self.text_a = nn.Parameter(torch.ones(text_dim))
        self.text_b = nn.Parameter(torch.zeros(text_dim))

    def forward(self, x, is_text=False, i=0):
        if is_text:
            a, b = self.text_a, self.text_b
        else:
            a, b = self.visual_a, self.visual_b
        return x * a.to(x.dtype) + b.to(x.dtype)


def _make_projection(variant, visual_dim, text_dim, rank, alpha=None):
    """Instantiate one injection operator for the requested ablation variant."""
    if variant == 'independent':
        return AffineProjection(visual_dim, text_dim)
    if variant == 'bridged':
        return CoupledAffineProjection(visual_dim, text_dim, source="text", rank=rank, alpha=alpha)
    if variant == 'bridged_rev':
        return CoupledAffineProjection(visual_dim, text_dim, source="visual", rank=rank, alpha=alpha)
    if variant == 'shared_scalar':
        return SharedScalarProjection(visual_dim, text_dim)
    raise ValueError(
        f"unknown model.mail_variant '{variant}'; expected one of "
        "independent | bridged | bridged_rev | shared_scalar")


def _build_affine_layers(visual_dim, text_dim, num_layers,
                         variant='independent', rank=1, alpha=None):
    """Build 4 sets of injection operators, one per transformer layer."""
    mk = lambda: _make_projection(variant, visual_dim, text_dim, rank, alpha)
    return (
        nn.ModuleList([mk() for _ in range(num_layers)]),   # ln_mlp
        nn.ModuleList([mk() for _ in range(num_layers)]),   # ln_att
        nn.ModuleList([mk() for _ in range(num_layers)]),   # mlp_proj
        nn.ModuleList([mk() for _ in range(num_layers)]),   # att_proj
        mk(),                                               # last_ln (CLS only)
    )


class PureMAILModel(nn.Module):
    """
    Frozen CLIP + per-layer affine (AL) projections, InfoNCE retrieval.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.clip_name = config.model.get('original_clip_name', 'ViT-B/32')
        self.use_all_gather = config.model.get('use_all_gather', True)

        print(f"[PureMAIL] Loading forked CLIP: {self.clip_name}")
        self.clip, _ = mail_clip.load(self.clip_name, device='cpu', jit=False)
        self.clip.float()

        for param in self.clip.parameters():
            param.requires_grad = False

        visual_dim = self.clip.visual.conv1.out_channels   # 768 ViT-B / 1024 ViT-L
        text_dim = self.clip.transformer.width             # 512 ViT-B / 768 ViT-L
        self.vision_layers = len(self.clip.visual.transformer.resblocks)
        self.text_layers = len(self.clip.transformer.resblocks)

        # ---- ablation switch: where do the modulation parameters come from? ----
        # independent : both branches own their scale and shift   (proposed)
        # bridged     : visual parameters are generated from textual ones
        # bridged_rev : textual parameters are generated from visual ones
        # shared_scalar: one scale and one shift shared by both modalities
        self.mail_variant = config.model.get('mail_variant', 'independent')
        self.bridge_rank = config.model.get('bridge_rank', 1)
        self.bridge_alpha = config.model.get('bridge_alpha', None)

        if self.mail_variant == 'independent':
            # Vision AL — one instance per vision transformer layer
            (self.vis_ln_mlp, self.vis_ln_att,
             self.vis_mlp_proj, self.vis_att_proj,
             self.vis_last_ln) = _build_affine_layers(
                visual_dim, text_dim, self.vision_layers,
                self.mail_variant, self.bridge_rank, self.bridge_alpha)

            # Text AL — one instance per text transformer layer (no last_ln needed)
            (self.txt_ln_mlp, self.txt_ln_att,
             self.txt_mlp_proj, self.txt_att_proj, _) = _build_affine_layers(
                visual_dim, text_dim, self.text_layers,
                self.mail_variant, self.bridge_rank)
        else:
            # Coupled variants: a single set of operators serves both branches,
            # so the derived modality can read the free parameters of the same
            # injection point.  The two branches are told apart by `is_text`,
            # which the transformer block already passes to every operator.
            n = max(self.vision_layers, self.text_layers)
            (self.vis_ln_mlp, self.vis_ln_att,
             self.vis_mlp_proj, self.vis_att_proj,
             self.vis_last_ln) = _build_affine_layers(
                visual_dim, text_dim, n, self.mail_variant, self.bridge_rank, self.bridge_alpha)
            self.txt_ln_mlp = self.vis_ln_mlp
            self.txt_ln_att = self.vis_ln_att
            self.txt_mlp_proj = self.vis_mlp_proj
            self.txt_att_proj = self.vis_att_proj

        self.logit_scale = nn.Parameter(torch.ones([]) * math.log(1 / 0.07))

        self._trainable_summary()

    def _trainable_summary(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[PureMAIL] variant: {getattr(self, 'mail_variant', 'independent')}"
              + (f" (bridge_rank={self.bridge_rank})"
                 if getattr(self, 'mail_variant', '') in ('bridged', 'bridged_rev')
                 else ""))
        print(f"[PureMAIL] Vision layers: {self.vision_layers}, "
              f"Text layers: {self.text_layers}")
        print(f"[PureMAIL] Params: {trainable:,} trainable / {total:,} total "
              f"({trainable/total:.2%})")

    # ── Vision Encoder ────────────────────────────────────────
    def encode_image(self, images):
        vit = self.clip.visual
        x = vit.conv1(images.type(self.clip.dtype))
        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
        cls_token = vit.class_embedding.to(x.dtype) + torch.zeros(
            x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device)
        x = torch.cat([cls_token, x], dim=1)
        x = x + vit.positional_embedding.to(x.dtype)
        x = vit.ln_pre(x)
        x = x.permute(1, 0, 2)

        packed = [x, 0,
                  self.vis_ln_mlp, self.vis_ln_att,
                  self.vis_mlp_proj, self.vis_att_proj]
        outputs = vit.transformer(packed)
        x = outputs[0]
        x = x.permute(1, 0, 2)

        cls_feat = vit.ln_post(x[:, 0, :])
        x = self.vis_last_ln(cls_feat, is_text=False, i=12)

        if vit.proj is not None:
            x = x @ vit.proj
        return F.normalize(x, dim=-1)

    # ── Text Encoder ──────────────────────────────────────────
    def encode_text(self, text_tokens):
        x = self.clip.token_embedding(text_tokens).type(self.clip.dtype)
        x = x + self.clip.positional_embedding.type(self.clip.dtype)
        x = x.permute(1, 0, 2)

        packed = [x, 0,
                  self.txt_ln_mlp, self.txt_ln_att,
                  self.txt_mlp_proj, self.txt_att_proj]
        outputs = self.clip.transformer(packed)
        x = outputs[0]
        x = x.permute(1, 0, 2)

        x = self.clip.ln_final(x).type(self.clip.dtype)
        x = x[torch.arange(x.shape[0]), text_tokens.argmax(dim=-1)]

        if self.clip.text_projection is not None:
            x = x @ self.clip.text_projection
        return F.normalize(x, dim=-1)

    # ── Forward (InfoNCE) ─────────────────────────────────────
    def forward(self, batch):
        images = batch["images"]
        text_tokens = batch.get("text_tokens")
        device = images.device

        image_features = self.encode_image(images)

        if text_tokens is None:
            return {'image_features': image_features}

        text_features = self.encode_text(text_tokens)

        if self.use_all_gather:
            gathered_img = diff_all_gather(image_features)
            gathered_txt = diff_all_gather(text_features)
        else:
            gathered_img, gathered_txt = image_features, text_features

        logit_scale = self.logit_scale.exp().clamp(max=100.0)
        logits_i2t = logit_scale * gathered_img @ gathered_txt.t()
        logits_t2i = logits_i2t.t()
        labels = torch.arange(gathered_img.shape[0], device=device)
        loss_infonce = (F.cross_entropy(logits_i2t, labels) +
                        F.cross_entropy(logits_t2i, labels)) / 2

        return {
            'loss': loss_infonce,
            'loss_global': loss_infonce,
            'loss_ot': torch.tensor(0.0, device=device),
            'loss_balance': torch.tensor(0.0, device=device),
            'image_features': image_features,
            'text_features': text_features,
            'logits_per_image': logits_i2t,
        }
