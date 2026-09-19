"""
Forked CLIP model with MAIL (Multi-modal Adaptation via Inter-layer Learning).
Adds ResidualAttentionBlock_MAIL — transformer blocks that accept per-layer
learnable projections injected into the attention and MLP paths.
"""
from collections import OrderedDict
from typing import Tuple, Union
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


# ═══════════════════════════════════════════
# Shared base classes (unchanged from CLIP)
# ═══════════════════════════════════════════

class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu3 = nn.ReLU(inplace=True)
        self.downsample = None
        self.stride = stride
        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            self.downsample = nn.Sequential(OrderedDict([
                ("-1", nn.AvgPool2d(stride)),
                ("0", nn.Conv2d(inplanes, planes * self.expansion, 1, stride=1, bias=False)),
                ("1", nn.BatchNorm2d(planes * self.expansion)),
            ]))
    def forward(self, x):
        identity = x
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu3(out)
        return out


class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim, embed_dim, num_heads, output_dim=None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim**2 + 1, embed_dim) / embed_dim**0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads
    def forward(self, x):
        x = x.flatten(start_dim=2).permute(2, 0, 1)
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)
        x = x + self.positional_embedding[:, None, :].to(x.dtype)
        x, _ = F.multi_head_attention_forward(
            query=x[:1], key=x, value=x, embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight, k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight, in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None, bias_v=None, add_zero_attn=False, dropout_p=0,
            out_proj_weight=self.c_proj.weight, out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True, training=self.training, need_weights=False,
        )
        return x.squeeze(0)


class ModifiedResNet(nn.Module):
    def __init__(self, layers, output_dim, heads, input_resolution=224, width=64):
        super().__init__()
        self.output_dim = output_dim
        self.input_resolution = input_resolution
        self.conv1 = nn.Conv2d(3, width // 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width // 2)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(width // 2, width // 2, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width // 2)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(width)
        self.relu3 = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(2)
        self._inplanes = width
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)
        embed_dim = width * 32
        self.attnpool = AttentionPool2d(input_resolution // 32, embed_dim, heads, output_dim)
    def _make_layer(self, planes, blocks, stride=1):
        layers = [Bottleneck(self._inplanes, planes, stride)]
        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes))
        return nn.Sequential(*layers)
    def forward(self, x):
        def stem(x):
            for conv, bn, relu in [(self.conv1, self.bn1, self.relu1),
                                    (self.conv2, self.bn2, self.relu2),
                                    (self.conv3, self.bn3, self.relu3)]:
                x = relu(bn(conv(x)))
            x = self.avgpool(x)
            return x
        x = x.type(self.conv1.weight.dtype)
        x = stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.attnpool(x)
        return x


class LayerNorm(nn.LayerNorm):
    def forward(self, x):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(1.702 * x)


# ═══════════════════════════════════════════
# MAIL-Modified Transformer Blocks
# ═══════════════════════════════════════════

class ResidualAttentionBlock(nn.Module):
    """Original CLIP block — kept for compatibility."""
    def __init__(self, d_model, n_head, attn_mask=None):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model)),
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask
    def attention(self, x):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]
    def forward(self, x):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class ResidualAttentionBlock_MAIL(nn.Module):
    """
    MAIL block — injects learnable projections into attention & MLP paths.

    Each block receives per-layer projections:
      - ln_proj_att:  pre-attention layer-norm projection
      - att_proj:     post-attention affine transform
      - ln_proj_mlp:  pre-MLP layer-norm projection
      - mlp_proj:     post-MLP affine transform

    These are shared across layers (deep-cloned from a single MAILProjection).
    """
    def __init__(self, d_model, n_head, attn_mask=None, text_layer=False, layer_idx=0):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model)),
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask
        self.text_layer = text_layer
        self.layer_idx = layer_idx

    def attention(self, x):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, packed):
        """
        Args:
            packed: [x, counter, ln_proj_mlp, ln_proj_att, mlp_proj, att_proj,
                     spatial_gates (optional)]
        Returns:
            [x, counter+1, ln_proj_mlp, ln_proj_att, mlp_proj, att_proj,
             spatial_gates]
        """
        if len(packed) == 7:
            x, counter, ln_mlp, ln_att, mlp_proj, att_proj, spatial_gates = packed
        else:
            x, counter, ln_mlp, ln_att, mlp_proj, att_proj = packed
            spatial_gates = None

        if counter >= len(att_proj):
            # Beyond MAIL-injected layers — standard forward
            x = x + self.attention(self.ln_1(x))
            x = x + self.mlp(self.ln_2(x))
        else:
            att_p = att_proj[counter]
            mlp_p = mlp_proj[counter]
            ln_a = ln_att[counter]
            ln_m = ln_mlp[counter]

            # ── Spatial gate: compute per-head gate_a from pre-AL features ──
            if spatial_gates is not None:
                sgm = spatial_gates[counter]
                gates = sgm(x)  # [B, 4, D] — 4 heads: ln_mlp, ln_att, mlp_proj, att_proj
                ln_m.spatial_gate_a = gates[:, 0, :]  # [B, D]
                ln_a.spatial_gate_a = gates[:, 1, :]  # [B, D]
                mlp_p.spatial_gate_a = gates[:, 2, :]  # [B, D]
                att_p.spatial_gate_a = gates[:, 3, :]  # [B, D]

            # Attention path with MAIL
            x = x + att_p(
                self.attention(ln_a(self.ln_1(x), is_text=self.text_layer, i=counter)),
                is_text=self.text_layer, i=counter,
            )
            # MLP path with MAIL
            x = x + mlp_p(
                self.mlp(ln_m(self.ln_2(x), is_text=self.text_layer, i=counter)),
                is_text=self.text_layer, i=counter,
            )

        return [x, counter + 1, ln_mlp, ln_att, mlp_proj, att_proj, spatial_gates]


class Transformer(nn.Module):
    """Standard CLIP Transformer (unchanged, for text branch compatibility)."""
    def __init__(self, width, layers, heads, attn_mask=None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[
            ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)
        ])
    def forward(self, x):
        return self.resblocks(x)


class Transformer_MAIL(nn.Module):
    """MAIL Transformer — uses ResidualAttentionBlock_MAIL with packed forward."""
    def __init__(self, width, layers, heads, attn_mask=None, text_layer=False):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[
            ResidualAttentionBlock_MAIL(width, heads, attn_mask, text_layer, i)
            for i in range(layers)
        ])
    def forward(self, packed):
        return self.resblocks(packed)


class VisionTransformer(nn.Module):
    """Standard CLIP ViT (unchanged)."""
    def __init__(self, input_resolution, patch_size, width, layers, heads, output_dim):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(3, width, patch_size, stride=patch_size, bias=False)
        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        n_patches = (input_resolution // patch_size) ** 2
        self.positional_embedding = nn.Parameter(scale * torch.randn(n_patches + 1, width))
        self.ln_pre = LayerNorm(width)
        self.transformer = Transformer(width, layers, heads)
        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))
    def forward(self, x):
        x = self.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(
            x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.ln_post(x[:, 0, :])
        if self.proj is not None:
            x = x @ self.proj
        return x


class VisionTransformer_MAIL(nn.Module):
    """
    MAIL ViT — accepts per-layer projection modules as extra forward args.
    The MAIL projections are injected inside each transformer block.
    """
    def __init__(self, input_resolution, patch_size, width, layers, heads, output_dim):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(3, width, patch_size, stride=patch_size, bias=False)
        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        n_patches = (input_resolution // patch_size) ** 2
        self.positional_embedding = nn.Parameter(scale * torch.randn(n_patches + 1, width))
        self.ln_pre = LayerNorm(width)
        self.transformer = Transformer_MAIL(width, layers, heads, text_layer=False)
        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    def forward(self, x, ln_mlp, ln_att, mlp_proj, att_proj, last_ln=None):
        x = self.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(
            x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)
        x = x.permute(1, 0, 2)

        packed = [x, 0, ln_mlp, ln_att, mlp_proj, att_proj]
        outputs = self.transformer(packed)
        x = outputs[0]
        x = x.permute(1, 0, 2)

        if last_ln is not None:
            x = last_ln(self.ln_post(x[:, 0, :]), is_text=False, i=12)
        else:
            x = self.ln_post(x[:, 0, :])

        if self.proj is not None:
            x = x @ self.proj
        return x


# ═══════════════════════════════════════════
# CLIP class (uses MAIL variants)
# ═══════════════════════════════════════════

class CLIP(nn.Module):
    def __init__(self, embed_dim, image_resolution, vision_layers,
                 vision_width, vision_patch_size, context_length,
                 vocab_size, transformer_width, transformer_heads,
                 transformer_layers):
        super().__init__()
        self.context_length = context_length

        if isinstance(vision_layers, (tuple, list)):
            vision_heads = vision_width * 32 // 64
            self.visual = ModifiedResNet(
                layers=vision_layers, output_dim=embed_dim,
                heads=vision_heads, input_resolution=image_resolution,
                width=vision_width,
            )
        else:
            vision_heads = vision_width // 64
            self.visual = VisionTransformer_MAIL(
                input_resolution=image_resolution,
                patch_size=vision_patch_size, width=vision_width,
                layers=vision_layers, heads=vision_heads,
                output_dim=embed_dim,
            )

        self.transformer = Transformer_MAIL(
            width=transformer_width, layers=transformer_layers,
            heads=transformer_heads, attn_mask=self.build_attention_mask(),
            text_layer=True,
        )

        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)
        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)
        if isinstance(self.visual, ModifiedResNet):
            if self.visual.attnpool is not None:
                std = self.visual.attnpool.c_proj.in_features ** -0.5
                nn.init.normal_(self.visual.attnpool.q_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.k_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.v_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.c_proj.weight, std=std)
            for resnet_block in [self.visual.layer1, self.visual.layer2,
                                 self.visual.layer3, self.visual.layer4]:
                for name, param in resnet_block.named_parameters():
                    if name.endswith("bn3.weight"):
                        nn.init.zeros_(param)
        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)
        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    def build_attention_mask(self):
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)
        return mask

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype

    def encode_image(self, image):
        return self.visual(image.type(self.dtype))

    def encode_text(self, text):
        x = self.token_embedding(text).type(self.dtype)
        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.ln_final(x).type(self.dtype)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
        return x

    def forward(self, image, text):
        image_features = self.encode_image(image)
        text_features = self.encode_text(text)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logit_scale * text_features @ image_features.t()
        return logits_per_image, logits_per_text


def convert_weights(model):
    def _convert(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.half()
            if l.bias is not None:
                l.bias.data = l.bias.data.half()
        if isinstance(l, nn.MultiheadAttention):
            for attr in ["in_proj_weight", "q_proj_weight", "k_proj_weight",
                         "v_proj_weight", "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.half()
        for name in ["text_projection", "proj"]:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None:
                    attr.data = attr.data.half()
    model.apply(_convert)


def build_model(state_dict):
    vit = "visual.proj" in state_dict
    if vit:
        vision_width = state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len([k for k in state_dict
                            if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_resolution = vision_patch_size * grid_size
    else:
        counts = [len(set(k.split(".")[2] for k in state_dict
                         if k.startswith(f"visual.layer{b}"))) for b in [1, 2, 3, 4]]
        vision_layers = tuple(counts)
        vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
        output_width = round((state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
        vision_patch_size = None
        assert output_width ** 2 + 1 == state_dict["visual.attnpool.positional_embedding"].shape[0]
        image_resolution = output_width * 32

    embed_dim = state_dict["text_projection"].shape[1]
    context_length = state_dict["positional_embedding"].shape[0]
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(set(k.split(".")[2] for k in state_dict
                                 if k.startswith("transformer.resblocks")))

    model = CLIP(embed_dim, image_resolution, vision_layers, vision_width,
                 vision_patch_size, context_length, vocab_size,
                 transformer_width, transformer_heads, transformer_layers)

    for key in ["input_resolution", "context_length", "vocab_size"]:
        if key in state_dict:
            del state_dict[key]

    convert_weights(model)
    try:
        model.load_state_dict(state_dict)
    except Exception:
        missing_keys, _ = model.load_state_dict(state_dict, strict=False)
        print(f'[clip_mail] Missing keys: {len(missing_keys)}')
    return model.eval()
