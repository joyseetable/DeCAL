#!/usr/bin/env python3
"""Training-free gradient and perturbation diagnostics for BSI.

Measured from an existing checkpoint, without any training:

  (1) Gradient norm per layer and per injection point.  After one backward pass
      of the training objective, the norm of the gradient of every learnable
      modulation parameter is recorded and grouped two ways: by transformer
      layer (are deep layers starved?) and by injection point (does one point
      dominate?).  A flat profile means the modulation receives a well-scaled
      learning signal at every depth.

  (2) Feature perturbation.  How far the adapted embeddings move from the frozen
      CLIP embeddings, by cosine similarity and relative L2 displacement,
      compared with the modality gap that the adaptation is meant to close.

The objective is the one the manuscript defines,
    L = L_con + lambda * L_ref,   lambda = 0.1,
with L_con the symmetric InfoNCE on adapted features and L_ref the per-modality
cosine penalty against the frozen encoders.  L_ref is reconstructed here because
the model file in this repository does not implement it.

Usage:
    python eval/grad_metric.py --checkpoint <ckpt> [--device cpu]
"""
import argparse
import json
import os
import re
import sys

import numpy as np
import torch
import torch.nn.functional as F
import clip

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.systems.retrieval_system import DistillSystem

LAMBDA_REF = 0.1
F30K_JSON = '/data/clc/data/flickr30k/annotations/test.json'
F30K_ROOT = '/data/clc/data/flickr30k/images'

# vis_ln_att.3.visual_a  ->  branch=vis, point=ln_att, layer=3
PAT = re.compile(r'^(vis|txt)_(ln_att|att_proj|ln_mlp|mlp_proj)\.(\d+)\.')
PAT_LAST = re.compile(r'^vis_last_ln\.')


def load_pairs(num):
    data = json.load(open(F30K_JSON))['images']
    pairs = []
    for it in data:
        p = os.path.join(F30K_ROOT, it.get('filename', ''))
        sents = it.get('sentences', [])
        if os.path.exists(p) and sents:
            pairs.append((p, sents[0]['raw']))
        if len(pairs) >= num:
            break
    return pairs


def collate(pairs, preprocess, device):
    from PIL import Image
    imgs = torch.stack([preprocess(Image.open(p).convert('RGB')) for p, _ in pairs]).to(device)
    toks = clip.tokenize([t for _, t in pairs], truncate=True).to(device)
    return imgs, toks


def infonce(v, t, logit_scale):
    s = logit_scale.exp().clamp(max=100.0) * v @ t.t()
    y = torch.arange(v.shape[0], device=v.device)
    return (F.cross_entropy(s, y) + F.cross_entropy(s.t(), y)) / 2


def group_grads(model):
    """name -> gradient norm, grouped by layer and by injection point."""
    by_layer, by_point = {}, {}
    for name, p in model.named_parameters():
        if not p.requires_grad or p.grad is None:
            continue
        g = float(p.grad.norm())
        m = PAT.match(name)
        if m:
            branch, point, layer = m.group(1), m.group(2), int(m.group(3))
            by_layer[f'{branch}_l{layer}'] = by_layer.get(f'{branch}_l{layer}', 0.0) + g
            by_point[f'{branch}_{point}'] = by_point.get(f'{branch}_{point}', 0.0) + g
        elif PAT_LAST.match(name):
            by_point['vis_last_ln'] = by_point.get('vis_last_ln', 0.0) + g
    return by_layer, by_point


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--clip_name', default='ViT-B/32')
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--batches', type=int, default=6)
    ap.add_argument('--batch_size', type=int, default=32)
    ap.add_argument('--perturb_samples', type=int, default=512)
    ap.add_argument('--out', default='grad_metric_results.json')
    args = ap.parse_args()

    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f'device={device}')

    system = DistillSystem.load_from_checkpoint(args.checkpoint, strict=False)
    model = system.model.to(device).eval()
    model.use_all_gather = False
    frozen, preprocess = clip.load(args.clip_name, device=device)
    frozen.eval()

    pairs = load_pairs(max(args.batches * args.batch_size, args.perturb_samples))
    print(f'pairs available: {len(pairs)}')

    # ---- (1) gradient norms ----------------------------------------------
    layers_acc, points_acc, contrib = {}, {}, {'con': [], 'ref': []}
    for b in range(args.batches):
        chunk = pairs[b * args.batch_size:(b + 1) * args.batch_size]
        if len(chunk) < 2:
            break
        imgs, toks = collate(chunk, preprocess, device)

        v = model.encode_image(imgs)
        t = model.encode_text(toks)
        with torch.no_grad():
            v_ref = F.normalize(frozen.encode_image(imgs), dim=-1)
            t_ref = F.normalize(frozen.encode_text(toks), dim=-1)

        l_con = infonce(v, t, model.logit_scale)
        l_ref = (1 - (v * v_ref).sum(-1)).mean() + (1 - (t * t_ref).sum(-1)).mean()

        # separate backward passes first (retain_graph), combined pass last,
        # because the final backward frees the graph
        model.zero_grad(set_to_none=True)
        l_con.backward(retain_graph=True)
        n_con = sum(float(p.grad.norm()) ** 2 for p in model.parameters()
                    if p.requires_grad and p.grad is not None) ** 0.5
        model.zero_grad(set_to_none=True)
        (LAMBDA_REF * l_ref).backward(retain_graph=True)
        n_ref = sum(float(p.grad.norm()) ** 2 for p in model.parameters()
                    if p.requires_grad and p.grad is not None) ** 0.5
        model.zero_grad(set_to_none=True)
        (l_con + LAMBDA_REF * l_ref).backward()
        bl, bp = group_grads(model)
        for k, val in bl.items():
            layers_acc.setdefault(k, []).append(val)
        for k, val in bp.items():
            points_acc.setdefault(k, []).append(val)

        contrib['con'].append(n_con)
        contrib['ref'].append(n_ref)
        print(f'  batch {b+1}/{args.batches}  L_con={float(l_con):.4f}  L_ref={float(l_ref):.4f}'
              f'  |g|_con={n_con:.3f}  |g|_ref={n_ref:.3f}')

    # ---- (2) feature perturbation ----------------------------------------
    print('\ncomputing feature perturbation ...')
    pert = {'cos_v': [], 'cos_t': [], 'rel_v': [], 'rel_t': [], 'gap_vt': []}
    with torch.no_grad():
        for b in range(0, args.perturb_samples, 64):
            chunk = pairs[b:b + 64]
            if not chunk:
                break
            imgs, toks = collate(chunk, preprocess, device)
            v = model.encode_image(imgs)
            t = model.encode_text(toks)
            v_ref = F.normalize(frozen.encode_image(imgs), dim=-1)
            t_ref = F.normalize(frozen.encode_text(toks), dim=-1)
            pert['cos_v'] += (v * v_ref).sum(-1).cpu().tolist()
            pert['cos_t'] += (t * t_ref).sum(-1).cpu().tolist()
            pert['rel_v'] += ((v - v_ref).norm(dim=-1) / v_ref.norm(dim=-1)).cpu().tolist()
            pert['rel_t'] += ((t - t_ref).norm(dim=-1) / t_ref.norm(dim=-1)).cpu().tolist()
            pert['gap_vt'] += (v_ref - t_ref).norm(dim=-1).cpu().tolist()

    def stat(x):
        a = np.asarray(x)
        return {'mean': float(a.mean()), 'std': float(a.std())}

    summary = {
        'checkpoint': args.checkpoint,
        'clip_name': args.clip_name,
        'lambda_ref': LAMBDA_REF,
        'batches': args.batches,
        'batch_size': args.batch_size,
        'grad_norm_by_layer': {k: stat(v) for k, v in sorted(layers_acc.items())},
        'grad_norm_by_point': {k: stat(v) for k, v in sorted(points_acc.items())},
        'grad_total': {'contrastive': stat(contrib['con']), 'reference': stat(contrib['ref'])},
        'perturbation': {k: stat(v) for k, v in pert.items()},
    }
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.out)
    json.dump(summary, open(out, 'w'), indent=2)
    print(f'\nwrote {out}\n')

    print('--- Feature perturbation vs frozen CLIP ---')
    for k in ('cos_v', 'cos_t', 'rel_v', 'rel_t', 'gap_vt'):
        s = summary['perturbation'][k]
        print(f'  {k:7s} mean={s["mean"]:.4f}  std={s["std"]:.4f}')
    print('\n--- Gradient norm by layer ---')
    for k, s in summary['grad_norm_by_layer'].items():
        print(f'  {k:8s} {s["mean"]:.4f} ± {s["std"]:.4f}')
    print('\n--- Gradient norm by injection point ---')
    for k, s in summary['grad_norm_by_point'].items():
        print(f'  {k:14s} {s["mean"]:.4f} ± {s["std"]:.4f}')
    print('\n--- Total parameter-gradient norm ---')
    for k, s in summary['grad_total'].items():
        print(f'  {k:12s} {s["mean"]:.4f} ± {s["std"]:.4f}')


if __name__ == '__main__':
    main()
