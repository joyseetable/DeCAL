#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""gap_metric.py — Decompose the CLIP modality gap into diagonal vs full-rank components.

WHAT THIS MEASURES
------------------
CLIP retrieval uses ONE d-dim vector per image/text (the projected CLS/EOT token),
NOT a token sequence. "diagonal vs full-rank" describes the transform applied to
those d CHANNELS:
  diagonal : y_i = a_i * x_i + b_i   (channel i independent, no channel mixing)
  full-rank: y_i = sum_j W_ij * x_j  (channels mixed)

The modality gap = how the visual-feature distribution differs from the
text-feature distribution. It splits into three separable parts:

  center      c = ||mu_v - mu_t|| / sqrt(d)          per-channel mean diff  -> `b` fixes
  scale       s = ||log sigma_v - log sigma_t|| / sqrt(d)  per-channel var diff -> `a` fixes
  orientation o = ||R_v - R_t||_F / d,  R=corr       cross-channel structure  -> only full-rank W fixes

A diagonal transform (a, b) can change per-channel mean/var but NOT the
between-channel correlation, so c/s are "diagonal" and o is "full-rank".

PRIMARY EVIDENCE: c + s >> o  =>  the gap is diagonal, so per-channel affine suffices.

EXPLORATORY (not primary): closed-form statistical alignments (centering /
z-score / whitening) and their effect on RSUM. NOTE: these are unsupervised
distribution alignments, NOT our trained task-driven affine — and whitening
(full-rank) in particular is destructive. This motivates "diagonal-only",
but the proper diagonal-vs-full-rank *retrieval* comparison needs a TRAINED
full-rank adapter (separate experiment), not whitening.

Reuses:
  * clip (official): encode_image/encode_text return UN-normalized features.
  * eval_utils: evaluate_i2t / evaluate_t2i / compute_rsum  (R@1/5/10 + RSUM).

Usage:
  python eval/gap_metric.py --clip_name ViT-B/32 --num_samples 5000 --device cuda
"""
import argparse
import os
import sys

import torch
import torch.nn.functional as F
from torchvision.datasets import CocoCaptions

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import clip                                       # official CLIP (un-normalized encode)
from eval_utils import evaluate_i2t, evaluate_t2i, compute_rsum  # reused metrics


# --------------------------------------------------------------------------- #
# Feature extraction (un-normalized, batched)
# --------------------------------------------------------------------------- #
def extract_features(model, preprocess, dataset, num_samples, device, batch=64):
    images, captions = [], []
    for i, (image, caps) in enumerate(dataset):
        if i >= num_samples:
            break
        images.append(image)
        captions.extend(caps[:5])

    img_feats, txt_feats = [], []
    with torch.no_grad():
        for i in range(0, len(images), batch):
            imgs = torch.stack([preprocess(im) for im in images[i:i + batch]]).to(device)
            img_feats.append(model.encode_image(imgs).cpu())
        for i in range(0, len(captions), batch):
            toks = clip.tokenize(captions[i:i + batch], truncate=True).to(device)
            txt_feats.append(model.encode_text(toks).cpu())

    Xv = torch.cat(img_feats, dim=0)   # (N, d)
    Xt = torch.cat(txt_feats, dim=0)   # (5N, d)
    return Xv.float(), Xt.float()      # float32 for stable stats / whitening


# --------------------------------------------------------------------------- #
# Gap components
# --------------------------------------------------------------------------- #
def _corr(X):
    Xc = X - X.mean(dim=0, keepdim=True)
    cov = Xc.t() @ Xc / (X.shape[0] - 1)
    s = cov.diag().sqrt().clamp(min=1e-8)
    return cov / (s[:, None] * s[None, :])


def gap_components(Xv, Xt):
    d = Xv.shape[1]
    mu_v, mu_t = Xv.mean(dim=0), Xt.mean(dim=0)
    center = (mu_v - mu_t).norm().item() / (d ** 0.5)

    sigma_v = Xv.std(dim=0) + 1e-8
    sigma_t = Xt.std(dim=0) + 1e-8
    scale = (torch.log(sigma_v) - torch.log(sigma_t)).norm().item() / (d ** 0.5)

    Rv, Rt = _corr(Xv), _corr(Xt)
    orientation = (Rv - Rt).norm().item() / d

    return {'Center': center, 'Scale': scale, 'Orientation': orientation}


# --------------------------------------------------------------------------- #
# Closed-form statistical alignments (EXPLORATORY only)
# --------------------------------------------------------------------------- #
def _center(X):
    return X - X.mean(dim=0, keepdim=True)


def _zscore(X):
    mu = X.mean(dim=0, keepdim=True)
    sigma = X.std(dim=0, keepdim=True) + 1e-8
    return (X - mu) / sigma


def _whiten(X):
    Xc = X - X.mean(dim=0, keepdim=True)
    cov = Xc.t() @ Xc / (X.shape[0] - 1)
    d = X.shape[1]
    cov = cov + 1e-2 * cov.diag().mean() * torch.eye(d)   # ridge for stability
    eigvals, eigvecs = torch.linalg.eigh(cov)
    eigvals = eigvals.clamp(min=1e-6)
    W = eigvecs @ torch.diag(1.0 / eigvals.sqrt()) @ eigvecs.t()
    return Xc @ W


# --------------------------------------------------------------------------- #
# Retrieval (reuses eval_utils recall + RSUM)
# --------------------------------------------------------------------------- #
def retrieval(Xv, Xt):
    Xv = F.normalize(Xv, dim=-1)
    Xt = F.normalize(Xt, dim=-1)
    sim = Xv @ Xt.t()                      # (N, 5N)
    n_img = Xv.shape[0]
    i2t = evaluate_i2t(sim, n_img)         # R@1, R@5, R@10 (%)
    t2i = evaluate_t2i(sim, n_img)
    rsum = compute_rsum(i2t, t2i)
    return i2t, t2i, rsum


# --------------------------------------------------------------------------- #
# Plot
# --------------------------------------------------------------------------- #
def plot_gap(components, rsum0, closed_form, out_path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    keys = ['Scale', 'Center', 'Orientation']
    labels = ['Scale\n(scale)', 'Center\n(shift)', 'Orientation\n(full-rank)']
    diag_colors = ['#1b7837', '#1b7837', '#888888']   # green=diagonal, grey=full-rank

    fig, ax = plt.subplots(figsize=(5.5, 4.2))

    vals = [components[k] for k in keys]
    ax.bar(labels, vals, color=diag_colors, width=0.6)
    ax.set_ylabel('Normalized Magnitude')
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, max(vals) * 1.35)
    for x, v in zip(labels, vals):
        ax.text(x, v + 0.01, f'{v:.3f}', ha='center', fontsize=9)

    ax.set_title('Modality Gap Decomposition — CLIP ViT-B/32 (MSCOCO val)', fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches='tight', facecolor='white')
    print(f'saved {out_path}')


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--clip_name', default='ViT-B/32')
    p.add_argument('--num_samples', type=int, default=5000)
    p.add_argument('--device', default='cuda')
    p.add_argument('--root', default='/data/clc/Long-CLIP/data/coco/val2017')
    p.add_argument('--annFile',
                   default='/data/clc/Long-CLIP/data/coco/annotations/captions_val2017.json')
    p.add_argument('--out', default='gap_decomposition.png')
    args = p.parse_args()

    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f'device={device}  clip={args.clip_name}  num_samples={args.num_samples}')

    model, preprocess = clip.load(args.clip_name, device=device)
    model.eval()

    # torchvision 0.23 CocoCaptions calls transforms(image, target); pass none, preprocess manually.
    dataset = CocoCaptions(root=args.root, annFile=args.annFile)

    Xv, Xt = extract_features(model, preprocess, dataset, args.num_samples, device)
    print(f'features: image {tuple(Xv.shape)}  text {tuple(Xt.shape)}')

    comps = gap_components(Xv, Xt)

    # zeroshot baseline retrieval (reference)
    i2t0, t2i0, r0 = retrieval(Xv, Xt)

    # exploratory: closed-form statistical alignments
    _, _, r_c = retrieval(_center(Xv), _center(Xt))
    _, _, r_s = retrieval(_zscore(Xv), _zscore(Xt))
    _, _, r_o = retrieval(_whiten(Xv), _whiten(Xt))
    closed_form = {'base': r0, 'Center': r_c, 'Scale': r_s, 'Orient': r_o}

    # ---- print ----
    print('\n' + '=' * 72)
    print('PRIMARY — Gap components (normalized magnitude)')
    print('=' * 72)
    for k, lab in [('Center', 'Center      (per-channel mean diff,  diag)'),
                   ('Scale', 'Scale       (per-channel var diff,  diag)'),
                   ('Orientation', 'Orientation (cross-channel structure, FULL-RANK)')]:
        print(f'  {lab:46s} {comps[k]:.4f}')
    print(f'  {"diagonal total (center+scale)":46s} {comps["Center"] + comps["Scale"]:.4f}')
    print(f'  {"full-rank (orientation)":46s} {comps["Orientation"]:.4f}')
    print(f'  -> diagonal/full-rank ratio = '
          f'{(comps["=Center"] + comps["Scale"]) / max(comps["Orientation"], 1e-8):.2f}x')

    print('\n' + '=' * 72)
    print('REFERENCE — zeroshot baseline retrieval')
    print('=' * 72)
    print(f'  I2T R@1/5/10 = {i2t0[0]:.2f} / {i2t0[1]:.2f} / {i2t0[2]:.2f}')
    print(f'  T2I R@1/5/10 = {t2i0[0]:.2f} / {t2i0[1]:.2f} / {t2i0[2]:.2f}')
    print(f'  RSUM = {r0:.2f}')

    print('\n' + '=' * 72)
    print('EXPLORATORY — closed-form statistical alignment vs RSUM')
    print('  (NOT our trained affine; whitening is destructive to CLIP structure)')
    print('=' * 72)
    print(f'  baseline            RSUM = {r0:.2f}')
    print(f'  -center             RSUM = {r_c:.2f}   (delta {r_c - r0:+.2f})')
    print(f'  -scale (z-score)    RSUM = {r_s:.2f}   (delta {r_s - r_c:+.2f})')
    print(f'  -orient (whiten)    RSUM = {r_o:.2f}   (delta {r_o - r_s:+.2f})')
    print('=' * 72)

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.out)
    plot_gap(comps, r0, closed_form, out)


if __name__ == '__main__':
    main()
