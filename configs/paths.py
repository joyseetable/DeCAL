"""
Centralized path configuration for GROUT-MAIL.
Overridable via environment variables.
Usage:
    from configs.paths import PATHS
    data_dir = PATHS['flickr30k_images']
"""
import os

_BASE = os.environ.get("GROUT_DATA_ROOT", "/data/clc")

PATHS = {
    # ── Flickr30K ──
    "flickr30k_images":       os.environ.get("FLICKR30K_IMAGES",       f"{_BASE}/data/flickr30k/images"),
    "flickr30k_annotations":  os.environ.get("FLICKR30K_ANNOTATIONS",  f"{_BASE}/data/flickr30k/annotations"),

    # ── MS-COCO ──
    "coco_val2017":           os.environ.get("COCO_VAL2017",           f"{_BASE}/Long-CLIP/data/coco/val2017"),
    "coco_annotations":       os.environ.get("COCO_ANNOTATIONS",       f"{_BASE}/Long-CLIP/data/coco/annotations/captions_val2017.json"),

    # ── RSICD ──
    "rsicd_images":           os.environ.get("RSICD_IMAGES",           f"{_BASE}/data/RSICD/images"),
    "rsicd_annotations":      os.environ.get("RSICD_ANNOTATIONS",      f"{_BASE}/data/RSICD/annocations/dataset_rsicd.json"),

    # ── RSITMD ──
    "rsitmd_images":          os.environ.get("RSITMD_IMAGES",          f"{_BASE}/data/RSIMTD/images"),
    "rsitmd_annotations":     os.environ.get("RSITMD_ANNOTATIONS",     f"{_BASE}/data/RSIMTD/dataset_RSITMD.json"),

    # ── Clustering / Prototypes ──
    "cluster_centers_160":    os.environ.get("CLUSTER_CENTERS_160",    f"{_BASE}/APSE-IPIK/NEW_APSEIPIK/offline/clustering_results/cluster_centers_160.npy"),
    "cluster_centers_vitl14": os.environ.get("CLUSTER_CENTERS_VITL14", f"{_BASE}/APSE-IPIK/NEW_APSEIPIK/offline/clustering_results/cluster_centers_vitl14_160.npy"),
    "rsi_cluster_centers":    os.environ.get("RSI_CLUSTER_CENTERS",    f"{_BASE}/APSE-IPIK/NEW_APSEIPIK/offline/RSI_clustering_results/cluster_centers_vitb32_RSI_32.npy"),

    # ── Checkpoints ──
    "checkpoint_flickr":      os.environ.get("CKPT_FLICKR",           f"{_BASE}/SSP/pth/RSUM=136.88.ckpt"),
    "checkpoint_rsicd":       os.environ.get("CKPT_RSICD",            f"{_BASE}/SSP/pth/RSUM=109.32.ckpt"),
    "checkpoint_rsitmd":      os.environ.get("CKPT_RSITMD",           f"{_BASE}/SSP/pth/RSUM=109.32.ckpt"),
}
