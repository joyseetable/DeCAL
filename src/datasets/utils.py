# 文件路径: src/datasets/utils.py

import torch
import torchvision.transforms as transforms

# The original get_transform function is often superseded by the processor,
# but can be kept for custom augmentation pipelines if needed.
# For simplicity, we'll keep it here as a reference, though our new
# DataModule will primarily rely on the Hugging Face processor.
def get_transform(split_name: str, image_size: int) -> transforms.Compose:
    """
    Returns a standard image transformation pipeline for CLIP.
    Note: The official Hugging Face CLIPProcessor includes normalization,
    so if you use the processor directly on PIL images, you might not need this.
    However, it's good practice to have explicit transforms.
    """
    # Mean and std for CLIP models, from OpenAI's implementation
    mean = [0.48145466, 0.4578275, 0.40821073]
    std = [0.26862954, 0.26130258, 0.27577711]
    
    if split_name == 'train':
        return transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.9, 1.0), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
    else:  # 'val' or 'test'
        return transforms.Compose([
            transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

# ----------------- CORRECTED FUNCTION HERE -----------------
def to_unicode(text: any) -> str:
    """
    Ensures the input text is a unicode string (str in Python 3).
    This is a modernized and simplified version of the original `convert_to_unicode`.
    It handles str and bytes types correctly in a Python 3 environment.
    """
    # In Python 3, the default string type `str` is already unicode.
    if isinstance(text, str):
        return text
    
    # If the input is bytes, decode it using utf-8.
    # The 'ignore' flag prevents errors from malformed byte sequences.
    if isinstance(text, bytes):
        return text.decode("utf-8", "ignore")
        
    # For other types (like numbers), convert them to string representation.
    return str(text)
# -----------------------------------------------------------