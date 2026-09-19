# 文件路径: src/datasets/__init__.py

from .flickr30k_datamoudle import PrototypeGuidedCLIPDataModule
from .mscoco_datamodule import MSCOCODatamodule # Uncomment when you create this file
from .RSICD_datamodule import RSICDDatamodule
def create_datamodule(config):
    """
    Factory function to create a DataModule based on the config.
    """
    dataset_name = config.dataset.name
    
    if 'f30k' in dataset_name or 'flickr30k' in dataset_name:
        return PrototypeGuidedCLIPDataModule(config)
    elif 'coco' in dataset_name:
        return MSCOCODatamodule(config)
    elif 'rsicd' in dataset_name:
        return RSICDDatamodule(config)
    elif 'rsitmd' in dataset_name:
        print("rsitmd被加载了")
        return RSICDDatamodule(config)
    elif 'ucm' in dataset_name:
        print("ucm被加载了")
        return RSICDDatamodule(config)
    # ... add other datasets here ...
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")