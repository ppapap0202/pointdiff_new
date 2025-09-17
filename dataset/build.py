import torchvision.transforms as standard_transforms
from .dataset import ImageDataset
from torch.utils.data import random_split
import torch
def loading_data(cfg):
    data_set = ImageDataset(
        root=cfg.data_root,
        mode='points',  # 你的標註是點
        tile_size=(256, 256),
        stride=(128, 128),
        pad_if_needed=True,
        image_exts=('.jpg', '.png'),
        gray=cfg.gray,
    )
    train_len = int(0.8 * len(data_set))
    val_len = len(data_set) - train_len
    g = torch.Generator().manual_seed(cfg.seed)
    train_dataset, val_dataset = random_split(data_set, [train_len, val_len], generator=g)
    return train_dataset,val_dataset