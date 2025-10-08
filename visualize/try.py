import logging
from datetime import datetime
import os
import argparse
import yaml
from pointdiff_new.dataset import build_dataset, dataset_pos_neg_stats
from pointdiff_new.dataset import dataset_pos_neg_stats
import torch
import time
from pointdiff_new.dataset.dataset import ImageDataset
from pointdiff_new.visualize import visualization
from torch.utils.data import DataLoader
def data(args):
    train_data, val_data= build_dataset(args)
    return train_data,val_data
def parse_args():
    def load_config(yaml_path):
        with open(yaml_path, 'r', encoding="utf-8") as f:
            return yaml.safe_load(f)
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=r'C:\pycharm\pointdiff_new\config\train.yaml', type=str)
    args, remaining_argv = parser.parse_known_args()
    cfg = load_config(args.config)
    #print(cfg)
    parser = argparse.ArgumentParser(parents=[parser],add_help=False)
    for key, value in cfg.items():
        parser.add_argument(f'--{key}', type=type(value), default=value)
    args = parser.parse_args()
    return args
def collate_points_padded(batch):
    import torch
    imgs, pts, metas = zip(*batch)
    imgs = torch.stack(imgs, 0)  # (B,C,H,W)

    # 計算此 batch 內最大點數
    max_n = 900#max(p.size(0) for p in pts)
    B = len(pts)
    padded = torch.full((B, max_n, 2), fill_value=-10.0)  # padding 用 -10
    mask = torch.zeros((B, max_n), dtype=torch.bool)

    for i, p in enumerate(pts):
        n = p.size(0)
        if n > 0:
            padded[i, :n] = p
            mask[i, :n] = True

    return imgs, padded, mask, list(metas)

args = parse_args()

dataset = ImageDataset(
    root=args.test_root,
    mode='points',
    tile_size=(256, 256),
    stride=(256, 256),
    gray=False,
    pad_if_needed=True,
    image_exts=('.jpg', '.png'),
    )

dataset = ImageDataset(
    root=args.test_root,
    mode='points',
    tile_size=(256, 256),
    stride=(256, 256),
    gray=False,
    pad_if_needed=True,
    image_exts=('.jpg', '.png'),
    )
train_data, val_data = data(args)


for a,b,c in dataset:
    visualization(a,b,c)