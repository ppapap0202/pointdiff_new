import torch
from tqdm import tqdm

@torch.no_grad()
def dataset_pos_neg_stats(data_loader):
    total_imgs = 0
    total_pos = 0
    total_neg = 0

    # 用來做分布統計（每張圖一個 pos、neg、ratio）
    pos_per_img = []
    neg_per_img = []
    ratio_neg_per_pos = []  # neg/pos

    for images, points_pad, mask, metas in tqdm(data_loader, desc="[stats]"):
        # mask: [B, N] (bool)
        B, N = mask.shape
        pos = mask.sum(dim=1)                  # [B]
        neg = (~mask).sum(dim=1)               # [B]

        total_imgs += B
        total_pos  += pos.sum().item()
        total_neg  += neg.sum().item()

        pos_per_img.append(pos.cpu())
        neg_per_img.append(neg.cpu())

        # 避免除0（空圖時視為 ratio = inf，可以用大數代替）
        r = (neg.float() / pos.clamp_min(1).float()).cpu()
        ratio_neg_per_pos.append(r)

    pos_per_img = torch.cat(pos_per_img)               # [num_images]
    neg_per_img = torch.cat(neg_per_img)               # [num_images]
    ratio_neg_per_pos = torch.cat(ratio_neg_per_pos)   # [num_images]

    avg_pos = pos_per_img.float().mean().item()
    avg_neg = neg_per_img.float().mean().item()
    avg_ratio = ratio_neg_per_pos.float().mean().item()

    # 也給一些分位數，幫助你抓極端情況
    q = torch.tensor([0.0, 0.25, 0.5, 0.75, 0.95, 1.0])
    pos_q = torch.quantile(pos_per_img.float(), q).tolist()
    neg_q = torch.quantile(neg_per_img.float(), q).tolist()
    ratio_q = torch.quantile(ratio_neg_per_pos.float(), q).tolist()

    print("=== Dataset Imbalance Stats ===")
    print(f"Images           : {total_imgs}")
    print(f"Avg pos / img    : {avg_pos:.2f}")
    print(f"Avg neg / img    : {avg_neg:.2f}")
    print(f"Avg neg:pos      : 1:{(avg_neg/max(avg_pos,1e-8)):.2f}  (或 neg/pos={avg_ratio:.2f})")
    print(f"Total pos        : {total_pos}")
    print(f"Total neg        : {total_neg}")
    print("Pos per image    : min/25%/50%/75%/95%/max =", [round(v,2) for v in pos_q])
    print("Neg per image    : min/25%/50%/75%/95%/max =", [round(v,2) for v in neg_q])
    print("neg/pos per image: min/25%/50%/75%/95%/max =", [round(v,2) for v in ratio_q])
