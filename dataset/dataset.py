import os
import glob
import csv
from typing import Tuple, List, Optional
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

class ImageDataset(Dataset):
    """
    支援三種標註型態：
      - mode='density': 讀取密度圖 (npy 或 png)，會同步裁切
      - mode='points' : 讀取 CSV 點標註 (每列: x,y)，會同步裁切並位移，超出裁切框的點會被濾除
      - mode='class'  : 讀取文字分類標籤 (單一整數)
    參數:
      root: e.g. 'dataset/train'，其下需有 images/ 與 labels/
      crop_size: (H, W) 要裁切的尺寸；若影像小於裁切尺寸會自動 zero-padding 再裁
      crop_type: 'random' 或 'center'
      image_exts: 允許的影像副檔名
      density_exts: 允許的密度圖副檔名（當 mode='density' 時使用）
      normalize: 是否將影像轉為 [0,1] 的 float tensor
    """
    def __init__(
        self,
        root: str,
        mode: str = 'points',            # 'points' | 'density' | 'class'
        tile_size: Tuple[int, int] = (256, 256),
        stride: Tuple[int, int] = (256, 256),
        pad_if_needed: bool = True,
        image_exts=('.jpg', '.jpeg', '.png'),
        density_exts=('.npy', '.png'),
        gray=True
    ):
        assert mode in ('points', 'density', 'class')
        self.root = root
        self.mode = mode
        self.th, self.tw = tile_size
        self.sh, self.sw = stride
        self.pad_if_needed = pad_if_needed
        self.image_dir = os.path.join(root, 'images')
        self.label_dir = os.path.join(root, 'ground_truth')
        self.image_exts = image_exts
        self.gray = gray
        self.density_exts = density_exts
        #print(self.image_dir,self.label_dir)
        # 收集影像
        self.img_paths = []
        for ext in image_exts:
            self.img_paths += glob.glob(os.path.join(self.image_dir, f'*{ext}'))
        self.img_paths.sort()
        if not self.img_paths:
            raise FileNotFoundError(f'No images in {self.image_dir}')

        # 展開所有 tile 索引
        self.index_map = []
        self._precompute_tiles()

    # ---------- label 路徑匹配（重點：points 支援 .txt 與 .csv） ----------
    def _match_label_path(self, img_path: str) -> Optional[str]:
        base = os.path.splitext(os.path.basename(img_path))[0]
        if self.mode == 'density':
            for ext in self.density_exts:
                p = os.path.join(self.label_dir, base + ext)
                if os.path.exists(p):
                    return p
            return None
        elif self.mode == 'points':
            # 先找 .txt，再找 .csv
            #print(base)
            p_txt = os.path.join(self.label_dir, base + '.txt')
            if os.path.exists(p_txt):
                return p_txt
            p_csv = os.path.join(self.label_dir, base + '.csv')
            return p_csv if os.path.exists(p_csv) else None
        else:  # class
            p = os.path.join(self.label_dir, base + '.txt')
            return p if os.path.exists(p) else None

    # ---------- 讀取 points from TXT or CSV ----------
    @staticmethod
    def _load_points_txt(txt_path: str) -> np.ndarray:
        """
        逐行解析 'x y'（以空白分隔）。允許多個空白、空行。
        """
        pts = []
        with open(txt_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()  # 任何空白分隔
                if len(parts) >= 2:
                    try:
                        x = float(parts[0]); y = float(parts[1])
                        pts.append([x, y])
                    except ValueError:
                        # 跳過非數值行
                        continue
        if len(pts) == 0:
            return np.zeros((0, 2), dtype=np.float32)
        return np.array(pts, dtype=np.float32)

    @staticmethod
    def _load_points_csv(csv_path: str) -> np.ndarray:
        pts = []
        with open(csv_path, 'r', newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) >= 2:
                    try:
                        x = float(row[0]); y = float(row[1])
                        pts.append([x, y])
                    except ValueError:
                        continue
        if len(pts) == 0:
            return np.zeros((0, 2), dtype=np.float32)
        return np.array(pts, dtype=np.float32)

    # 根據副檔名呼叫對應 reader
    def _load_points(self, path: Optional[str]) -> np.ndarray:
        if path is None:
            return np.zeros((0, 2), dtype=np.float32)
        if path.lower().endswith('.txt'):
            return self._load_points_txt(path)
        if path.lower().endswith('.csv'):
            return self._load_points_csv(path)
        # 其他副檔名當作 txt 解析
        return self._load_points_txt(path)

    # ---------- grid & tiles ----------
    def _compute_grid(self, H: int, W: int):
        th, tw, sh, sw = self.th, self.tw, self.sh, self.sw
        if self.pad_if_needed:
            import math
            n_rows = 1 if H <= th else math.ceil((H - th) / sh) + 1
            n_cols = 1 if W <= tw else math.ceil((W - tw) / sw) + 1
            tops  = [min(r * sh, max(0, H - th)) for r in range(n_rows)]
            lefts = [min(c * sw, max(0, W - tw)) for c in range(n_cols)]
        else:
            if H < th or W < tw:
                return []
            n_rows = 1 + (H - th) // sh
            n_cols = 1 + (W - tw) // sw
            tops  = [r * sh for r in range(n_rows)]
            lefts = [c * sw for c in range(n_cols)]
        return [(t, l) for t in tops for l in lefts]

    def _precompute_tiles(self):
        for img_idx, p in enumerate(self.img_paths):
            with Image.open(p) as im:
                W, H = im.size
            for j, (top, left) in enumerate(self._compute_grid(H, W)):
                self.index_map.append((img_idx, top, left, j))

    # ---------- Dataset API ----------
    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, global_idx: int):
        img_idx, top, left, j = self.index_map[global_idx]
        img_path = self.img_paths[img_idx]
        label_path = self._match_label_path(img_path)
        if self.mode == 'points':
            if label_path is None or not os.path.exists(label_path):
                print(f"[LABEL MISSING] {img_path}  ->  expected at: {self.label_dir}")
        # 影像
        if self.gray:
            img = Image.open(img_path).convert('L')  # 灰階
        else :
            img = Image.open(img_path).convert('RGB')  # 彩色
        img_t = TF.to_tensor(img)  # (C,H,W) in [0,1]
        C, H, W = img_t.shape

        th, tw = self.th, self.tw
        need_ph = max(0, top + th - H)
        need_pw = max(0, left + tw - W)
        if (need_ph > 0 or need_pw > 0) and self.pad_if_needed:
            img_t = TF.pad(img_t, (0, 0, need_pw, need_ph), fill=0)

        # 標註
        if self.mode == 'points':
            pts = self._load_points(label_path)  # (N,2), 原圖座標系（x,y）
        elif self.mode == 'density':
            if label_path is None:
                raise FileNotFoundError(f'No density label for {img_path}')
            if label_path.endswith('.npy'):
                den = np.load(label_path).astype(np.float32)
            else:
                den = np.array(Image.open(label_path).convert('F'), dtype=np.float32)
            den_t = torch.from_numpy(den).unsqueeze(0)
            if (need_ph > 0 or need_pw > 0) and self.pad_if_needed:
                den_t = TF.pad(den_t, (0, 0, need_pw, need_ph), fill=0)
        else:  # class
            if label_path is None:
                raise FileNotFoundError(f'No class label for {img_path}')
            with open(label_path, 'r', encoding='utf-8') as f:
                cls = int(f.read().strip())

        # 切 tile
        img_tile = img_t[:, top:top+th, left:left+tw]

        if self.mode == 'points':
            # 只保留落在 tile 的點並位移到 tile 座標（0~tw/0~th）
            if pts.size == 0:
                pts_in = pts
            else:
                x, y = pts[:, 0], pts[:, 1]
                in_x = (x >= left) & (x < left + tw)
                in_y = (y >= top)  & (y < top + th)
                mask = in_x & in_y
                pts_in = pts[mask].copy()
                pts_in[:, 0] -= left
                pts_in[:, 1] -= top
            label_out = torch.from_numpy(pts_in.astype(np.float32))  # (N,2)

        elif self.mode == 'density':
            den_tile = den_t[:, top:top+th, left:left+tw]
            label_out = den_tile

        else:  # class
            label_out = torch.tensor(cls, dtype=torch.long)

        meta = {
            'image_path': img_path,
            'orig_size': (H, W),
            'tile_top': int(top),
            'tile_left': int(left),
            'tile_size': (th, tw),
            'img_index': int(img_idx),
            'tile_index_in_img': int(j),
        }

        return img_tile, label_out, meta