import logging
from datetime import datetime
import os
import argparse
import yaml
from dataset import build_dataset, dataset_pos_neg_stats
from torch.utils.data import DataLoader
from models import build_model, build_optimizers, Diffusion_schedule
from models.train_loop import train_one_epoch,validate_one_epoch
import torch
import time
from visualize import visualization

# --- Logging 初始化 ---
def setup_logging():
    os.makedirs('logs', exist_ok=True)
    log_file = os.path.join('logs', f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file, mode='w', encoding='utf-8'),
            logging.StreamHandler()
        ]
    )


def parse_args():
    def load_config(yaml_path):
        with open(yaml_path, 'r', encoding="utf-8") as f:
            return yaml.safe_load(f)
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=r'config\train.yaml', type=str)
    args, remaining_argv = parser.parse_known_args()
    cfg = load_config(args.config)
    #print(cfg)
    parser = argparse.ArgumentParser(parents=[parser],add_help=False)
    for key, value in cfg.items():
        parser.add_argument(f'--{key}', type=type(value), default=value)
    args = parser.parse_args()
    return args


def data(args):
    train_data, val_data= build_dataset(args)
    return train_data,val_data

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
def main():
    #讀取參數
    device = torch.device( "cuda" if torch.cuda.is_available() else "cpu")
    import logging
    logging.info(f'device={device}')
    args = parse_args()
    #訓練資料處理
    train_data,val_data=data(args)

    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=8,  # 先試 CPU 核心數的一半
        pin_memory=True,  # 加速 CPU→GPU 拷貝
        persistent_workers=True,  # 避免每個 epoch 重啟 worker
        prefetch_factor=4,  # 每個 worker 預取 4 個 batch
        collate_fn=collate_points_padded
    )

    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate_points_padded)
    # for a,b,c in train_data:
    #     visualization(a,b,c)
    # _ = dataset_pos_neg_stats(train_loader)
    # _ = dataset_pos_neg_stats(val_loader)
    for imgs, pts, mask, metas in train_loader:
        logging.info(f'images.shape: {imgs.shape}')  # (B, C, H, W)
        logging.info(f'points.shape: {pts.shape}')  # (B, max_len, 2)
        logging.info(f'mask.shape: {mask.shape}')  # (B, max_len)
        logging.info(metas)
        break
    model = build_model(args, training=True).to(device)
    #print(model)
    optim = build_optimizers(model, lr=args.lr, lr_backbone=args.lr_backbone, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    T = args.diffusion_T
    sched, signal_scale = Diffusion_schedule(T, device=device, signal_scale=args.signal_scale)

    best_val = 1e9
    os.makedirs(args.out_dir, exist_ok=True)

    checkpoint = torch.load(r"C:\pycharm\pointdiff_new\output3\last_epoch0111.pth", map_location="cuda:0")
    #
    # # 載入模型與優化器參數
    model.load_state_dict(checkpoint['model_state'])
    optim.load_state_dict(checkpoint['optim_state'])
    scaler.load_state_dict(checkpoint['scaler_state'])

    print('start training')

    for epoch in range(1, args.epochs+1):
        time_start = time.time()
        tr_loss = train_one_epoch(model, train_loader, device, optim, scaler, sched, T)
        val_loss, val_MAE = validate_one_epoch(model, val_loader, device, sched, signal_scale, T)

        logging.info(f"[Epoch {epoch:04d}] train={tr_loss:.4f}  val={val_loss:.4f} val_MAE={val_MAE:.4f}")
        last_path = os.path.join(args.out_dir, f"last_epoch{epoch:04d}.pth")
        torch.save({
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optim_state": optim.state_dict(),
            "scaler_state": scaler.state_dict(),
            "best_val": best_val,
        }, last_path)
        if val_loss < best_val:
            best_val = val_loss
            best_path = os.path.join(args.out_dir, f"best_epoch{epoch:04d}_val{val_loss:.2f}.pth")
            print('save model',best_path)
            torch.save(model.state_dict(), best_path)
        time_end = time.time()
        print('time cost', time_end - time_start)



if __name__ == '__main__':
    setup_logging()
    main()