# Agent Instructions

這個專案是 Windows/PyTorch 研究實驗工作區，主要在做 PointDiff crowd counting / point proposal / diffusion refinement 相關實驗。後續 agent 請優先保留實驗脈絡與可重現性，不要把工作區當成乾淨產品 repo 來整理。

## User Preferences

- 使用繁體中文回覆，技術名詞可保留英文。
- 直接處理問題；除非資訊不足會造成高風險，否則不要停在詢問或空泛建議。
- 回報要短而具體：說明改了什麼、怎麼驗證、還有什麼風險。
- 尊重正在進行或歷史實驗。不要任意刪除、重新命名、整理 `logs/`、`vis_results/`、`config/` 裡的實驗檔案。
- 若要新增實驗設定，偏好複製現有 YAML 並用清楚命名記錄差異，例如包含 stage、epoch、batch size、DDIM、threshold、resume 範圍或主要 loss 名稱。
- 不要未經要求啟動長時間 GPU 訓練；若需要，先說明將使用的 config、輸出位置、預期副作用。

## Project Layout

- `main.py`: 訓練入口，讀取 `--config` 指向的 YAML，建立 dataset/model/optimizer，輸出 log 與 checkpoint。
- `test.py`: 推論/測試與 threshold、NMS、DDIM 相關流程。
- `validate_diagnostics.py`: coverage、duplicate、matching、prior diagnostics。
- `models/`: 模型、diffusion utilities、training loop、proposal prior。
- `dataset/`: ShanghaiTech-style image/point dataset loader 與統計工具。
- `config/`: 實驗設定。這裡的檔名本身是實驗紀錄，修改前要先確認目標檔。
- `scripts/`: 長訓練 launcher、watcher、metric/visualization 輔助腳本。
- `logs/`: PID、status、stdout/stderr、validation JSON 與 training log。這些常用來追蹤未完成或已完成實驗，不要清理。

## Environment Notes

- 預設工作目錄是 `C:\pycharm\pointdiff_new`。
- 常見 Python 環境是 `C:\Users\crf\AppData\Local\anaconda3\envs\pointdiff\python.exe`，部分 PowerShell 腳本已寫死此路徑。
- `config/train.yaml` 目前含有本機資料與輸出路徑，例如 ShanghaiTech dataset 在 `C:\pycharm\dataset\...`，checkpoint/output 可能在 `D:\output\...`。
- 主要執行環境預期有 CUDA。CPU smoke test 可以做，但不能代表訓練效能或完整正確性。

## Common Commands

```powershell
python main.py --config config\train.yaml
```

```powershell
python validate_diagnostics.py --config config\validate_density_prior_stage1_epoch0005_gate009_probe.yaml
```

```powershell
python test.py --config config\train.yaml
```

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_randcover15_match7p5_resume.ps1
```

使用這些命令前先確認 config 裡的 `ckpt_path`、`out_dir`、`data_root`、`test_root`、`batch_size`、`num_workers`、`epochs`、`ddim_steps` 與 threshold 設定。

## Editing Guidelines

- 先讀相關檔案與目標 config，再改程式。這個 repo 有不少實驗旗標，避免只改一處卻漏掉 train/validate/test 三條路徑。
- 盡量維持既有風格：單檔內多用 argparse + YAML key 覆寫、`getattr(args, "...", default)` 做向後相容、logging/print 並存。
- 新增 config key 時，需同步確認：
  - `main.py` 是否有傳入 train/validate 函式。
  - `models/train_loop.py` 是否有合理 default。
  - `test.py` 或 `validate_diagnostics.py` 是否也需要相同推論邏輯。
  - 舊 checkpoint 載入是否需要 `strict=False` 或 shape-compatible handling。
- 對 tensor shape、座標系統與半徑單位要特別保守。常見座標轉換是 pixel `<->` `[-1, 1]`，coverage/dup radius 多以 pixel 表示。
- 不要大幅重構研究程式碼，除非是為了解決明確 bug。優先做局部、可回退、可比較的改動。
- 若發現亂碼註解，除非任務要求修正文件編碼，否則不要順手改大量註解，避免製造不必要 diff。

## Experiment Hygiene

- 新實驗請把輸出寫到新的 `out_dir`，不要覆蓋既有 checkpoint 目錄。
- launcher/status 檔應寫入 `logs/`，內容至少包含 config、PID、stdout、stderr、開始時間與狀態。
- 如果要停止訓練或 watcher，先確認 PID/status 檔對應的程序，避免殺錯長跑實驗。
- 評估時優先保留原始 metric 名稱，例如 `val_ddim_candidate_cover@6`、`val_ddim_dup@6`、`val_conf_no_nms_mae`、`val_prior_cover_gain@6`，方便和既有 log 比較。
- 對 threshold sweep、NMS radius、DDIM steps、proposal prior gate/mode 的改動，要在 config 檔名或 status/log 中留下可辨識資訊。

## Verification Expectations

- 小型程式改動：至少跑語法檢查或目標腳本的 `--help`/短路徑檢查，若環境不允許就明確回報。
- 訓練邏輯改動：優先做 import/compile 檢查，再做最小 batch 或短 epoch smoke test；不要直接啟動完整訓練。
- metric 或 diagnostics 改動：用既有 JSON/log 格式檢查輸出 key 是否穩定，避免破壞後續 watcher/parser。
- 若無法跑測試，回覆時要說明原因，例如缺 dataset、缺 CUDA、缺 checkpoint 或依賴未安裝。

## Git And Workspace Safety

- 目前工作區可能長期保持 dirty 狀態。不要 revert 使用者或其他 agent 的變更。
- 不要刪除未追蹤 config、log、PID、visualization 或 checkpoint 相關檔案。
- 修改前後用 `git status --short` 檢查範圍；最終回報只列出本次新增/修改的檔案。
- 除非使用者明確要求，不要 commit、push、reset、clean 或 checkout。
