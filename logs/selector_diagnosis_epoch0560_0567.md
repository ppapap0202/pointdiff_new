# Selector Duplicate / Competition 診斷總結

- 對象 checkpoint：`last_epoch0560.pth`（probe 2–4）、`last_epoch0567.pth`（probe 5、label assignment）
- 來源 run：`density_only_prior_t50_conf_add64_localcomp_r3_groupassign_pos32_v1_from_epoch0550_to0650_bs32_ddim5_nonms`
- 樣本規模：120 batches × bs4 = 480 tiles，19,962 個「size ≥ 2」的 GT group
- 診斷工具（皆為唯讀，未改動訓練路徑）：
  - `scripts/analyze_selector_competition.py`（probe 2–5）
  - `scripts/analyze_exist_label_assignment.py`（label assignment / L_rand_bg）
- 輸出：`logs/selector_competition_probe_epoch0560.json`、`logs/selector_oracle_probe_epoch0567.json`、`logs/exist_label_assignment_epoch0567.json`

---

## 一句話結論

**原本設定的題目（讓 candidate 在 local group 中互相競爭以消除 duplicate）針對的是次要問題。**
duplicate 只佔錯誤的 11%，far background FP 佔 43%；而 recall 的瓶頸是 positive 分數被壓得不夠高
（只有 42.7% 過門檻），不是候選覆蓋不足（candidate_cover@6 = 0.88）。

---

## 基準指標（epoch 0560 validation）

```
val_ddim_raw_cover@6        = 0.8907
val_ddim_candidate_cover@6  = 0.8803
val_ddim_dup@6              = 3.8294
val_ddim_candidates_per_gt  = 7.1659
val_conf_no_nms_recall@6    = 0.4118
val_conf_no_nms_precision@6 = 0.4398
val_conf_no_nms_dup@6       = 0.6778
val_conf_no_nms_selected_per_gt = 0.9365
```

錯誤結構（`logs/selector_error_breakdown_r3_farbg_epoch0550.json`）：

| threshold | matched TP / gt | far background FP / gt | near duplicate FP / gt |
|---|---:|---:|---:|
| 0.5 | 0.578 | **0.725（佔 FP 72.7%）** | 0.273（27.3%） |
| 0.6 | 0.417 | **0.430（43%）** | 0.114（11%） |

`selected_per_gt = 0.94` —— **輸出數量已經是對的**，錯的是選在哪裡。

---

## Probe 2：組內 winner margin

| 指標 | 值 | 隨機基準 |
|---|---:|---:|
| winner_is_nearest | **0.3414** | **0.3094** |
| lift over random | **+0.0320** | — |
| winner 距離排名（normalized） | 0.4811 | 0.5 |
| pooled within-group corr（score vs dist） | **−0.0285** | 0 |
| per-group Spearman 平均 | −0.0617 | 0 |
| margin（logit）median | 0.516 | — |
| top1 logit median | **0.2624** | — |
| top2 logit median | −0.3579 | — |

group size：mean 4.01、median 3、max 16。
`groups_with_0_candidates = 3897`（13.7%，對應 candidate_cover 0.88 的缺口）。

**解讀**：分數分得開（margin 0.516），但排序與「誰更接近 GT」幾乎無關（corr −0.029）。
不過後來確認 **這個指標本身問錯了問題** —— group 定義是 GT 6px 內，而評估的 bipartite matching
邊也是 6px，所以組內任何成員都能匹配該 GT，選誰不影響 `matched_gt`。

真正有意義的是 `top1 logit median = 0.2624` < threshold 0.6 對應的 0.405
—— **一半的 group 連贏家都過不了門檻**，這才是 recall 0.41 的來源。

---

## Probe 3：LocalCompetitionSelector 的實際作用

| 指標 | 值 |
|---|---:|
| learned_effective_strength | **0.03019**（init 0.030，上限 0.25） |
| mean shift（有鄰居） | −0.0191 |
| mean shift（無鄰居） | +0.0029 |
| bias_ratio | 0.151（→ 確實只作用在鄰居上） |
| leader − nonleader shift | +0.0129 |
| neighbor_count median | 1（p95 = 3） |

checkpoint 軌跡（125 epochs，`strength_logit`）：

| run | epoch | strength_logit | eff_strength | score_delta 末層 \|w\|max |
|---|---:|---:|---:|---:|
| r3_v1 | 441 | −1.99252 | 0.029998 | 3.37e−04 |
| r3_v1 | 490 | −1.98389 | 0.030226 | 5.27e−03 |
| r3_farbg | 551 | −1.98052 | **0.030316（峰值）** | 7.75e−03 |
| groupassign | 565 | −1.98690 | 0.030146 | 7.09e−03 |

**解讀**：機制方向正確（只壓鄰居、leader 幾乎不動），但幅度只有 margin 的 3.7%，等於沒開。
`strength_logit` 不是被凍結（`configure_trainable_params` 與 `build_optimizers` 都正確收錄），
而是 **125 epoch 只移動 0.0056**。按 AdamW 性質，梯度若方向一致應移動 ≈ 0.069
→ **有效更新率僅 8%**，代表梯度方向劇烈震盪、正負相消。且 0551 見頂後開始回退。

---

## Probe 4：merge 的特徵平均

| pair 類型 | mean cos | p5 | n |
|---|---:|---:|---:|
| 被 merge 合併掉（merge 前） | **0.9970** | 0.9886 | 771,528 |
| merge 後同 GT group 存活者 | **0.9866** | 0.9539 | 161,821 |
| 參考 pair（不相關） | 0.8906 | 0.3779 | 221,712 |

- `merge_destroyed_fraction_of_available_spread = 0.0275`
- cluster_size mean 2.28、max 63；**每張圖 900 slot 被 merge 成約 394（掉 56%）**

**解讀**：merge **無害** —— 被合併掉的 pair 本來就幾乎是同一個向量。
但同組存活者的相似度仍達 0.9866，換算可用差異只剩
`(1−0.9866)/(1−0.8906) = 12.2%` —— **`PointConditioner`（patch=5 @ P4 stride=4）
產生的 `pro` 特徵本身就對 2–6px 位移不敏感**，這是結構性的。

---

## Probe 5：特徵 oracle 上限

「若 head 完美學會單獨使用該特徵，winner_is_nearest 的天花板」（19,962 groups，隨機 0.3094）：

| 準則 | 上限 | vs 隨機 |
|---|---:|---:|
| nn1_dist | 0.3471 | +0.0377 |
| selector 目前分數 | 0.3414 | +0.0320 |
| **centroid_dist** | **0.3102** | **+0.0008** |
| centroid_rank | 0.3040 | −0.0054 |
| local_scale | 0.2873 | −0.0220 |
| prior_density | 0.2856 | −0.0237 |
| prior_occupancy | 0.2789 | −0.0305 |

**解讀**：所有特徵天花板都 < 0.35。已實作的相對幾何特徵（質心位移）oracle 僅 +0.0008，
與隨機無異 —— 「局部質心 ≈ GT」的假設不成立。
`prior_density` / `prior_occupancy` 甚至是**反指標**，而 selector 目前正在吃它們作為輸入。

---

## Label assignment：`exist_pos_radius` 的影響

複製訓練的 random-start branch（prior → `forward_noisy(t=50)` → 單次 denoise）。

| `exist_pos_radius` | positive | duplicate | background | dup >6px | 佔 dup | 其中過 thr 0.6 |
|---|---:|---:|---:|---:|---:|---:|
| **32（現況）** | 26,792 | 172,624 | 60,417 | **118,142** | **68.4%** | 6.6% |
| 8 | 25,923 | 80,036 | 153,874 | 25,554 | 31.9% | 15.2% |
| 6 | 24,746 | 54,482 | 180,605 | 0 | 0% | — |

各 role 的分數分佈（radius 32）：

| 標籤 | n | p50 | p75 | p95 |
|---|---:|---:|---:|---:|
| positive | 26,792 | **0.530** | 0.776 | 0.952 |
| duplicate ≤6px | 54,482 | 0.388 | 0.589 | 0.860 |
| duplicate >6px | 118,142 | **0.140** | 0.289 | 0.663 |
| background（>32px） | 60,417 | **0.005** | 0.019 | 0.115 |

`positive_over_thresh_0.6 = 0.427` ← 直接對應 `recall = 0.41`。

**解讀**：68.4% 的 role=2「duplicate」其實距離最近 GT 超過 6px（median 8.58px、p95 26px）。
它們**有**被壓（0.140），但力道遠弱於真正的 background 標籤（0.005，差 28 倍）。
`0.066 × 118,142 = 7,797` 個誤標點越過門檻，相對於 `0.43 × 26,792 = 11,521` 的 far background FP
總量 —— **約佔 68%**。

---

## `L_rand_bg` 量級（重要：發現一個 dead config）

`train_loop.py:2101-2108`：

```python
if rand_target_roles is not None:
    bg_mask = rand_valid_bool[b] & (rand_target_roles[b] == 0)   # ← 實際走這條
else:
    bg_mask = ... & (dmin >= ignore_radius)                       # ← rand_bg_ignore_radius 在這
```

`exist_label_mode: nearest_gt_group` ⇒ `role_labels_available=True` ⇒
**`rand_bg_ignore_radius: 32.0` 從未被讀取**。bg 樣本集合完全由 `exist_pos_radius` 決定。

`L_rand_bg = mean(prob)` over `roles==0`，`lambda_rand_bg = 10.0`：

| radius | bg 點數 | bg mean prob | L_rand_bg | ×10 貢獻 | 相對現況 |
|---|---:|---:|---:|---:|---:|
| **32（現況）** | 60,417 | 0.0243 | 0.0243 | 0.243 | 1.00× |
| **8** | 153,874 | 0.1142 | 0.1142 | **1.142** | **4.69×** |
| 6 | 180,605 | 0.1482 | 0.1482 | 1.482 | 6.09× |

總 loss ≈ 28（`Lrandsoft` 12.6、`Lcnt` 7.9、`Lcollapse` 2.25、`Lrandgrp` 1.82）。
改 radius 8 的絕對增量 **+0.90，佔總 loss 3.2%** —— 量級安全，不需調整 `lambda_rand_bg`。

**風險**：bg mean prob 從 0.024 → 0.114 是因為新納入的 6–8px 點本來分數就高，其中混有
positive 的同人鄰居。`L_rand_bg` 是無差別 `prob.mean()`（無 focal、無 top-k），
會均勻壓低所有 bg 點 —— 在 recall 已是瓶頸（positive 僅 42.7% 過門檻）的情況下有拖累風險。

---

## 診斷過程中被推翻的假設

| # | 假設 | 依據 | 結果 |
|---|---|---|---|
| 1 | 同組 candidate 特徵不可分辨（4e） | 靜態讀碼 | ❌ margin median 0.516，分得開 |
| 2 | `score_delta` 是全域 bias（4b） | 靜態讀碼 | ❌ bias_ratio 0.151，確實只作用於鄰居 |
| 3 | `strength_logit` 被凍結 | 靜態讀碼 | ❌ trainable 與 optimizer 都正確 |
| 4 | merge 抹掉了組內差異 | 有 probe2/3 數據後仍判斷錯 | ❌ 只摧毀 2.75% |
| 5 | 「選組內最近者」是關鍵目標 | 貫穿 probe 2–5 | ❌ 組內都在 6px 內，選誰都能匹配 |
| 6 | 6–32px 的點「從未被當背景罰」 | 上一輪推論 | ⚠️ 有被罰，但力道弱 28 倍 |
| 7 | `rand_bg_ignore_radius` 需要調整 | 上一輪建議 | ❌ dead config，改了無效 |

---

## 建議優先序（修正後）

1. **`exist_pos_radius: 32 → 8`**（唯一真正生效的半徑，同時控制 label assignment 與 bg 樣本集合）
   - 預期消掉約 68% 的 far background FP
   - positive 只損失 3.2%（26,792 → 25,923）；radius 6 會損失 7.6%，不建議
   - `lambda_rand_bg` 維持 10.0
2. **對齊競爭尺度**：`rand_soft_compete_radius` / `_bg_radius` 32 → 8，`_winner_radius` 32 → 6
3. **recall 的正向訊號** —— positive p50 僅 0.530，需要獨立處理，不能只靠壓 bg
4. duplicate 相關（組內 normalize、相對幾何特徵）**押後** —— 只佔 11%，且 oracle 顯示可用訊號極少

`selector-relative-geometry` 分支上的 18 維相對幾何特徵已通過工程驗證
（zero-init 等價、梯度正常、退化輸入安全），但 probe 5 顯示其核心假設無效，**不建議作為下一個實驗**。

---

## 尚未驗證（第一階段結束時，下方第二階段已逐一驗證）

- 改 radius 後的實際訓練效果（以上皆為固定權重的靜態分析）
- far background 是否可由 appearance/幾何特徵區分（未做可分性測試）
- positive 分數偏低的成因（是被組內競爭壓低，還是正向訊號本身不足）
- `Lrandsoft` 長期卡在 12.5–12.7 的確切機制（推論為梯度相消，未直接量測）

---
---

# 第二階段：實驗與再診斷

第一階段內容寫於 k-fold 驗證之前。之後跑了四輪訓練實驗與四項診斷，
**其中推翻了第一階段的多項建議**，以下逐一記錄。

新增工具：
- `scripts/analyze_near_far_separability.py`（linear probe + k-fold）
- `scripts/analyze_inference_selection.py`（推論端選點演算法比較 + recall 分解）
- `scripts/analyze_group_mass.py`（組內機率質量）

新增輸出：`logs/near_far_separability_kfold_epoch0551.json`、
`logs/inference_selection_epoch0551.json`、`logs/group_mass_epoch0551.json`、
`logs/recall_decomposition_epoch0551.json`

---

## 樣本規模的表述問題（方法論修正）

第一階段多處用「88,030 slots」「640,282 slots」描述樣本規模，掩蓋了真正的獨立單位是
**image**。第一版可分性測試只有 **23 張圖**（test 僅 7 張），而 3 張圖的 smoke test
顯示 selector AUC 會在 **0.5224–0.7057** 之間擺盪 —— 與後來量到的 gap 同一量級。

**之後所有測試一律先報告 image 數，slot 數僅作次要資訊。**

---

## 四輪訓練實驗（全部從 `last_epoch0551.pth` 起跑，可直接對照）

基準（epoch 0551）：`recall 0.4393  precision 0.4289  dup@6 0.7273  conf_mae 105.4  val_MAE 95.34`

| 實驗 | 改動 | 結果 |
|---|---|---|
| pos8 | `exist_pos_radius` 32→8、soft_compete 對齊 8/6 | recall →**0.3641**，mae →133.7 |
| pos8+bgtopk | 加 `rand_bg_topk=32`、`lambda_rand_bg` 10→2 | recall →**0.3834**，mae →119.2 |
| A (dupw025) | `exist_duplicate_weight` 1.0→0.25 | 峰值 0.4557@0552 → 衰退至 **0.4347**@0575 |
| B (randexist) | `lambda_exist` 40→10、`lambda_rand_exist` 8→24 | 進行中 |

**第一階段的第 1、2 項建議（`exist_pos_radius 32→8`、對齊 soft_compete）經實驗證明有害。**

### 實驗 A 的兩個發現

1. 假設**部分成立**：`selected_per_gt` 1.0244→1.0846，`val_MAE` 一度到 **91.84**（全程最佳）。
   但 duplicate 同步上升（0.7273→0.7633）—— **模型無法選擇性地只抬 positive**。
2. 衰退原因：`Lranddup` 從 0.2246 升到 0.2520+ 並持續高位。
   **組內抑制至少有四個來源**（exist CE、`rand_dup`、`localcomp`、`soft_compete`），
   鬆開一個，其他會補上。

### 新增的程式改動（皆向後相容，預設值等同舊行為）

- `models/train_loop.py`：`rand_bg_topk`（預設 0 = 原本的無加權 mean）
- `models/diffusion_utils.py`：`exist_duplicate_weight`（預設 1.0，實測 diff = 0.00e+00）
  兩條 exist loss 路徑（`criterion.forward` 與 `p2p_exist_loss`）都已套用

---

## near/far 可分性：k-fold（182 images / 640,282 slots / 5-fold / 隨機 image split）

任務：預測「該 candidate 在某個 GT 的 6px 內」。

| 特徵集 | 維度 | test AUC |
|---|---:|---:|
| conf_feat（進 head 前） | 64 | **0.8247 ± 0.0163** |
| pro + relgeom + prior | 84 | 0.8244 ± 0.0165 |
| prior（occ+density） | **2** | 0.8092 ± 0.0151 |
| pro | 64 | 0.7890 ± 0.0181 |
| relgeom | 18 | 0.6196 ± 0.0063 |

| 單特徵 | test AUC |
|---|---:|
| prior_density | **0.8139 ± 0.0144** |
| selector 實際分數 | **0.6436 ± 0.0185** |
| relgeom_centroid_dist | 0.4153（反指標） |

**配對比較：`conf_feat` probe 比 selector 高 `+0.1811 ± 0.0083`，5/5 folds 全勝。**

`ConfidenceHead` 是 64→256→1 的非線性網路，容量嚴格大於打敗它的單層 linear probe。
**資訊已經送到 head 門口，是訓練把 head 帶去了別的地方 —— 不需要動 conditioner 或 backbone。**

但書：selector 的訓練目標本來就不是 near/far 二分類，
`build_region_representative_targets` 要求每組只有一個 positive，
所以 0.18 的落差有一部分是刻意的，不全是浪費。

---

## 組內機率質量（182 images / 98,831 GT groups）

`mass = Σ sigmoid(score)`，範圍為該 GT 6px 內、且以它為最近 GT 的 candidates。

```
[rand branch 訓練]  mass mean=1.279 median=1.203  over=0.556  in[0.8,1.2]=0.178
[ddim branch 推論]  mass mean=1.050 median=0.985  over=0.493  in[0.8,1.2]=0.218
```

**推論分支的 mass 中位數是 0.985 —— 總量已經正確。** quota loss 的施力空間比預期小得多。

兩分支的密集區趨勢**相反**（rand: 11+ 鄰居 1.276；ddim: 0.922），
quota 在 rand 上壓密集區會讓 ddim 的密集區更不足。

---

## 推論端選點演算法（182 images / 98,831 GT，不需重新訓練）

| selector | MAE | recall | prec |
|---|---:|---:|---:|
| ORACLE topk（真實 GT 數） | 0.00 | 0.4777 | 0.4777 |
| **topk by density count** | **80.86** | 0.4594 | 0.4724 |
| mass_transfer r2 | 126.55 | 0.4991 | 0.4718 |
| **固定 threshold（現況）** | **138.41** | 0.3941 | 0.4837 |
| threshold + NMS r4 | 164.46 | 0.3600 | 0.4938 |
| ORACLE mass 集中 | 215.49 | 0.6032 | 1.0000 |
| adaptive NMS（4 個 alpha） | 146–268 | 0.23–0.38 | — |

- **density count top-k 把 MAE 從 138.41 降到 80.86（−42%），完全不需訓練**
- adaptive NMS（`r = alpha*4/sqrt(density)`）四個 alpha **全部比不做還差**，放棄
- 隨機 bias 對照：recall 每升一分 precision 就掉一分，MAE 從 116 惡化到 316

---

## recall 損失分解（threshold sweep，182 images / 98,831 GT）

```
coverage   = GT 6px 內完全沒有候選點的比例       (threshold 無關)
magnitude  = (1 - coverage) - I_recall           (集中後 mass 仍過不了門檻)
dispersion = I_recall - A_recall                 (mass 夠但分散在多點)
```

| thr | A_recall | I_recall | coverage | magnitude | dispersion | A_prec | A_MAE |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.10 | **0.8793** | 0.6773 | 0.099 | 0.224 | **0.000** | 0.155 | 2547 |
| 0.20 | 0.8691 | 0.6744 | 0.099 | 0.227 | 0.000 | 0.195 | 1881 |
| 0.30 | 0.8293 | 0.6662 | 0.099 | 0.235 | 0.000 | 0.259 | 1195 |
| 0.40 | 0.7317 | 0.6517 | 0.099 | 0.249 | 0.000 | 0.339 | 630 |
| 0.50 | 0.5739 | 0.6302 | 0.099 | 0.271 | 0.056 | 0.418 | 218 |
| 0.60 | 0.3941 | 0.6032 | 0.099 | 0.298 | **0.209** | 0.484 | 138 |

**三個關鍵結論：**

1. **`dispersion` 在 thr ≤ 0.4 時為 0** —— 低 threshold 下 `A_recall` 反超 `I_recall`。
   「組內無法集中」不是獨立的失敗機制，**幾乎完全是 threshold 0.6 訂太高的表現**。
2. **`coverage` 是 0.0990，不是先前寫的 0.170。** 那個 0.170 來自 `group_mass` 腳本的
   `nearest_gt_only` 分組（candidate 被鄰近 GT 搶走就算 empty），不是覆蓋的正確定義。
3. **recall 從來不是瓶頸。** thr 0.1 時 recall 已達 0.8793，離覆蓋上限 0.901 只差 2%。
   代價全在 precision（0.155）。**模型該找的都找到了，問題是分不出哪些是真的。**

---

## 第二階段推翻的假設

| # | 假設 | 推翻它的證據 |
|---|---|---|
| 8 | `exist_pos_radius 32→8` 能消 68% far-bg FP | 實驗：recall 0.4393→0.3641 |
| 9 | `L_rand_bg` 的無加權 mean 是 recall 下降主因 | 改 top-k 後 recall 照樣掉 |
| 10 | 弱化 role=2 能選擇性提升 positive | duplicate 同步上升，24 epoch 後衰退 |
| 11 | `prior_density` 對 near/far 無用 | k-fold 單特徵 AUC **0.8139**（probe 5 測的是另一個任務） |
| 12 | 「組內無法集中」是獨立的失敗機制 | threshold sweep：dispersion 在 thr≤0.4 時為 **0** |
| 13 | slot embedding 值得投入 | 同上 + 隨機 bias 對照顯示 MAE 只會惡化 |
| 14 | 覆蓋損失是 17.0% | 正確定義下是 **9.9%** |
| 15 | adaptive NMS（density 推導半徑）可行 | 4 個 alpha 全部比不做更差 |
| 16 | oracle mass 集中 recall 可達 0.80 | 那是 1 張圖的 smoke test；182 圖實測 **0.6032** |

---

## 目前的結論

**確定的**

- recall 不是瓶頸（thr 0.1 可達 0.879，覆蓋上限 0.901）
- **precision 才是**：模型吐出 5.7 倍的點，無法區分哪些是真的
- `conf_feat` 內含 0.8247 AUC 的 near/far 資訊，selector 只用出 0.6436（5/5 folds，gap ±0.0083）
- `ConfidenceHead` 容量足夠 —— 不需要動 conditioner / backbone
- 六輪 loss 改動（半徑×2、bg topk、dupw、randexist、以及更早的 groupassign）
  沒有一次帶來穩定改善
- 推論端 density count top-k 不需訓練即可讓 MAE −42%

**唯一未被否證的方向**

讓 selector 把 `conf_feat` 裡已有的 near/far 資訊用出來。三條獨立證據都指向它：
probe 的 0.18 AUC 落差、mass 顯示總量已對而分佈不對、分解顯示問題在 precision 而非 recall。

**已否證的方向**：slot embedding、group quota、adaptive NMS、相對幾何特徵、
以及所有調整半徑的作法。
