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

## 尚未驗證

- 改 radius 後的實際訓練效果（以上皆為固定權重的靜態分析）
- far background 是否可由 appearance/幾何特徵區分（未做可分性測試）
- positive 分數偏低的成因（是被組內競爭壓低，還是正向訊號本身不足）
- `Lrandsoft` 長期卡在 12.5–12.7 的確切機制（推論為梯度相消，未直接量測）
