# 评价指标设计

四类指标。Reconstruction 是主结果；其余三类既是 manipulation check，也是机制回归的自变量。

---

## 1. Reconstruction（主结果，outcome — 激活空间）

衡量目标激活 `h_l` 与重建 `ĥ_l = AR(T(z))` 的接近程度。**在激活向量上度量,不是文本**（NLA 自编码激活,见 paper_findings.md）。

| 指标 | 定义 | 说明 |
|------|------|------|
| `fve` | `1 − ‖h_l − ĥ_l‖² / ‖h_l − h̄_l‖²` | **论文主指标**。0=预测均值,1=完美。论文达 0.6–0.8 |
| `activation_cosine` | `cos(h_l, ĥ_l)` | 方向重建(尺度无关,FVE 的补充) |
| `activation_mse` | `‖h_l − ĥ_l‖²` | FVE 里的 loss ℒ |

> `h̄_l = E[h_l]` 为评测集的均值激活（FVE 的 baseline）,在评测集上算一次缓存。

**派生主结果**：
- `fve`：主指标。
- `Δfve_k = fve(C0) − fve(Ck)`：相对 original 的配对下降（主分析量）。
- `fve_norm = (fve − fve_floor) / (fve_C0 − fve_floor)`：用 C9(random text) 做 floor 归一化。

---

## 2. Semantic similarity（manipulation check）

验证「语义保持/漂移」操纵是否成功，并作为机制回归的语义协变量。

| 指标 | 定义 | 用途 |
|------|------|------|
| `sim_zz'` | `cos(emb(z), emb(z'))`（解释**文本**） | 改写语义保持度（核心 check） |
| `nli_entail` | z↔z' 的 NLI 双向蕴含分数 | 更严格的语义等价判定（可选） |

> 注意:此处语义相似度在**解释文本** `z` 上算;reconstruction(FVE)在**激活**上算,二者不同空间。

**Manipulation check 判据（建议阈值，pilot 后校准）**：
- 语义保持条件（C1/C2/C3/C5/C6）：`sim_zz' ≥ τ_keep`（如 0.85）。不达标样本剔除或单列。
- 语义漂移条件（C4）：`sim_zz' ≤ τ_drift`（如 0.6）。确认确实漂移。
- 报告每条件 `sim_zz'` 的分布（均值/IQR），证明操纵有效。

---

## 3. Length difference

| 指标 | 定义 |
|------|------|
| `len_z`, `len_z'` | token 数（也记字符数） |
| `delta_len` | `len(z') − len(z)` |
| `len_ratio` | `len(z') / len(z)` |
| `len_abs_ratio` | `|len(z')−len(z)| / len(z)` |

format-preserving / drift / random 条件应控制 `len_ratio ≈ 1`，避免长度混淆。

---

## 4. Token / statistical shift（表面扰动量）

量化 `z → z'` 的表面/统计变化；用作机制回归自变量，检测隐写/统计通道。

| 指标 | 定义 | 探测 |
|------|------|------|
| `jaccard_tokens` | `|Tz ∩ Tz'| / |Tz ∪ Tz'|` | 词面重叠 |
| `ngram_overlap` | 1/2/3-gram 重叠率 | 局部表面结构 |
| `edit_distance_norm` | 归一化 Levenshtein（字符或 token） | 表面距离 |
| `kl_token_dist` / `js_divergence` | token 频率分布 KL / JS | 统计分布漂移 |
| `ppl_shift` | `ppl(z') − ppl(z)`（参考 LM） | 自然度/编码异常 |
| `char_ngram_cosine` | 字符 n-gram TF 向量余弦 | 拼写/形态层 |
| `entropy_shift` | token 熵差 | 信息密度变化 |

**复合表面扰动**：可对上述做 PCA/标准化加权得 `surface_shift` 单一维度，用于回归。

---

## 5. 指标 → 分析的映射

| 分析目标 | 用到的指标 |
|----------|-----------|
| 主效应（条件是否降 FVE） | `fve`, `Δfve` |
| 操纵是否成功 | `sim_zz'`（manipulation check） |
| 长度混淆控制 | `len_ratio`, `delta_len` |
| 语义 vs 表面通道判别 | `Δfve ~ surface_shift + sim_zz'` 回归 |
| 跨条件归一 | `fve_norm`（floor = C9） |

---

## 6. 输出表 schema（每行 = 一个 (activation, condition) 配对观测）

```
input_id, condition, domain, source_text,
len_z, len_z_prime, delta_len, len_ratio,
sim_zz_prime, surface_shift, jaccard_tokens, js_divergence, ppl_shift,
fve, activation_cosine, activation_mse,
delta_fve, fve_norm,
target_model, layer_l, av_weights, ar_weights, seed, run_id
```

> 激活向量 `h_l` / `ĥ_l` 本身落盘到 `data/interim/<run_id>/`（npy/npz），metrics 表只存标量。
落盘为 `results/<run_id>/metrics.parquet`（或 csv），供 stats 模块直接读取。
