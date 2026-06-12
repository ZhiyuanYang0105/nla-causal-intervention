# Local-Budget 方案（MacBook M5 / 16GB / 本地执行）

把 NLA 因果干预实验压缩成**单机可跑的最小可行实验 (MVP)**。
硬件:MacBook M5,16GB unified memory,1TB,纯本地。

核心原则:**冻结所有 LLM,只训练小 head(或不训练)。任何 8B 仅推理,绝不训练。**

---

## 1. 模型选择

| 角色 | 模型 | 用途 | 是否训练 | 内存(bf16) |
|--|--|--|--|--|
| **M(目标模型)** | **`meta-llama/Llama-3.2-1B`** (base) | 产生残差流激活 `h_l` | ❌ 冻结,仅前向 | ~2.5 GB |
| M 备选 | `Llama-3.2-3B` (base) | 更强激活(可选) | ❌ | ~6.4 GB |
| **AV(verbalizer)** | `Llama-3.2-3B-Instruct` | 把激活/文本 → 解释 `z` | ❌ **训练-free(prompt)** | ~6.4 GB |
| **改写器(conditions)** | 同上(或 1B-Instruct 更快) | paraphrase/drift | ❌ prompt | 复用 AV |
| **AR(reconstructor)** | 冻结 M + **ridge head** | `z` → `ĥ_l` | ✅ **只训 ridge/线性**(闭式) | 复用 M |
| 语义 embedder | `all-MiniLM-L6-v2` | `sim_zz'` manipulation check | ❌ | ~90 MB |
| 8B(可选) | `Llama-3.1-8B-Instruct` 4-bit (MLX) | 仅作高质量改写器**推理** | ❌ 仅推理 | ~4.5 GB |

> **运行时建议用 MLX**(`mlx-lm`)做生成,比 transformers-MPS 快 2–3×;激活抽取用 transformers-MPS(需要 hidden states)。

---

## 2. AV 轻量重设计(训练-free)

论文 AV 需整模型微调 → 本地不可行。两档替代,**均无需训练**:

### Tier-0(pilot 默认):summary-proxy AV
- `z = Instruct-LLM.summarize(source_snippet)`。论文 AV 本就用 text-summarization proxy 初始化,这是忠实的简化。
- **零训练**,只是 prompt 一个 3B-Instruct。
- **已知近似**:它描述的是「源文本」而非「激活本身」。对 pilot 可接受(激活高度编码源文本内容);作为主要 validity caveat 记录。`z` 不依赖条件,**预计算一次**。

### Tier-1(升级,更忠实):activation-conditioned AV
- Patchscopes / SelfIE 风格:把 `h_l` 经一个小线性投影注入 Instruct 模型的某个 token 槽,解码出描述。
- 训练量:**只训投影**(几十万参数)或完全 training-free(用 logit-lens 式直读)。
- 留给全量实验;pilot 不需要。

---

## 3. AR 轻量重设计(只训一个 head)

论文 AR = M 截断到前 l 层 + affine map。本地版:**冻结特征提取 + 闭式 ridge 回归**。

⚠️ **关键设计约束**:AR 必须对**表面 token 敏感**,否则检测不了隐写通道。
**不能**用语义 embedder(mpnet/MiniLM)做 AR——它 paraphrase-invariant,会让 FVE 对改写天然鲁棒 → 结论必然 H1、循环论证。语义 embedder 只用于 `sim_zz'` manipulation check。

三档 AR,按表面敏感度排列(可多档对比以定位通道):

| AR | 特征提取 `z → 特征` | head | 表面敏感 | 成本 | 作用 |
|--|--|--|--|--|--|
| **AR-bow**(最便宜) | hashing n-gram TF 向量 | ridge(闭式) | ✅ 高 | 即时,纯 CPU | 「纯表面」参考 / 极简 pilot |
| **AR-readout**(推荐,忠实) | 冻结 M 前向 `z`,取层 l' 最后 token hidden | ridge/线性 | ✅ 中(LLM 读 token) | 每 z 一次前向(推理) | **主 AR** |
| AR-semantic(参考上界) | 语义 embedder | ridge | ❌ 无 | 即时 | 「纯语义」参考(预期对改写鲁棒) |

**实验增强**:同时跑 AR-bow / AR-readout / AR-semantic,看 ΔFVE 谱:
- 若 AR-readout 的 ΔFVE 接近 AR-bow(高) → 信息走表面/隐写通道 (H2)。
- 若接近 AR-semantic(≈0) → 走语义通道 (H1)。

**训练 = 闭式 ridge**:`W = (XᵀX + αI)⁻¹ Xᵀ H`,X∈ℝ^{N×d_feat},H∈ℝ^{N×d_act}。N≤500、d≤3072 → 毫秒级,无需 GPU、无反向传播。

---

## 4. 激活缓存策略(只存 selected-layer + pooled)

| 维度 | 规则 |
|--|--|
| 层 | **只 hook 1 层**(M=1B 取 layer 8–10;中后层),绝不 `output_hidden_states` 全层 |
| 位置 | **pooled**:`last`(论文)或 `mean`(序列均值);只存 **1 个向量/样本** |
| 精度 | **float16** 落盘 |
| seq len | **128**(或 64),随机截断 |

**为什么关键**:全层全序列 = 33 层 × 128 × 3072 × 2B ≈ **26 MB/样本** → 500 样本 13 GB。
selected-layer + pooled = **1 向量/样本** = 3072 × 2B ≈ **6 KB/样本** → 500 样本 **3 MB**。差 ~4000×。

---

## 5. Pilot 规模与 seq len

- **N = 200**(可 100–500)。配对设计 5 条件 → 1000 obs,足够估效应量 + 功效分析。
- seq len = **128**(短文本 64)。
- 条件:核心 5(C0–C4)。扩展条件留后。

---

## 6. 预算(200 样本 pilot,M=1B,改写器=3B-Instruct)

### 内存(峰值,分阶段加载,**同时只驻留 1 个 LLM**)
| 阶段 | 驻留 | 峰值 |
|--|--|--|
| 采集激活 | M(1B) | ~3 GB |
| AV 摘要 + 改写 | 3B-Instruct | ~7 GB |
| AR-readout 特征 | M(1B) | ~3 GB |
| ridge/sim/stats | MiniLM + numpy | ~1 GB |
| **峰值** | 单模型 + 开销 | **≤ 8–9 GB**(留 7 GB 余量) |

### 磁盘
| 项 | 大小 |
|--|--|
| 模型权重:1B(2.5)+3B(6.4)+MiniLM(0.09)+mpnet(0.42) | ~9.5 GB |
| 可选 8B-4bit (MLX) | +4.5 GB |
| 激活 + z/z' + metrics(200×5) | < 0.2 GB |
| **合计** | **~10–15 GB**(1TB 绰绰有余) |

### 运行时(范围:transformers-MPS 慢端 ↔ MLX 快端)
| 阶段 | 估时 |
|--|--|
| 采集激活(200 × seq128 前向,1B) | 2–6 min |
| AV 摘要(200 × ~80 tok,3B) | 10–30 min |
| 改写(4 LLM 条件 × 200 ≈ 800 代 × ~80 tok) | 40–120 min ← **瓶颈** |
| AR-readout(1000 前向 1B)+ ridge | 5–15 min |
| sim_zz'(1000 对 MiniLM)+ stats | < 2 min |
| **合计** | **~1–3 小时** |

**降本旋钮**:改写器换 1B-Instruct(快 3–4×)/ N 降到 100 / 只跑 AR-bow(省去 readout 前向)/ 用 MLX。最快可压到 ~30–45 min。

---

## 7. 本地执行流程(分阶段落盘,绝不并驻多模型)

```
stage 1  harvest:    M(1B) 前向 FineWeb 200 片段 → h_l(layer l, pooled, fp16)  → data/interim/acts.npz
stage 2  verbalize:  3B-Instruct 摘要 source → z                               → data/interim/z.jsonl
stage 3  intervene:  3B-Instruct 改写 z → z'_k (C0–C4)                          → data/interim/zprime.jsonl
stage 4  reconstruct: 冻结 M readout(或 bow)特征 → ridge.fit → ĥ_l            → 内存
stage 5  metrics:    FVE / sim_zz'(MiniLM)/ token-shift                        → results/<run>/metrics.parquet
stage 6  stats:      manipulation_check → Friedman → Wilcoxon → 机制回归        → results/<run>/stats_report.json
```

每阶段结束**卸载模型**再进下一阶段,保证峰值只压一个 LLM。

---

## 8. 与全量方案的差异(诚实记录)

| | 论文 / 全量 | 本地 MVP |
|--|--|--|
| M | Claude / Llama-8B | **Llama-3.2-1B** |
| AV | 微调整模型 | **训练-free 摘要 proxy** |
| AR | 截断 LLM + affine | **冻结 readout + 闭式 ridge** |
| 激活 | (内部) | selected-layer + pooled, fp16 |
| N | 大规模 | **200** |
| 训练 | 多 GPU | **零反向传播(ridge 闭式)** |

**主要 validity caveat**:Tier-0 AV 描述源文本而非激活;AR 是 readout 而非生成式。
→ 结论限定为「在该轻量 NLA 代理上」,作为全量实验的**假设筛选 + 方法验证**,不是终局证据。
