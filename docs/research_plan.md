# 研究计划：NLA 中间文本的因果中介干预实验

> AV → intermediate text (z) → AR
> 测试语义保持改写后 AR reconstruction 是否显著下降，判断信息走语义通道还是表面/隐写通道。

---

## 1. 研究问题与背景

本项目针对 transformer-circuits 2026《Natural Language Autoencoders》（见 [paper_findings.md](paper_findings.md)）。
NLA 自编码的是目标模型 M 的**残差流激活向量** `h_l`，中间文本只是 bottleneck：

```
激活 h_l ──AV(Activation Verbalizer)──▶ 自然语言解释 z ──AR(Activation Reconstructor)──▶ 重建激活 ĥ_l
```

- **AV**：与 M 同架构的 LLM（M 的微调副本），把单个激活 `h_l`（中到后层、片段最后 token）verbalize 成解释 `z`。
- **AR**：M 截断到前 `l` 层 + 学习的 affine map，把 `z` 重建成激活 `ĥ_l`。
- **重建度量是 FVE（激活空间）**，不是文本相似度。论文达 0.6–0.8 FVE。

`z` 是人类可读自然语言，充当信息瓶颈。

**核心疑问**：AR 重建激活 `h_l` 时，依赖的是 `z` 的 **语义内容（semantic channel）**，还是 `z` 的 **表面形式 / 词面统计特征 / 隐写编码（surface / steganographic channel）**？AV 完全可能把激活信息隐写进措辞/格式而非语义——这直接关系到 NLA 解释的**可解释忠实性**。

- 如果是语义通道：只要改写保持语义，AR 的 FVE 应当**不显著下降**；只有语义漂移才会破坏重建。
- 如果是表面/隐写通道：即便语义完全保持，只要打乱词面/格式/统计特征，FVE 就会**显著下降**，且下降随表面扰动单调增大。

这是「自发隐写 / non-robust encoding」的一个干净因果检验：把解释 `z` 当作中介做 `do()` 干预。论文已有的 rabbit→mouse edit 是**改语义**的干预；本项目补上**语义保持 paraphrase** 这一空白。

### 1.1 主研究问题 (RQ)

- **RQ1**：语义保持的改写（paraphrase / strong paraphrase / format-preserving）是否导致 AR 的 **FVE** 显著下降？
- **RQ2**：在控制语义相似度不变的前提下，FVE 的下降幅度是否由表面/统计扰动量（token overlap、length、distributional shift）解释？
- **RQ3**：语义漂移对照（semantic-drift control）是否产生远大于语义保持改写的下降？（语义通道存在的必要证据）

---

## 2. 假设 (Hypotheses)

按竞争假设设计，便于 falsification。

| ID | 假设 | 预测 |
|----|------|------|
| **H0** | 零假设 | 各条件 FVE 无显著差异 |
| **H1 (语义通道)** | 信息主要走语义通道 | 语义保持条件 ≈ original（FVE 无显著降）；只有 semantic-drift 显著降 |
| **H2 (表面/隐写通道)** | 信息主要走表面/统计通道 | 即使语义保持，FVE 随表面扰动单调显著下降；semantic-drift 与 strong-paraphrase 降幅相近 |
| **H3 (混合)** | 双通道并存 | 语义保持改写有中等但显著下降；semantic-drift 下降最大；下降量同时由语义差与表面差解释 |

**判别逻辑（关键）**：

- 若 `format-preserving paraphrase`（保格式、保统计、换词面但**保语义**）重建良好，而 `strong paraphrase`（换词面+换格式、保语义）重建崩溃 → 偏向**表面/格式通道**。
- 若 `semantic-drift`（保格式、保长度，但**改语义**）重建崩溃，而所有语义保持条件重建良好 → 偏向**语义通道**。
- 在固定 `sim(z, z')` 的回归中，若表面扰动指标的系数显著 → 存在表面通道（H2/H3）。

---

## 3. 变量定义

### 3.1 Treatment（自变量 / 操纵变量）
作用在中介变量（解释文本 `z`）上的**变换条件** `T`（分类变量），见 §4。
对应因果上的 `do(z := T(z))`：固定 AV/AR/激活不变，只替换喂给 AR 的解释文本。

### 3.2 Mediator（中介变量）
解释文本 `z = AV(h_l)`（original）及其干预后版本 `z' = T(z)`。
中介本身的属性被测量为 covariates：`len(z')`、`sim(z, z')`、token overlap 等。

### 3.3 Outcome（因变量）
AR 在 `z'` 上重建激活的 **FVE**：`FVE(h_l, AR(z'))`（激活空间，见 metrics.md）。
主结果用相对基线的下降量 `Δfve = FVE(C0) − FVE(Ck)`（paired，同一个激活 `h_l`）。

### 3.4 Covariates / Controls
- 输入侧：激活来源 domain、`h_l` 范数 / 在 PCA 上的位置（激活无「长度」概念）。
- 中介侧：`len(z)`、`sim(z, z')`（语义保持的 manipulation check）、表面扰动量。
- 模型侧：目标模型 M、层 `l`、AV 采样温度、AR greedy、random seed（固定以可复现）。

### 3.5 因果图 (DAG)

```
      h_l  （目标模型 M 的残差流激活，ground truth）
        │ AV
        ▼
        z ──────T(do)──────▶ z'
                              │ AR
                              ▼
                            ĥ_l  ──FVE(h_l, ĥ_l)──▶ reconstruction
covariates: domain, len(z), sim(z,z'), token_shift  ──┘
```

干预切断 `z→z'` 自然路径，强制替换为受控变换，从而识别「`z` 的哪些属性」驱动激活重建（FVE）。

---

## 4. 实验条件 (Conditions)

核心条件（≥5，必做），按 (语义保持度 × 表面/格式保持度) 二维正交设计：

| Code | 条件 | 语义保持 | 表面/词面 | 格式/长度 | 作用 |
|------|------|:---:|:---:|:---:|------|
| **C0** | **Original** (no-op) | 高 | 高 | 高 | 上界 / baseline |
| **C1** | **Paraphrase** (light) | 高 | 中 | 中 | 轻度词面扰动 |
| **C2** | **Strong paraphrase** | 高 | 低 | 低 | 重度词面+句法扰动，仍保语义 |
| **C3** | **Format-preserving paraphrase** | 高 | 中 | **高（强制保格式/长度/结构）** | 隔离「格式/统计通道」 |
| **C4** | **Semantic-drift control** | **低（改语义）** | 中 | 高 | 负对照：语义通道存在的必要条件 |

> C3 vs C2 对比 = 「格式/统计特征是否承载信息」；C4 = 「语义是否真的被用到」。

### 扩展条件（可选，强化机制判别）

| Code | 条件 | 探测的通道 |
|------|------|-----------|
| C5 | **Synonym substitution**（仅同义词替换，保句法） | 词汇 vs 语义 |
| C6 | **Back-translation**（往返翻译） | 自然 paraphrase 分布 |
| C7 | **Token shuffle**（打乱词序，保 bag-of-words） | 句法 vs 词袋/统计 |
| C8 | **Stopword/punctuation strip**（去停用词/标点） | 表面冗余位 |
| C9 | **Random text**（无关文本，等长） | 下界 / floor |

> C9 给出 reconstruction 的随机基线（floor）。所有效应量用 (C0 − floor) 归一化更可比。

每个条件对**同一批输入** `x` 应用，构成 paired / repeated-measures 设计。

---

## 5. 评价指标（摘要，详见 metrics.md）

1. **Reconstruction**（主结果，激活空间）：**FVE = 1 − ‖h_l − ĥ_l‖² / ‖h_l − h̄_l‖²**（论文主指标）；辅以激活 cosine、激活 MSE。**不用** BLEU/ROUGE（那是文本指标）。
2. **Semantic similarity**（manipulation check，文本上）：`sim(z, z')` 用 sentence-embedding 余弦。验证语义保持条件确实保住语义、drift 条件确实漂移。
3. **Length difference**：`Δlen = len(z') − len(z)`、`len_ratio`、token 数差（解释文本 `z`）。
4. **Token / statistical shift**：Jaccard token overlap、n-gram overlap、normalized edit distance、token 分布 KL/JS divergence、perplexity shift。量化「表面扰动」作为机制回归自变量。

---

## 6. 统计分析方案（摘要，详见 statistical_analysis.md）

- **配对设计**：同一激活 `h_l` 跑过所有条件 → 配对观测，消除 item 间方差。
- **总体差异**：repeated-measures ANOVA（若正态）或 **Friedman test**（非参，默认）。
- **两两对比**：各条件 vs C0 的 **Wilcoxon signed-rank**（配对，对 FVE），Holm / Bonferroni 校正。
- **效应量**：paired Cohen's dz；非参用 rank-biserial / Cliff's δ。
- **混合效应模型**：`fve ~ condition + len_ratio + (1|input_id)`，分离条件主效应与协变量。
- **机制回归（核心）**：`Δfve ~ surface_shift + sim(z,z')`；若在 `sim` 控制下 `surface_shift` 系数显著 → AV 把激活信息隐写进表面形式（支持 H2/H3）。
- **Bootstrap CI** + **预先功效分析**定样本量；建议 **预注册** 假设与分析。

---

## 7. 实验流程 (Pipeline)

```
1. 采集激活集 H：跑 M 在 pretraining-like 文本上，取层 l、最后 token 的 h_l   [data/harvest]
                  （多 domain 分层；held-out split 与 AV/AR 训练数据不重叠）
2. AV(h_l) → z                       [nla.av: ActivationVerbalizer]
3. 对每个 z 应用 T_k → z'_k           [conditions/*]   ← manipulation check: sim(z,z')
4. AR(z'_k) → ĥ_l,k                   [nla.ar: ActivationReconstructor]
5. 计算 metrics: FVE(h_l, ĥ_l,k) + sim/length/token-shift   [metrics/*]
6. 配对统计分析                        [stats/*]
7. 出图 + 报告                         [results/, notebooks/]
```

固定：M / 层 l、AV/AR 权重版本、AV 采样温度、AR greedy、seed。激活与中间产物落盘保证可复现。

---

## 8. 风险与对照 (Threats to validity)

- **改写质量**：paraphrase 可能无意中漏掉语义 → 用 `sim(z,z')` manipulation check 过滤不合格样本。
- **drift 强度**：semantic-drift 必须真的改语义但保格式 → 校验 `sim(z,z')` 显著低于 paraphrase 组。
- **AV/AR 隐写泄漏**：若 AV 把激活信息编码进格式（隐写），正是我们要检测的——C3/C7 专门探测。
- **长度混淆**：解释长度变化可能影响 AR → 控制 `len_ratio`，并设等长条件（C3/C4/C9）。
- **分布漂移混淆（关键）**：激活必须取自 **pretraining-like 文本**（AV/AR 训练分布内），否则 FVE 下降会混入 OOD 效应而非 paraphrase 效应。
- **AV 采样随机性**：AV 对同一 `h_l` 采样一组 `z`；固定 z（或对多采样取均值）以保证 paired 对比干净。
- **多重比较**：所有对比做 Holm 校正。
- **可复现**：固定 seed、温度、M/层/权重版本；记录到 `results/run_metadata.json`。

---

## 9. 交付物

- `docs/`：本计划 + 设计 + 指标 + 统计方案 + 忠实复现 findings。
- `src/nla_intervention/`：nla（忠实 AV/AR + 训练）/ conditions / metrics / stats / data 模块。
- `experiments/exp04_nla/`：忠实 NLA 主实验配置。
- `results/`：metrics 表、统计结果、checkpoint。
- 报告：结论回答 RQ1–RQ3，判定 H1/H2/H3（见 nla_faithful_findings.md）。

---

## 10. 里程碑

| 阶段 | 内容 | 产物 |
|------|------|------|
| M0 | 框架搭建（本次） | 目录 + 计划 + scaffold |
| M1 | 接入 AV/AR + 数据 | nla 流程跑通 1 条 |
| M2 | 实现 5 个核心条件 + manipulation check | conditions 可用 |
| M3 | 指标 + pilot（小样本） | pilot metrics 表 |
| M4 | 全量 run + 统计分析 | 显著性结果 |
| M5 | 报告 + 结论 | 回答 RQ，判定假设 |
