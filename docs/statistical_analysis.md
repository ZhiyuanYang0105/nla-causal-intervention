# 统计分析方案

配对设计（同一 input 跑过所有条件）→ 优先配对/重复测量方法，最大化功效、消除 item 方差。

---

## 1. 设计

- **单位**：input `x`（N 个）。
- **重复测量因子**：condition（C0…C4 核心，+ 扩展条件）。
- **配对结构**：每个 `x` 在所有条件下都有观测 → within-subject / repeated-measures。
- **主结果**：`fve`（激活空间 Fraction of Variance Explained），及配对差 `Δfve_k = fve_C0 − fve_Ck`。

---

## 2. 预备：manipulation check（先做，门控后续分析）

1. 语义保持条件 `sim_zz' ≥ τ_keep`，漂移条件 `sim_zz' ≤ τ_drift`。
2. 报告各条件 `sim_zz'` 分布；用 Wilcoxon 确认 drift 组 `sim_zz'` 显著低于 paraphrase 组。
3. 不达标样本：主分析剔除（记录剔除率），敏感性分析保留。

---

## 3. 主分析

### 3.1 总体差异（omnibus）
- 默认非参：**Friedman test**（fve ~ condition，配对）。
- 若残差近正态且方差齐：**repeated-measures ANOVA**（报 Greenhouse–Geisser 校正）。
- H0：各条件 fve 分布相同。拒绝 → 进入两两对比。

### 3.2 两两对比（核心，回答 RQ1/RQ3）
- 每个 Ck vs **C0** 的 **Wilcoxon signed-rank test**（配对，单边：Ck ≤ C0）。
- 关键计划对比：
  - C1, C2, C3 vs C0 —— 语义保持改写是否显著降？（RQ1）
  - C4 vs C0 —— 漂移是否显著降？（语义通道必要证据，RQ3）
  - C2 vs C3 —— 强改写 vs 保格式改写（格式通道）
  - C4 vs C2 —— 漂移 vs 强改写（语义 vs 表面的相对贡献）
- **多重比较校正**：Holm–Bonferroni（默认）或 BH-FDR；报告校正前后 p。

### 3.3 效应量（必报，不只看 p）
- 配对：**Cohen's dz**；非参：**rank-biserial correlation** / **Cliff's δ**。
- 报 95% **bootstrap CI**（配对 bootstrap，重抽 input）。
- 归一化降幅：`(fve_C0 − fve_Ck)/(fve_C0 − fve_floor)`，floor=C9。

---

## 4. 混合效应模型（控制协变量）

```
fve_ij ~ condition + len_ratio + (1 | input_id)
```
- 随机截距 `(1|input_id)` 吸收激活间异质性（不同激活本身可重建性不同）。
- 用 LRT / Wald 检验 condition 主效应；估计各条件相对 C0 的边际 FVE 下降。
- 可选随机斜率 `(condition | domain)` 看激活来源 domain 的调节。

---

## 5. 机制回归（核心判别：语义 vs 表面通道，回答 RQ2）

在**语义保持条件子集**（C1/C2/C3/C5/C6）上：

```
delta_fve ~ surface_shift + sim_zz_prime + len_ratio + (1 | input_id)
```

判读：

| 结果 | 解释 |
|------|------|
| `surface_shift` 系数显著正（控制 sim 后表面越乱降越多） | **存在表面/隐写通道**（支持 H2/H3） |
| `surface_shift` 不显著，唯 `sim_zz'` 驱动 | **纯语义通道**（支持 H1） |
| 两者都显著 | **双通道**（H3） |

补充 **dose-response**：把 surface_shift 分箱，画 FVE 随表面扰动的单调趋势（Page's trend test / Jonckheere–Terpstra）。

---

## 6. 决策规则（把统计结果映射到假设）

| 观测模式 | 判定 |
|----------|------|
| C1–C3 vs C0 均不显著；C4 显著降 | **H1 语义通道** |
| C1–C3 随 surface_shift 单调显著降；C4 与 C2 降幅相近 | **H2 表面/隐写通道** |
| C1–C3 中等显著降 + C4 最大降 + 回归两系数均显著 | **H3 双通道** |
| 全不显著 | **H0**（或功效不足，看 CI/功效分析） |

---

## 7. 样本量与功效

- 先 pilot（N≈20–30 个激活）估 `Δfve` 与 within-subject SD。
- 对最小关心效应（如 dz=0.4）做配对 Wilcoxon 功效分析定 N（目标 power ≥ 0.8, α=0.05 双边、Holm 后）。
- 报告实际达成功效。

---

## 8. 稳健性 / 敏感性

- 多个 reconstruction 指标（fve / activation_cosine / activation_mse）重复主分析（结论应一致）。
- manipulation-check 阈值 ±0.05 的敏感性。
- 去掉 length 不匹配样本后重跑。
- 换 AV/AR 模型或 seed 的复现子集。

---

## 9. 可复现与预注册

- 预注册 RQ、H、主指标、主对比、校正方法、剔除规则、样本量（在 run 前冻结）。
- 固定 seed、解码参数、模型版本，落 `results/<run_id>/run_metadata.json`。
- 所有 p 值、效应量、CI、N、剔除率进结果表与报告。
