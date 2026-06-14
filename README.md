# NLA Causal Intervention

因果干预实验,针对 transformer-circuits 2026《Natural Language Autoencoders》。

NLA 在**激活空间**自编码:`h_l ──AV──▶ 解释文本 z ──AR──▶ ĥ_l`。
核心问题:干预中间文本 `z`(语义保持的改写)后,AR 重建激活的 **FVE** 是否下降?
若下降 → 信息走**表面形式/隐写通道**;若只有语义漂移才下降 → 走**语义通道**。

- **AV** = Activation Verbalizer:`h_l → z`(激活作为 embedding 注入,自回归生成)
- **AR** = Activation Reconstructor:`z → ĥ_l`(过模型取 layer-l 激活 + 学习的 affine map)
- **关键**:隐写是 **AV/AR 联合训练**的产物——不联合训练就不可能出现隐写,也就无从检测。

> 本地复现配置(MacBook M5 / 16GB / 纯本地):M = `Qwen2.5-0.5B`(免授权,替代闭源 Opus 4.6),
> 其余尽量忠实论文。详见 **[docs/local_budget_plan.md](docs/local_budget_plan.md)**。

---

## ⚠️ 两个阶段,务必区分

本仓库经历了两个阶段,结论性质完全不同:

### 阶段一(exp02_open):文本空间代理 — **结论已作废**
早期把 NLA 近似成:AV=源文本摘要(非激活输入)、AR=独立拟合的 ridge(**未与 AV 联合训练**)。
当时报告"支持 H1 语义通道"。**该结论无效**:AV/AR 没有联合训练,隐写从设置上就不可能存在,
等于没测真正的问题。保留 `docs/results_exp02.md` 仅作过程记录与反面教材。

### 阶段二(exp03/exp04_nla):忠实 NLA — **方法正确**
按论文重建:**激活注入式 AV** + **截断 AR + affine** + warm-start SFT + **GRPO 联合训练**(KL 惩罚)。
唯一替换是 Opus 4.6 → Qwen2.5-0.5B。这才是能检测隐写的正确设置。

---

## 实验结果(exp04_nla,忠实 NLA,N=2500,本地真实运行)

**训练**(修复 KL bug 后):
- 忠实度核查发现一个真实 bug:KL 应朝 **warm-start 后的 AV**(`AV_φ_init`),原实现错误地朝**底座**。
  修复后 GRPO `kl/tok` 从 ≈0 起步(av=ref)、随 RL 增长(旧 bug 版一开始就 ~1.0)。
- warm-start AR-only FVE +0.14;稳定化 GRPO(经验回放+per-token KL)后 **chain FVE +0.06 → +0.175**
  (修正 KL 后近单调爬升,峰值过 +0.29,进入论文 warm-start 区间 0.3–0.4)。

**隐写干预测试**(修正 NLA,强基线,真实 Qwen-Instruct 改写,n_eval=350):

| 条件 | FVE 中位数 | Δ vs C0 | 配对 Wilcoxon(Holm) | sim(z,z') |
|--|--|--|--|--|
| C0 原始 | +0.345 | — | — | 1.00 |
| C1 轻改写 | +0.274 | 0.071 | 不显著 dz=0.04 | 0.85 |
| C2 强改写 | +0.222 | 0.122 | 显著 dz=0.07 | 0.85 |
| C3 保格式改写 | +0.243 | 0.102 | 不显著 dz=0.08 | 0.88 |
| C4 语义漂移 | +0.315 | 0.030 | 不显著 dz=0.01 | 0.58 |

**Friedman omnibus p=0.176(不显著)。** 操纵有效(改写 sim≈0.85、漂移≈0.58)。

### 结论:隐写问题在本地规模下**无法判定(UNDETERMINED)**——但方法完全正确

- omnibus 不显著;只有 C2 显著(dz 极小、与 C1/C3 不自洽);**跨子集不稳**
  (n=120 时漂移 C4 降 0.19、n=350 时只降 0.03,负对照矛盾)→ 噪声主导。per-sample 均值仍为负(重尾)。
- **弱而较一致的描述性迹象**:两次都见"保语义改写(C1-C3)降 FVE > 漂移 C4",强改写 C2 两次都显著——
  **若非噪声**会隐指 H2(表面/隐写通道),但 omnibus 不显著 → **不能下结论**。
- 这次是**方法完全正确**(真共训练、KL 修复、强基线 FVE +0.175、操纵有效)下仍判不出 → 纯因 0.5B/2500
  规模重建重尾、功效不足。要判定隐写需论文规模(更大模型/数据 → 更强更稳 FVE → 足够功效)。

完整方法+结果+诊断见 **[docs/nla_faithful_findings.md](docs/nla_faithful_findings.md)**。

---

## 文档

- [论文事实依据](docs/paper_findings.md) — AV/AR/M、FVE、激活采集、**训练方法(GRPO+KL)**
- [忠实复现 findings](docs/nla_faithful_findings.md) ⭐ — 忠实 NLA 的实现、训练、隐写测试、结论
- [本地预算方案](docs/local_budget_plan.md) — M5/16GB 下的轻量化设计
- [研究计划](docs/research_plan.md) / [评价指标](docs/metrics.md) / [统计方案](docs/statistical_analysis.md)
- [results_exp02](docs/results_exp02.md) / [pilot_findings](docs/pilot_findings.md) — 阶段一过程记录(结论已作废)

## 结构

```
src/nla_intervention/
  nla/          忠实 NLA:av(激活注入) / ar(截断+affine) / train(warmstart+GRPO)
  conditions/   干预变换 (paraphrase/drift/...) + 改写器 (LLM/Fake)
  metrics/      reconstruction(FVE,激活) / semantic(sim_zz') / token-shift
  stats/        配对检验 + 机制回归 + 功效
  pipeline/     AV/AR 接口、runner、ridge readout(阶段一)
  data/         激活采集器 (跑 M → 收 h_l)
scripts/
  build_nla_data.py   采集激活 + 批量生成摘要(warm-start 数据)
  train_nla.py        warm-start SFT + GRPO 联合训练
  steg_intervention.py 共训练 AV/AR 上的隐写干预测试 ⭐
  (阶段一: run_local_pilot / final_analysis / compare_ar / dry_run)
experiments/    exp02_open(阶段一) / exp03_nla / exp04_nla(忠实)
docs/  results/  data/  tests/
```

## 快速开始

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[models,nlp]" && pip install peft     # 含 torch/transformers/peft

# 忠实 NLA(本地,MacBook):
python scripts/build_nla_data.py  --reuse exp02_open --out exp04_nla --n 2500   # 数据(批量摘要)
python scripts/train_nla.py       --config experiments/exp04_nla/config.yaml --grpo --grpo-steps 200
python scripts/steg_intervention.py --config experiments/exp04_nla/config.yaml --n-eval 120

# 纯分析/测试(无需模型):
pip install -e ".[dev]" && make test
```

> 训练在 0.5B/2500 规模约数小时;GRPO 收敛有残余震荡;结论限定为"在该本地规模代理上"。
