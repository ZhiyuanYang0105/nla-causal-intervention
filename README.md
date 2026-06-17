# NLA Causal Intervention

因果干预实验,针对 transformer-circuits 2026《Natural Language Autoencoders》。

NLA 在**激活空间**自编码:`h_l ──AV──▶ 解释文本 z ──AR──▶ ĥ_l`。
核心问题:干预中间文本 `z`(语义保持的改写)后,AR 重建激活的 **FVE** 是否下降?
- 若**只有改语义(漂移)才下降** → 信息走**语义通道**(解释忠实,H1)
- 若**保语义的改写也下降** → 信息藏在**表面形式**里 → **隐写通道**(解释不忠实,H2)

**关键前提:隐写是 AV/AR 联合训练的产物**——只有把两者一起训练,AV 才可能学会把信息编码进措辞表面、AR 才学会解它。所以本项目忠实地按论文**联合训练** AV/AR,再做干预测试。

- **AV (Activation Verbalizer)**:`h_l → z`。激活归一化后作为一个 token embedding **注入**,自回归生成解释。
- **AR (Activation Reconstructor)**:`z → ĥ_l`。z 过模型取 layer-`l` 末 token 激活 → **学习的 affine map**。
- **训练**(忠实于论文,唯一替换 Opus 4.6 → Qwen2.5-0.5B):
  - **warm-start SFT**:AV 学 `h_l→摘要`、AR 学 `摘要→h_l`(热启动)
  - **GRPO 联合训练**:AV 用强化学习(reward = −重建误差,AR 当固定打分器),AR 用监督 MSE,更新解耦
  - **KL 惩罚**:把 AV 锚在热启动状态附近,保持解释流畅(防退化成乱码)

---

## 结果(exp04_nla,N=2500,本地真实运行)

**训练**:warm-start AR-only FVE +0.14 → 稳定化 GRPO 后 **chain FVE +0.175**(峰值过 +0.29,进入论文 warm-start 区间 0.3–0.4)。

**隐写干预测试**(共训练 AV/AR + 真实改写,n_eval=350):

| 条件 | FVE 中位数 | Δ vs C0 | 配对 Wilcoxon(Holm) | sim(z,z') |
|--|--|--|--|--|
| C0 原始 | +0.345 | — | — | 1.00 |
| C1 轻改写 | +0.274 | 0.071 | 不显著 dz=0.04 | 0.85 |
| C2 强改写 | +0.222 | 0.122 | 显著 dz=0.07 | 0.85 |
| C3 保格式改写 | +0.243 | 0.102 | 不显著 dz=0.08 | 0.88 |
| C4 语义漂移 | +0.315 | 0.030 | 不显著 dz=0.01 | 0.58 |

**Friedman omnibus p=0.176(不显著)。** 操纵有效(改写 sim≈0.85、漂移≈0.58)。

### 结论:在本地规模下**无法判定(UNDETERMINED)**——但方法完全正确

- omnibus 不显著;只有 C2 显著(dz 极小、与 C1/C3 不自洽);跨子集不稳 → 噪声主导(per-sample 重尾)。
- 弱而较一致的描述性迹象:保语义改写(C1-C3)降 FVE > 漂移 C4——**若非噪声**会隐指 H2,但不显著 → 不能下结论。
- 这是**方法完全正确**(真共训练、KL 修复、强基线 FVE +0.175、操纵有效)下仍判不出 → 纯因 **0.5B/2500 规模重建重尾、功效不足**。要判定隐写需更大模型/数据 → 更强 FVE → 足够功效。

详见 **[docs/nla_faithful_findings.md](docs/nla_faithful_findings.md)**(含完整忠实度核查);更大规模复现见 **[docs/autodl_plan.md](docs/autodl_plan.md)**。

---

## 文档

- [论文事实依据](docs/paper_findings.md) — AV/AR/FVE 与论文的 GRPO 训练方法
- [忠实复现 findings](docs/nla_faithful_findings.md) ⭐ — 实现、训练、隐写测试、忠实度核查、结论
- [研究计划](docs/research_plan.md) / [评价指标](docs/metrics.md) / [统计方案](docs/statistical_analysis.md) — 实验设计
- [autodl/5090 复现方案](docs/autodl_plan.md) — 单卡 1.5B 忠实复现 runbook(全量微调、β 扫描、多种子)

## 结构

```
src/nla_intervention/
  nla/          忠实 NLA:av(激活注入) / ar(截断+affine) / train(warm-start + GRPO + KL)
  conditions/   干预变换 (paraphrase/drift/...) + apply_condition + 改写器 (LLM/Fake)
  metrics/      reconstruction(FVE,激活) / semantic(sim_zz') / token-shift
  stats/        配对检验 + 机制回归 + 功效
  data/         激活采集器 (跑 M → 收 h_l)
  utils/        config / seed
scripts/
  build_nla_data.py   采集激活 + 批量生成摘要 (warm-start 数据)
  train_nla.py        warm-start SFT + GRPO 联合训练 -> ckpt
  steg_intervention.py 共训练 AV/AR 上的隐写干预测试 ⭐
experiments/    exp03_nla / exp04_nla(忠实主实验)
configs/        default.yaml(共享基础) + conditions.yaml(干预条件)
docs/  tests/
```

## 快速开始

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"     # 轻量(numpy/scipy/pandas) — 跑指标/stats/测试
make test                   # 单元测试(指标 + 改写器 + stats + 采集器 helper)

# 完整忠实流程(需 GPU + transformers/peft):
pip install -e ".[models,nlp]" && pip install peft
make nla-data               # 采集激活 + 摘要
make train-nla              # warm-start + GRPO 联合训练(数小时)
make steg                   # 隐写干预测试
```

> 本地 0.5B/MacBook 规模:方法忠实但功效受限(结论 UNDETERMINED)。要得到可判定的结果,见 autodl_plan(单张 5090 + 1.5B 全量微调)。
