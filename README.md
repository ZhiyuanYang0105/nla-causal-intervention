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

**训练(本地 Qwen2.5-0.5B)**:warm-start AR-only FVE 约 +0.14;一次较早的运行里 GRPO 把 chain FVE
推到 ~+0.175(峰值 +0.29)给出量级参考。

> ⚠️ 这些**具体数字来自修复前的运行**(在 AR 截断、激活单位归一化、分析门控等修复**之前**)。
> 当前修正后的管线**尚未重跑**(需 GPU);本地 `data/interim/`、`results/*/ckpt/` 里的旧数据/权重
> 与现行代码**不兼容**(AR 已截断、激活已归一化),须重新 harvest + 重训才能产出现行管线的结果。

**隐写干预:本地 0.5B 规模下结论为 UNDETERMINED,且无法支撑任何方向性判定。** 原因(经独立审计确认):
- **负对照不可靠**:语义漂移 C4 平均 sim≈0.58、约半数没真漂移——0.5B 改写器太弱,漂移≈"第四种改写"。
- **NLA 太弱**:set-level chain FVE 仅 ~+0.06,重建信号接近噪声底;约 40% "保语义"样本也没过 sim 门控。

干预分析本身已按审计修正为**正确管线**(操纵门控过滤 + set-level FVE + 稳健 cosine 配对检验 + `mechanism_regression` 作 H1/H2 判别器 + 门控判定),但**要得到可信的 H1/H2 判定,需要更大模型/数据**(更强且稳定的 FVE + 真能漂移的负对照)。

详见 **[docs/nla_faithful_findings.md](docs/nla_faithful_findings.md)**(含完整忠实度核查与两轮独立审计的修复);更大规模复现见 **[docs/autodl_plan.md](docs/autodl_plan.md)**。

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
