# NLA Causal Intervention

因果中介干预实验,针对 transformer-circuits 2026《Natural Language Autoencoders》。

NLA 在**激活空间**自编码:`h_l ──AV──▶ 解释文本 z ──AR──▶ ĥ_l`。
干预中间文本 `z`(语义保持的改写),测试 AR 重建激活的 **FVE** 是否显著下降,
判断信息传递依赖**语义通道**还是**表面形式 / 隐写通道**。

- **AV** = Activation Verbalizer:`h_l → z`
- **AR** = Activation Reconstructor:`z → ĥ_l`
- **本地预算配置(MacBook M5 / 16GB)**:M = `Llama-3.2-1B`(冻结),AV = 训练-free 摘要 proxy,
  **AR = 冻结特征 + 闭式 ridge(零反向传播)**,激活只缓存 selected-layer pooled fp16,N=200 pilot,seq≤128。
  内存峰值 ≤8–9GB,磁盘 ~10–15GB,pilot ~1–3 小时。详见 **[local_budget_plan.md](docs/local_budget_plan.md)**。
  - (论文原用非公开 Claude 模型;本项目走轻量本地复现路线)

## 文档

- [论文事实依据](docs/paper_findings.md) — AV/AR/M、FVE、激活采集(本项目的事实约束)
- [研究计划](docs/research_plan.md) — 问题、假设、变量、条件、流程
- [评价指标](docs/metrics.md) — FVE(激活) / semantic(文本) / length / token-shift
- [统计方案](docs/statistical_analysis.md) — 配对设计、显著性检验、机制回归

## 状态

- [x] M0 框架搭建（目录 + 计划 + scaffold）
- [x] M1 数据/指标/pipeline：FVE + token-shift 指标、激活采集器(Llama+FineWeb)、runner、端到端 dry-run
- [x] **干预条件 + 统计链路**：全部 9 个条件可用(model-free + LLM 改写)、sim_zz' 语义相似度、**完整 stats 模块**(Friedman/Wilcoxon/Holm/混合效应/机制回归/功效)
- [x] **Local-budget 重设计**:轻量 AR(`RidgeReconstructor` + bow/frozen-M readout,闭式)、激活 pooled/fp16 缓存、N=200 pilot、内存/磁盘/时间预算(见 local_budget_plan)
- [x] **6 阶段本地编排脚本** `run_local_pilot.py`(断点续跑、单模型驻留、`--smoke` 已验证)
- [ ] M2-local 在 MacBook 上跑真实 1B/3B(`make local-pilot`)
- [ ] M3 manipulation check 校准 + 多 AR 档对比(bow/readout/semantic 定位通道)
- [ ] M4 扩样 + 报告

**已实现且测试覆盖(47 tests)**：reconstruction(FVE/cosine/MSE)、length/token-shift、semantic sim、9 个干预条件(`identity`/`token_shuffle`/`stopword_strip`/`random_text` + 改写器驱动的 `paraphrase`/`strong`/`format_preserving`/`semantic_drift`/`synonym`/`back_translation`)、runner、激活采集器、统计分析全链路。`make dryrun` + `analyze_results.py` 跑通 metrics→stats→report。

**仍待真实模型**：训练好的 AV/AR 权重、真实改写器(LLMRewriter 注入 `complete` 回调即可)、真实 sentence embedder(SentenceTransformerEmbedder)。fake 版已能驱动全流程。

## 结构

```
docs/         研究计划与设计文档
configs/      实验/条件/模型配置 (yaml)
data/         raw → interim → processed
src/nla_intervention/
  conditions/ 干预变换 (paraphrase / drift / ...) — 作用在解释文本 z 上
  pipeline/   AV (Verbalizer) / AR (Reconstructor) 接口 + runner
  metrics/    reconstruction(FVE,激活) / semantic / length / token-shift
  stats/      配对检验 + 机制回归
  data/       激活采集器 (跑 M → 收 h_l)
  utils/      io / seed / logging
experiments/  各次实验的配置与产物
results/      metrics 表、统计结果、图
notebooks/    探索与出图
tests/        单元测试
```

## 快速开始

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"     # 轻量(numpy/scipy/pandas/statsmodels) — 跑指标/stats/测试
make test                   # 53 单元测试(指标 + AR ridge + stats + 端到端)
make local-pilot-smoke      # 6 阶段编排用 fakes 跑通(无模型),验证 staging + 断点续跑

# 本地真实 pilot(MacBook M5 / 16GB):
pip install -e ".[models,mlx,nlp]"
make local-pilot            # harvest→AV→改写→ridge AR→metrics→stats,~1–3h,见 local_budget_plan
```
