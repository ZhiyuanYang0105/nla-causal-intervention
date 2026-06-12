# NLA Causal Intervention

因果中介干预实验,针对 transformer-circuits 2026《Natural Language Autoencoders》。

NLA 在**激活空间**自编码:`h_l ──AV──▶ 解释文本 z ──AR──▶ ĥ_l`。
干预中间文本 `z`(语义保持的改写),测试 AR 重建激活的 **FVE** 是否显著下降,
判断信息传递依赖**语义通道**还是**表面形式 / 隐写通道**。

- **AV** = Activation Verbalizer:`h_l → z`
- **AR** = Activation Reconstructor:`z → ĥ_l`
- **本地预算配置(MacBook M5 / 16GB)**:M = `Qwen2.5-0.5B`(免授权,冻结;Llama 需 HF 授权故未用),
  AV = 训练-free 摘要 proxy,**AR = 冻结特征 + 闭式 ridge(零反向传播)**,激活只缓存 selected-layer pooled fp32,
  seq≤96。内存峰值 ≤8–9GB,磁盘 ~10–15GB。详见 **[local_budget_plan.md](docs/local_budget_plan.md)**。
  - (论文原用非公开 Claude 模型;本项目走轻量本地复现路线)

## 实验结果（exp02_open，本地真实运行）

**配置**:M = `Qwen2.5-0.5B`(冻结,layer 12,mean-pool),AV = 训练-free 摘要 proxy
(`Qwen2.5-0.5B-Instruct`),AR = 冻结 readout + 闭式 ridge,N = 304(FineWeb),5 折交叉验证。
MacBook M5 / 16GB,全本地,无 GPU 训练。

### 结论:支持 **H1(语义通道)** — 中间文本主要通过「意义」而非表面形式传递信息

| 条件 | FVE 中位数 | vs C0 配对检验(Holm) | sim(z,z') |
|--|--|--|--|
| C0 原始 | **+0.18**(基线 >0,AR 确实在重建✓) | — | 1.00 |
| C1 轻改写 | +0.20 | 不显著 (p=0.67) | 0.89 |
| C2 强改写 | +0.24 | 不显著 (p=0.67) | 0.87 |
| C3 保格式改写 | +0.18 | 不显著 (p=0.67) | 0.89 |
| **C4 语义漂移**(负对照) | **−0.02**(重建被摧毁) | **显著 (p<1e-5)** | 0.51 |

Friedman omnibus χ²=25.3,**p=4.3e-5**(由 C4 驱动)。

**三档 AR 的 ΔFVE 谱**(set-level FVE,5 折 CV)——用来定位通道:

| AR(表面敏感度) | C0 | C1 | C2 | C3 | C4 |
|--|--|--|--|--|--|
| bow(纯表面 n-gram) | −0.02 | −0.02 | −0.02 | −0.02 | −0.02 |
| **readout(真实 AR,读 token)** | **+0.08** | +0.05 | +0.06 | +0.05 | **+0.01** |
| semantic(改写不变,地板) | −0.00 | −0.01 | −0.01 | +0.00 | −0.02 |

**三条收敛证据**:① 语义保持改写(C1–C3)不显著降重建;② 语义漂移(C4)显著摧毁重建,
证明重建真的依赖语义内容;③ 纯表面 bow-AR 根本无法重建(FVE<0)→ **无可利用的隐写/表面通道**。

**局限(诚实)**:信号弱(基线 FVE 仅 +0.08);readout AR 偏语义(但 bow 的失败补上此缺口);
0.5B + 训练-free AV + N=304 是**轻量代理 / 假设筛选**,非论文复现。详见 **[results_exp02.md](docs/results_exp02.md)**
与诊断记录 **[pilot_findings.md](docs/pilot_findings.md)**。

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
- [x] **M2-local 真实运行**:Qwen2.5-0.5B 在 MacBook 上跑通完整 pipeline(exp02_open,N=304)
- [x] **M3 稳健分析 + 通道定位**:5 折 CV、manipulation check、bow/readout/semantic 三档 AR 对比 → 支持 H1(见上方结果)
- [ ] M4 扩样 + 换更大改写器(Qwen2.5-1.5B)提质 + 终局复现(训练真实 AV/AR)

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
