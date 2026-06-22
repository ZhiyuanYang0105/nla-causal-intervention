# NLA Causal Intervention

对 transformer-circuits 2026《Natural Language Autoencoders》的因果干预实验。

NLA 在**激活空间**自编码:`激活 h_l ──AV──▶ 解释文本 z ──AR──▶ 重建激活 ĥ_l`。
本项目要回答的问题:**干预中间文本 `z`(保持语义地改写)后,AR 的重建会不会变差?**
- 若**只有改变语义(漂移)才变差** → 信息走**语义通道**(解释忠实,H1)
- 若**保持语义的改写也让它变差** → 信息藏在**措辞表面** → **隐写通道**(解释不忠实,H2)

> 隐写只可能在 **AV/AR 联合训练**时产生(AV 学会把信息编码进措辞、AR 学会解它),
> 所以本项目忠实地按论文**联合训练** AV/AR,再做干预测试——这是能检验隐写的前提。

## 方法(忠实于论文,唯一替换:闭源 Claude → 开源 Qwen2.5)

- **AV (Activation Verbalizer)** `h_l → z`:激活单位归一化后作为一个 token embedding **注入**,自回归生成解释。
- **AR (Activation Reconstructor)** `z → ĥ_l`:`z` 过模型**截断到前 l 层**,取末 token 激活 + 学习的 **affine map**。
- **训练**:
  - **warm-start SFT**:AV 学 `h_l→摘要`、AR 学 `摘要→h_l`(用更强的 Instruct 模型生成摘要)。
  - **GRPO 联合训练**:AV 用强化学习(奖励 = −重建误差,AR 当固定打分器),AR 用监督 MSE,二者更新解耦。
  - **KL 惩罚**:把 AV 锚在 warm-start 状态附近(精确逐 token KL),保持解释流畅。
- **干预测试**:对留出激活生成 z → 施加条件(C1–C3 保语义改写 / C4 语义漂移负对照)→ 重建 → 比较。
  分析带**操纵门控**(过滤未保语义/未真漂移的样本)、**set-level FVE** + 稳健 cosine 配对检验、
  以及 **mechanism regression**(控制语义相似度后看"表面扰动"是否解释重建下降)作为 H1/H2 判别器。

## 当前状态(诚实)

- ✅ **代码完整、已端到端验证**:`build → train → steg` 三阶段在本地(Qwen2.5-0.5B)真实跑通;46 个单元测试通过;设备自动选择(cuda/mps/cpu)。
- ⚠️ **本地 0.5B 结论:UNDETERMINED(无法判定)**。原因是规模限制,**不是方法问题**:
  - 0.5B 改写器太弱 → 负对照(漂移)漂不动,门控会判 INVALID-CONTROL;
  - 0.5B 重建信号弱、接近噪声底,干预测试功效不足。
- 🎯 **要得到可信的 H1/H2 判定**:在 **A100/H100** 上用 **Qwen2.5-1.5B** + **Qwen2.5-7B-Instruct 改写器**跑(配置 `experiments/exp05_hpc`)——更强的模型 + 真能漂移的负对照 + 足够功效。
- 规模边界:模型仍远小于论文的 Claude 级,结论是"关于这个 1.5B NLA 的可信科学结论",不自动外推。

## 如何运行

### 1. 安装
```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"            # 轻量:跑指标/统计/单元测试
make test                          # 46 个单元测试
# 需要跑模型时:
pip install -e ".[models,nlp]"     # torch / transformers(>=4.56)/ datasets / peft / sentence-transformers
```

### 2. 本地小验证(强烈建议在花钱租 GPU 前先跑)
```bash
make smoke      # Qwen2.5-0.5B, N=40, 全流程三阶段, ~10 分钟。exit 0 即代码路径无误。
```

### 3. HPC(A100/H100,1.5B,推荐 80GB)—— 两步
```bash
# STEP 1(登录节点,有网,无需 GPU):建环境 + 装依赖 + 预下模型 + 预取 FineWeb 切片
bash scripts/hpc_setup.sh
# STEP 2(改好 SBATCH 头与 conda 行后)提交 GPU 作业(此后可全程离线运行)
sbatch scripts/run_hpc.slurm
```
**为什么分两步**:许多集群的**计算节点是断网的**。`hpc_setup.sh` 在登录节点把模型和语料都备好,
GPU 作业用本地语料(`--text-file`)+ `HF_HUB_OFFLINE` **离线跑**——既能在断网计算节点上跑,又**不浪费付费 GPU 时间**下载/装包。

**提交前按集群改**(脚本里已注释):`run_hpc.slurm` 顶部 `#SBATCH`(`--partition`、GPU 约束如 `--constraint=a100_80gb`)和 `conda activate` 那行;两个脚本的 `HF_HOME` 必须一致;墙内可在 `hpc_setup.sh` 设 `HF_ENDPOINT=https://hf-mirror.com`。

脚本顶部 knob:`N=20000`、`GRPO_STEPS=1000`、`N_EVAL=500` 等。
**省钱建议**:正式跑前先把这些调小(如 `N=200/GRPO_STEPS=20`)`sbatch` 一次,确认环境+流程通(几分钟、几块钱),再改回大参数。

### 流水线三阶段(`run_hpc.slurm` 依次执行,也可单独 `make` 跑)
| 阶段 | 命令 | 做什么 | 产物 |
|--|--|--|--|
| ① 建数据 | `make nla-data` | 跑 M 采集激活 `h_l` + 批量生成摘要 | `data/interim/<run>/{acts.npz, summaries.json}` |
| ② 训练 | `make train-nla` | warm-start SFT + GRPO 联合训练 | `results/<run>/ckpt/` |
| ③ 干预测试 | `make steg` | 共训练 AV/AR 上的隐写干预 + 门控判定 | `results/<run>/intervention_{metrics.csv,report.json}` |

> 注:`data/`、`results/`、`.venv/` 已 gitignore(不进仓库)。模型权重从 HuggingFace 自动下载。

## 结构

```
src/nla_intervention/
  nla/          忠实 NLA:av(激活注入) / ar(截断+affine) / train(warm-start + GRPO + KL)
  conditions/   干预变换 (paraphrase/drift/...) + apply_condition + 改写器(LLM/Fake)
  metrics/      reconstruction(FVE/cosine,激活) / semantic(sim_zz') / token-shift
  stats/        配对检验 + 机制回归(随机效应塌缩时回退 OLS+聚类稳健 SE)+ 功效
  data/         激活采集器(跑 M → 收 h_l,单位归一化)
  utils/        config 加载(extends)/ 设备选择 / seed
scripts/
  build_nla_data.py   ① 采集激活 + 摘要
  train_nla.py        ② warm-start + GRPO
  steg_intervention.py ③ 隐写干预测试(门控分析)
  hpc_setup.sh        登录节点一次性:环境+依赖+预下模型+预取语料(离线就绪)
  run_hpc.slurm       A100/H100 GPU 作业:离线三阶段
experiments/    exp03_nla / exp04_nla(0.5B 本地)/ exp05_hpc(1.5B HPC)/ smoke(验证)
configs/        default.yaml(共享基础)+ conditions.yaml(干预条件)
docs/           方法依据、忠实复现 findings、实验设计、HPC runbook
tests/          46 个单元测试
```

## 文档
- [论文事实依据](docs/paper_findings.md) — AV/AR/FVE 与论文的 GRPO 训练方法
- [忠实复现 findings](docs/nla_faithful_findings.md) ⭐ — 实现、结果、与论文的逐项忠实度核查、结论
- [研究计划](docs/research_plan.md) / [评价指标](docs/metrics.md) / [统计方案](docs/statistical_analysis.md)
- [HPC/GPU 复现 runbook](docs/autodl_plan.md) — A100/H100 + SLURM、β 扫描、多种子

## 忠实度边界(详见 findings)
与论文一致:激活注入 AV、截断 AR + affine、解耦的 AV(RL)+AR(MSE) 联合训练、KL→warm-start、单位归一化激活、末 token 池化。
为本地可行性保留的偏差:LoRA 而非全量微调、reward 用 −MSE 而非 −log、GRPO 未用 PPO clipping、AR 用回放多步。
