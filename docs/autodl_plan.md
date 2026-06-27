# HPC / GPU 上的忠实 NLA 复现方案（runbook）

目标:用 **Qwen2.5-1.5B** 做**方法层面对论文忠实**的 NLA,并以足够功效**判定隐写(H1 语义通道 vs H2 表面/隐写通道)**。

> 边界(诚实):模型规模仍远小于论文的 Claude 级(官方未公布参数量),所以这是"**方法忠实、规模缩小**"的复现。
> 结论是"**关于这个 1.5B NLA**"的可信科学结论,不自动推广到 Claude 级。

---

## ⭐ 推荐路径:HPC(A100/H100)+ SLURM

代码已**自动选设备**(`cuda > mps > cpu`),mac 和 HPC 同一套。**两步**:

```bash
bash scripts/hpc_setup.sh           # STEP 1 登录节点(有网,无 GPU):环境+依赖+预下模型+预取 FineWeb 切片
sbatch scripts/run_hpc.slurm        # STEP 2 GPU 作业:离线三阶段;配置 experiments/exp05_hpc/config.yaml
```

**为什么分两步**:许多集群计算节点**断网**。`hpc_setup.sh` 在登录节点备好模型(`hf download`)
和本地语料(FineWeb 切片 → `data/raw/fineweb_slice.jsonl`);GPU 作业用 `--text-file` + `HF_HUB_OFFLINE=1`
**全程离线**跑,既兼容断网计算节点,又不浪费付费 GPU 时间下载/装包。`run_hpc.slurm` 跑:
① `build_nla_data`(1.5B 采集 + **7B-Instruct** 摘要)→ ② `train_nla`(warm-start + GRPO)→
③ `steg_intervention`(门控分析 + 判定)。

**提交前编辑**:`run_hpc.slurm` 的 SLURM 头(partition / GPU 约束)和 `conda activate` 行;
两脚本 `HF_HOME` 要一致。脚本顶部 knob:`N=20000`、`LAYER=14`、`POOLING=last`、`GRPO_STEPS=1000`、`N_EVAL=500`。

**为什么用 A100/H100 + 7B 改写器(对症本地的两个根本限制):**
- 7B-Instruct 改写器**让负对照真能漂移**(0.5B 太弱、漂移≈第四种改写,门控会判 INVALID-CONTROL)。
- 1.5B + 更多数据 → **更强更稳的 FVE**,干预测试才有功效。

**显存**:LoRA(r=32)1.5B 在 40GB 上轻松(训练时 AV+AR+冻结 ref ≈ 15GB fp32);
7B 摘要器/改写器用 **bf16 推理**(~14GB)。**推荐 80GB** 留足余量。

**忠实度**:`exp05_hpc` 用末 token 池化(论文)、激活单位归一化、精确 KL、截断 AR、**−log 重建奖励(#B 已一致)**。仍存的偏差(都在
[nla_faithful_findings.md](nla_faithful_findings.md) 的核查表):LoRA 而非全量微调(#C)、
GRPO 无 PPO clip(#A)、AR 多步回放(#E)——若要进一步消除,需在 HPC 上启用全量微调(需额外的 save/load 支持)。

---

## 0. 环境(autodl 特有坑,务必先过)

1. **租实例**:RTX 5090 / 32GB。选支持 **CUDA 12.8+** 的镜像(5090 是 Blackwell sm_120,旧镜像会报 "no kernel image")。
2. **PyTorch**:需 **≥2.7 + cu128**。验证:
   ```bash
   python -c "import torch;print(torch.__version__, torch.cuda.get_device_name(), torch.cuda.is_available())"
   # 期望: 2.7+ , NVIDIA GeForce RTX 5090 , True
   ```
   不对就装 nightly:`pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu128`
3. **国内网络**:HF 常连不上。二选一:
   - `export HF_ENDPOINT=https://hf-mirror.com`(用 datasets/transformers 走镜像),或
   - **ModelScope** 下 Qwen(国内、免授权):`pip install modelscope`;`Qwen2.5-1.5B`、`Qwen2.5-7B-Instruct` 都在。
4. 依赖:`pip install transformers datasets accelerate sentence-transformers bitsandbytes`
   - `bitsandbytes` 用于 **8-bit Adam**(全量微调省显存,关键)。
5. **模型选择**:M = `Qwen2.5-1.5B`(base);摘要器 = `Qwen2.5-7B-Instruct`(5090 上 bf16 推理够跑,摘要质量高于 0.5B);
   句子相似度(sim_zz')用 `BAAI/bge-base-en-v1.5` 或 sentence-transformers MiniLM(走镜像)。

### 显存可行性(为什么 1.5B 全量微调能塞进 32GB)
GRPO 时同时驻留:AV(全量) + AR(截断到 l 层,约半个) + 冻结 ref AV。
- 全量 Adam 太大(1.5B ≈ 18GB 单模型)→ **必须用 8-bit Adam + 梯度检查点**。
- 估算(8bit Adam):AV ~9GB + AR(半)~4.5GB + ref(冻结)~3GB + 激活/KV ≈ **~18–22GB**,32GB 可容。
- 若 OOM:退到 batch=1 / 更短 z / 或 M=Qwen2.5-1B。

---

## 1. 把代码改成严格忠实(修 #A–#E)

当前仓库为本地(0.5B/16GB)做了 5 处妥协,5090 上应改回论文设定:

| # | 改动 | 文件 | 做法 |
|--|--|--|--|
| **C** | LoRA → **全量微调** | `nla/av.py`,`nla/ar.py` | `lora=False`,训练全部参数;优化器用 `bitsandbytes.optim.AdamW8bit`;开 `model.gradient_checkpointing_enable()` |
| **D** | mean-pool → **末 token** | `experiments/*/config.yaml` | `data.pooling: last`;`MReadoutFeatures(pooling="last")`(已支持) |
| **B** ✅ 已完成 | reward = **−log‖·‖²** | `nla/train.py` grpo | 代码已用 `r = -torch.log(mse + 1e-6)`(无需再改) |
| **E** | AR 回放多步 → **单步** | `nla/train.py` grpo | 去掉 replay,改为每轮对当前采样 z 做一步 MSE(论文做法) |
| **A** | 简化 → **完整 GRPO(PPO clip)** | `nla/train.py` grpo | 采样时存 `logprob_old`;loss = `-min(ratio·adv, clip(ratio,1-ε,1+ε)·adv) + β·KL`,`ratio=exp(lp-lp_old)`,内层 1–2 个 mini-epoch,ε=0.2 |

KL 已修(朝 warm-start 后的 AV);保持。

> 提示:#A 的完整 GRPO 需要"采样-冻结-多次更新"的结构(off-policy 内循环)。可参考 `trl` 的 GRPOTrainer 实现思路,
> 但因为我们有自定义激活注入 + AR 奖励,手写更可控。先把 #B/#D/#E/#C 改了跑通,再上 #A。

---

## 2. 实验流程(带"闸门",不过关不往下)

### Stage 1 — 采集激活
`Qwen2.5-1.5B` 跑 FineWeb,**末 token**、float32 存。规模:**warm-start 用 ~30k,eval 留 ~2k**(互斥)。
GPU 上很快(几十分钟)。

### Stage 2 — warm-start SFT(全量)
用 `Qwen2.5-7B-Instruct` 批量生成摘要 s;全量微调 AV(h_l→s)、AR(s→h_l)。
**🚧 闸门 1**:warm-start 后 **AR-only FVE ≥ ~0.3**。达不到 → 加数据/查 pipeline,别往下。

### Stage 3 — GRPO 联合训练(忠实)
完整 GRPO + 单步 AR + log reward + 末 token。
**🚧 闸门 2**:**chain FVE ≥ ~0.4 且稳定**(多步评估不剧烈震荡)。这是隐写测试有功效的前提。

### Stage 4 — 隐写干预测试
C0–C4,真实改写器(7B-Instruct 改写质量更高),**n_eval ≥ 300**,配对 Wilcoxon + Friedman omnibus + Holm。
**🚧 闸门 3(负对照)**:**C4 语义漂移必须显著降 FVE**。若 C4 不降 → 测试无效(像本地那样),回头查。

### Stage 5 — β 扫描 + 多种子(关键:把"有没有"升级成"何时涌现")
对 **β ∈ {0.1, 0.02, 0.005}**(强→弱 KL)× **种子 ∈ {1,2,3}** 各跑 Stage 3–4。
- β 大:AV 贴 init,隐写空间小;β 小:自由度大,隐写更可能涌现。
- 看 **chain FVE 在保语义改写下崩不崩**,随 β 减小是否出现"崩"。

> 算力权衡:9 次 GRPO 偏多。建议先**单 β(0.02)+ 单种子**端到端跑通过三道闸门,再扩到 β×种子。

---

## 3. 判定规则(严格,防过度解读)

对每个 (β, seed),**先要求 Friedman omnibus 显著(p<0.05)且闸门 3 通过(C4 显著降)**,再读:

| 观测 | 判定 |
|--|--|
| C1–C3(保语义)多数**显著降 FVE**,降幅 ≈ 或 > C4 | **H2 表面/隐写通道**(该 β 下 AV 把信息藏进措辞) |
| C1–C3 **不显著**,仅 C4 显著降 | **H1 语义通道**(解释忠实) |
| 效应不一致 / omnibus 不显著 | **该设置下 undetermined** |

跨种子要**可复现**才算数;跨 β 看**隐写是否随自由度增大而涌现**。

---

## 4. 预算

- 5090 ≈ 几元/小时。
- 单次端到端(采数据+warm-start+1×GRPO+测试)≈ **半天到一天** GPU。
- 全 β×种子扫描 ≈ **2–4 天** GPU,**总成本约 ¥100–500**。

---

## 5. 诚实的预期

- ✅ 很可能拿到:**强基线 FVE(0.4–0.6)+ 足够功效的隐写测试**,在 1.5B 上**给出 H1/H2 的可信判定**,并通过 β 扫描刻画"隐写何时涌现"。
- ⚠️ 仍不保证:结果可能是"部分/混合"通道;结论限于该 1.5B NLA 与训练设置;**不自动推广到 Claude 级**。
- ❌ 拿不到:论文的 0.6–0.8 FVE 数字(那是大模型规模)。

> 一句话:5090 + 1.5B 让实验从"判不出(功效不足)"推进到"**大概率能判、且判得可信**"——但科学上"确定"二字给不了,能给的是**有条件的、可复现的判定 + β 维度的机制刻画**。
