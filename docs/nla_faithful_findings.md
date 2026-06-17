# 忠实 NLA 复现:实现、运行与规模限制（exp03_nla）

回应一个正确的批评:此前的 pipeline 有两个硬伤——(1) AV 输入用的是源文本而非**激活**;
(2) 没有**联合训练** AV/AR,而隐写正是联合训练的产物,不联合训练当然测不出隐写。
本轮按论文方法忠实重建,唯一替换是 Opus 4.6 → Qwen2.5-0.5B(免授权,本地可跑)。

## 忠实实现(全部已建并验证可跑)

| 组件 | 实现 | 文件 |
|--|--|--|
| **AV** | 激活注入:`h_l` 归一化×常数 → 替换占位 token 的 embedding → 自回归采样 z;LoRA | `src/nla_intervention/nla/av.py` |
| **AR** | z 套固定 prompt → 过模型 → 取 `hidden_states[l]` 末 token → 学习的 affine map → ĥ_l;LoRA+affine | `nla/ar.py` |
| **warm-start** | SFT:AV 学 h_l→s、AR 学 s→h_l;按 eval 早停 | `nla/train.py` |
| **GRPO 联合** | 组采样 z、reward=−‖h_l−AR(z)‖²、AR 当固定打分器;AV 用 GRPO 策略梯度、AR 监督 MSE;**更新解耦**;KL-to-init 惩罚(LoRA disable 取 init) | `nla/train.py` |

架构 sanity 通过:注入真实 `h_l` → AV 生成连贯文本、序列 logprob 可微、AR 重建可微。

## 本地规模下的结果(Qwen2.5-0.5B / MacBook M5 / ~700 激活)

**warm-start**(数据 240 → 700 的对比,关键):
- 240 样本:eval AR-only FVE 最高 ~−0.03(过拟合,不泛化)
- **700 样本:eval AR-only FVE 最高 +0.109**(早停@ep3)——**泛化随数据量出现了**
- chain FVE(AV→AR 贪婪)= −0.04 → **瓶颈在 AV 生成**(AR 能从真摘要重建,AV 自采样质量不够)

**GRPO 联合训练**(100 步,基于 700 样本 warm-start):
- 循环跑通、无报错;reward/MSE **剧烈震荡**(0.16~1.3),chain eval_FVE 在 −0.10~+0.00 间**徘徊,无上升趋势**(step0 −0.024 → step50 +0.001)
- 即 GRPO 在该规模下**不收敛、抬不动 chain FVE**

## 结论:方法忠实,但本地规模不足以训出可测隐写的 NLA

- 论文 warm-start 达 0.3–0.4 FVE、RL 后 0.6–0.8,靠的是**大模型 + 大规模数据 + 大量算力**。
- 本地 0.5B / 700 样本:warm-start 仅泛化到 +0.11、GRPO 不收敛 → **达不到"正且稳定的 chain FVE"**,
  而这是做 paraphrase 隐写干预的前提。
- 因此**隐写测试在本地规模下无法可信地进行**——这是**规模限制(用户施加的硬件约束),不是方法错误**。
  trajectory(240→700 让 FVE 从 −0.03 升到 +0.11)表明:**加数据是对的方向**。

## 训练(exp04_nla,N=2500)

- 数据 240→700→2500,warm-start AR-only FVE 升到 **+0.14**。
- **KL bug 已修**:论文 KL 朝 `AV_φ_init`(warm-start **之后**的 AV);原实现用 `disable_adapter()`
  错取到了**底座**。修复=GRPO 前冻结一份 warm-start 后的 AV 作参考(验证:step0 `kl/tok≈0`,旧 bug 版~1.0),
  并改用**精确逐 token KL**(原先精确 KL 函数是死代码)。修复后 **chain FVE 从 +0.06 升到 +0.175**(峰值过 +0.29)。
- **AR 截断已修**:AR 原本跑全模型读 `hidden_states[l]`,导致 >l 层的 LoRA 是死的;现截断到前 l 层
  (=论文 "truncated to its first l layers"),消除死 LoRA。

## 隐写干预:分析已修正,本地结果作废

一次独立审计指出干预分析有多处缺陷,已全部修正(`scripts/steg_intervention.py`):
- **操纵门控**:先按 sim 阈值**过滤**掉没保住语义/没真漂移的样本,再统计;若**负对照(漂移)未真正漂移则判 INVALID-CONTROL**。
- **稳健统计**:头条用 **set-level FVE**(论文指标),配对显著性检验改用 **per-sample cosine**(有界稳健),
  不再用病态的 per-sample FVE 比值(它在 `‖h−h̄‖` 小时会爆到 −20)。
- **真正的判别器**:在保语义子集上跑 `mechanism_regression`——**控制 sim 后看 surface_shift 系数**是否显著解释重建下降,
  这才是 H1/H2 的判别(而非"数几个 pairwise 显著")。

> ⚠️ **此前本地记录的具体数字(n=120 / n=350 的 FVE 表)已作废**,原因有二:(a) AR 截断修复改变了模型,旧 ckpt 不兼容、需重训;
> (b) 分析管线已重做。早期文档里"操纵有效"的说法也**不成立**(见下)。

**本地 0.5B 为何无法支撑任何 H1/H2 判定**(修正后的门控会直接判 INVALID-CONTROL / UNDETERMINED):
- **负对照不可靠**:C4 漂移平均 sim≈0.58、约半数未低于阈值 0.60——0.5B 改写器太弱,漂移≈"第四种改写",负对照破了。
- 约 **40% 的"保语义"样本未达 sim≥0.85**(未过门控)。
- **NLA 太弱**:set-level chain FVE 仅 ~+0.06,重建信号接近噪声底。

→ **隐写问题在本地 0.5B 规模 UNDETERMINED**,且无法靠它得出任何方向性结论。要判定需 **1.5B/5090 + 更强改写器 + 本套门控/机制分析**(见 [autodl_plan.md](autodl_plan.md))。

## 与论文的忠实度核查(完整对照)

逐组件核对(论文 = transformer-circuits 2026,唯一许可替换:Opus 4.6/4.5 → Qwen2.5-0.5B)。

### 与论文一致 ✅
| 组件 | 论文 | 实现 |
|--|--|--|
| AV 激活注入 | 归一化单位 L2 + 固定常数缩放,替换 token embedding,T=1 采样 | `h/‖h‖·act_scale`,占一 embedding 槽,temperature=1.0 |
| AR 结构 | 截断前 l 层 + 末 token layer-l 激活做 affine,z 套固定 prompt | 全模型取 `hidden_states[l]`(数值等价截断)+ affine + 固定 prompt |
| 联合训练解耦 | AR 不回传 AV,AR 当固定打分器 | reward 在 `no_grad` 下用 AR;AV/AR 优化器独立 |
| 组采样 | 每个 h_l 采样一组候选 z | `av.generate(h, n=group)` |
| **KL → AV_init** | β·D_KL(AV_φ‖AV_φ_init),朝 warm-start 后的 AV | 冻结 warm-start AV 作 `ref_av`(**本轮修复**;旧版错朝底座) |
| warm-start | (h_l,s) 训 AV、(s,h_l) 训 AR,s=文本摘要 | 同 |
| FVE | 1 − ℒ/E‖h−h̄‖² | 同 |

### 偏差 ⚠️(诚实列全)
| # | 项 | 论文 | 实现 | 原因 |
|--|--|--|--|--|
| A | GRPO 目标 | 完整 GRPO(组相对优势 + **PPO clipped ratio** + KL) | 组相对优势×logprob + KL,**无 clipping**(≈REINFORCE+组基线) | 实现简化 |
| B | reward | −log‖h−AR(z)‖² | −MSE(均值) | 实现简化 |
| C | 微调方式 | M 的**完整副本全量微调** | **LoRA** 适配器(冻结底座) | 16GB 硬约束 |
| D | 激活池化 | **末 token** | **mean-pool** | 末 token 本地 FVE 近零 |
| E | AR 更新 | 当前采样 z 上**单步** | 回放缓冲 **4 步/轮** | 稳定 GRPO |
| — | 模型 | Opus 4.6/4.5 | Qwen2.5-0.5B / -Instruct | 你许可的替换 |

**结论**:**训练算法的结构与论文一致**(激活注入 AV、affine AR、解耦的 AV 强化 + AR 监督联合训练、KL-to-init、warm-start),
且关键 KL bug 已修。但**非逐字符忠实**:#A(GRPO 简化掉 clipping)、#B(reward 非 log)是算法层简化,
#C/#D/#E 是本地可行性/稳定性妥协。#A/#B/#E 原则上可改严格;#C/#D 受硬件/本地信号限制。

## 要真正测隐写,需要(任一/组合)

1. **更大数据**:数千~上万 (h_l, summary) 对(主要成本是摘要生成)。
2. **更大模型**:1.5B–3B 做 AV/AR(论文用闭源大模型)。
3. **更稳的 GRPO**:AR 用经验回放/多步、更小 AV 学习率、KL 用 k3 估计、更大 group。
4. 更多算力/时间(GPU 或多小时本地)。

## 复现

```bash
pip install -e ".[models,nlp]" && pip install peft
python scripts/build_nla_data.py --out exp04_nla --n 2500   # 采集激活 + 摘要
python scripts/train_nla.py --config experiments/exp03_nla/config.yaml --ws-epochs 8          # warm-start
python scripts/train_nla.py --config experiments/exp03_nla/config.yaml --grpo --grpo-steps 100  # +GRPO
```
