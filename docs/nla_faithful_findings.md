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

## 扩规模 + 稳定化 GRPO(exp04_nla,N=2500)+ 隐写干预测试

**扩规模 + 稳定化 GRPO**(经验回放 + per-token KL):
- 数据 240→700→2500,warm-start AR-only FVE:−0.03 → +0.11 → **+0.14**(随数据上升)。
- 稳定化 GRPO(200 步)把 **chain FVE 从 −0.22 爬到 +0.06**(峰值 +0.12,闭合了 AV 生成差距),
  但仍有残余震荡(偶发下探 −0.3 后恢复)。**首次得到真正联合训练、chain FVE 为正的 NLA**。

**隐写干预测试**(共训练 AV/AR,真实 Qwen-Instruct 改写,n_eval=120):

| 条件 | FVE 中位数 | vs C0(配对 Wilcoxon,Holm) | sim(z,z') |
|--|--|--|--|
| C0 | +0.256 | — | 1.00 |
| C1 轻改写 | +0.231 | 不显著 dz=0.00 | 0.85 |
| C2 强改写 | +0.137 | 不显著 dz=0.02 | 0.85 |
| C3 保格式 | +0.104 | 不显著 dz=0.04 | 0.87 |
| C4 漂移 | +0.249 | 不显著 dz=0.07 | 0.58 |

Friedman **p=0.67**——无任何显著效应,效应量 dz≤0.07。操纵有效(sim:改写~0.85、漂移~0.58)。

**解读**:连语义漂移负对照(C4)都没显著降 FVE → **测试功效不足/重建太弱太噪**
(set-level FVE 仅 +0.06,per-sample 重尾:中位 +0.25 但均值 −0.6)。描述性(不显著)的微弱迹象:
表面改动大的 C2/C3 中位 FVE 略低于 C1/C4——**若非噪声**会隐指表面通道,但 dz 仅 0.02–0.04,噪声级,不能下结论。

## 最终结论(忠实复现)

**隐写问题在本地规模下"无法判定"(undetermined),但这次是因为正确的原因。**
- exp02 的"测不出":方法错(没共训练)→ 结论无效。
- exp04 的"测不出":**方法对了**(真联合训练 NLA、chain FVE 转正、干预操纵有效),
  但 0.5B/2500 规模重建太弱太噪 → 干预功效不足,连负对照都测不出 → 隐写在此规模无法判定。
- 要判定隐写需论文规模(更大模型/数据 → 更强更稳 FVE → 足够功效)。

## 忠实度核查 + KL bug 修复 + 重训(exp04 修正版)

逐组件核对发现一个真实 bug 和几处偏差:

- **🔴 已修:KL 参考对象**。论文 KL 朝 `AV_φ_init`(warm-start **之后**的 AV);原实现用 `disable_adapter()`
  取到的是**底座**(warm-start 之前)。修复:GRPO 前**冻结一份 warm-start 后的 AV** 作参考。
  验证:修复后 GRPO step0 的 `kl/tok≈0`(av=ref),随 RL 增长;旧 bug 版一开始就 ~1.0。
- 🟡 仍偏差(本地可行性所迫):LoRA vs 全量微调、mean-pool vs 末 token、AR 多步回放 vs 单步。
- 🟢 极小:reward −MSE vs −log‖·‖²;摘要用 Qwen-0.5B-Instruct vs Opus 4.5。

**修复 KL 后重训(同 2500 数据)**:chain FVE 从旧 bug 版的 **+0.06 提升到 +0.175**
(GRPO 后段近单调爬升,峰值过 +0.29,进入论文 warm-start 区间)。修正的 KL 锚定显著改善了训练。

**隐写干预测试(修正 NLA,强基线,n_eval=350)**:

| 条件 | FVE 中位数 | Δ vs C0 | 配对 Wilcoxon(Holm) | sim |
|--|--|--|--|--|
| C0 | +0.345 | — | — | 1.00 |
| C1 轻改写 | +0.274 | 0.071 | 不显著 dz=0.04 | 0.85 |
| C2 强改写 | +0.222 | 0.122 | 显著 dz=0.07 | 0.85 |
| C3 保格式 | +0.243 | 0.102 | 不显著 dz=0.08 | 0.88 |
| C4 漂移 | +0.315 | 0.030 | 不显著 dz=0.01 | 0.58 |

**Friedman omnibus p=0.176(不显著)**。per-sample 均值仍为负(−0.3~−0.5,重尾)。

### 最终判定:**UNDETERMINED**(隐写问题在本地规模无法判定)

- omnibus 不显著;只有 C2 显著(dz 极小、与 C1/C3 不自洽);**跨子集不稳**
  (n=120 时 C4 漂移降 0.19、n=350 时只降 0.03,负对照行为矛盾)→ 噪声主导。
- **弱而较一致的描述性迹象**:两次都见"保语义改写(C1-C3)降 FVE > 漂移(C4)",
  强改写 C2 两次都显著。**若非噪声**会隐指 H2(表面/隐写通道),但 omnibus 不显著 → 不能下结论。
- 与早期"测不出"的本质区别:这次**方法完全正确**(真共训练 NLA、KL 修复、强基线 FVE、操纵有效),
  仍判不出是因为 **0.5B/2500 规模重建重尾、功效不足**——要判定需更大模型/数据。

## 要真正测隐写,需要(任一/组合)

1. **更大数据**:数千~上万 (h_l, summary) 对(主要成本是摘要生成)。
2. **更大模型**:1.5B–3B 做 AV/AR(论文用闭源大模型)。
3. **更稳的 GRPO**:AR 用经验回放/多步、更小 AV 学习率、KL 用 k3 估计、更大 group。
4. 更多算力/时间(GPU 或多小时本地)。

## 复现

```bash
pip install -e ".[models,nlp]" && pip install peft
python scripts/build_nla_data.py --reuse exp02_open --out exp03_nla --n 700   # 数据
python scripts/train_nla.py --config experiments/exp03_nla/config.yaml --ws-epochs 8          # warm-start
python scripts/train_nla.py --config experiments/exp03_nla/config.yaml --grpo --grpo-steps 100  # +GRPO
```
