# 结果:exp02_open（本地真实 pilot）

> ⚠️ **结论已作废（保留作过程记录/反面教材）**。此 pipeline 有两个硬伤:
> (1) AV 用源文本摘要而非**激活**输入;(2) AV/AR **未联合训练**。隐写是联合训练的产物,
> 不联合训练就不可能出现 → 这里测的不是真正的问题,"支持 H1"的结论无效。
> 忠实复现见 [nla_faithful_findings.md](nla_faithful_findings.md)。

**配置**:M = Qwen2.5-0.5B（冻结，layer 12，mean-pool，float32），AV = 训练-free 摘要 proxy
（Qwen2.5-0.5B-Instruct，贪婪），AR = 冻结 readout + 闭式 ridge，N = 304（FineWeb），5 折 CV。
硬件:MacBook M5 / 16GB / MPS。全本地。

> ⚠️ 这是论文 NLA 的**轻量代理 + 假设筛选**,不是复现:0.5B 模型、训练-free AV、readout AR、N=304。
> 结论限定为"在该代理上"。

## 主结果（readout AR，5 折 CV，配对 Wilcoxon + Holm）

| 条件 | FVE 中位数 | set-level FVE | ΔFVE vs C0 | 显著性 | sim(z,z') |
|--|--|--|--|--|--|
| C0 original | +0.179 | +0.080 | — | — | 1.00 |
| C1 轻改写 | +0.196 | +0.054 | +0.03 | 否 (p=0.67) | 0.886 |
| C2 强改写 | +0.238 | +0.061 | +0.02 | 否 (p=0.67) | 0.866 |
| C3 保格式改写 | +0.179 | +0.054 | +0.03 | 否 (p=0.67) | 0.893 |
| **C4 语义漂移** | **−0.024** | **+0.006** | **+0.07** | **是 (p<1e-5)** | **0.508** |

Friedman omnibus χ²=25.3, **p=4.3e-5**（显著，由 C4 驱动）。
基线 C0 FVE > 0 ✓（AR 确实在重建）。manipulation check:改写组 sim≈0.88(保语义)、漂移组 sim≈0.51(改语义)，操纵有效。

## 通道定位:三档 AR 的 ΔFVE 谱（set-level FVE，5 折 CV）

| AR（表面敏感度） | C0 | C1 | C2 | C3 | C4 |
|--|--|--|--|--|--|
| **bow**（纯表面，n-gram） | −0.023 | −0.023 | −0.023 | −0.023 | −0.023 |
| **readout**（真实 AR，读 token） | +0.080 | +0.054 | +0.061 | +0.054 | +0.006 |
| **semantic**（改写不变，地板参考） | −0.002 | −0.009 | −0.009 | +0.001 | −0.022 |

## 结论:支持 H1（语义通道）

三条收敛证据:

1. **语义保持改写不显著降重建**:C1/C2/C3 的 FVE ≈ C0（中位数 +0.18~0.24），配对检验全不显著（Holm p=0.67）。
2. **语义漂移摧毁重建**:C4 把 FVE 打到 ~0，显著（p<1e-5）。负对照证明重建**真的依赖语义内容**。
3. **纯表面 AR（bow）根本无法重建**（基线 FVE = −0.023 < 0）:如果信息隐写在词面/格式里，bow 应该能重建——它不能 → **没有可利用的表面/隐写通道**。

→ **中间文本主要通过语义（meaning）把信息传给重建器,而非表面形式。** 这正是 H1 的预测。

## 诚实的局限

- **信号弱**:readout 基线 FVE 仅 +0.08,效应绝对值小。
- **readout AR 偏语义**:Qwen 中层 mean-pool 表示本身偏语义,对改写天然鲁棒——但 bow 的失败补上了这一缺口(纯表面真的没信号)。
- **C1-C3 仍有微小降幅**(ΔFVE≈0.03,不显著):不能完全排除极弱的表面贡献。
- **代理 ≠ 复现**:0.5B、训练-free AV、N=304。全量实验(训练 AV/AR、更大模型、更多样本)才是终局证据。
- 改写质量受 0.5B 限制:C2 强改写偶有语义损失(sim 0.87),C4 漂移靠强化 prompt 才达成(sim 0.51)。

## 复现

```bash
pip install -e ".[models,nlp]"
python scripts/run_local_pilot.py  --config experiments/exp02_open/config.yaml   # 6 阶段
python scripts/final_analysis.py   --config experiments/exp02_open/config.yaml   # 5 折稳健统计
# 三档 AR 谱见 git 历史中的内联脚本 / 可整理进 compare_ar.py
```

产物:`results/exp02_open/{metrics.csv, metrics_kfold.csv, stats_report.json, stats_report_kfold.json}`,
`data/interim/exp02_open/{acts.npz, z_zprime.jsonl, readout_feats.npz}`。
