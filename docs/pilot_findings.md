# Pilot Findings — 真实本地运行的诊断与修复

记录在 MacBook M5 / 16GB 上用 Qwen2.5-0.5B 真实跑 pilot 时发现的问题、根因分析、修复。
方法论价值 > 具体数字（小模型 + 小样本，数字仅供参考）。

## Run 1（exp02_open，N=112，last-pooling，readout AR alpha=10）

**结果**:Friedman p=0.085(不显著);各条件 ΔFVE 无显著差异。
**但 C0 基线 FVE = −0.40(负!)** → AR 比"预测均值"还差,**重建根本没成立**。
→ 主结果的"零效应"是 **AR 失效的噪声**,不是真实结论。无效。

## 根因分析(零成本诊断,全在缓存上做)

1. **扫 alpha**:readout/bow AR 在所有 alpha 下基线 FVE 都 ≤ 0;alpha→∞ 时 →−0.047(=均值预测)。
2. **PCA 降目标维**(k=8…64):仍负 → 不是目标太高维。
3. **MiniLM 语义特征**(384 维)→ h_l:仍负。
4. **MiniLM(源文本)→ h_l**(源文本正是产生 h_l 的文本,理论上界):**仍负!**
   → 排除"摘要耦合太弱",问题在**样本量**:n_train=84 ≪ 特征维度 → p>n,ridge 把一切正则化成均值。
5. **mean-pooling 目标**(而非 last-token):FVE 升到 −0.004(正好临界)。
   → 关键洞察:**last-token 激活主要反映随机截断处那个 token,不是整段语义**;
   全局语义 embedding 当然预测不了它。mean-pooling 让目标反映整体内容。
6. **特征+目标都 PCA 降到 32 维**(n=84>32):仍微负 → n=84 的估计本身太噪(eval=28)。

**结论**:三重问题叠加 —— (a) last-pooling 目标病态;(b) n 太小(p>n);(c) 0.5B 改写质量差
(C4 漂移没改掉意思、C2 强改写把语义改坏)。

## 修复(Run 2,exp02_open 改进版)

| 问题 | 修复 |
|--|--|
| last-token 目标病态 | `pooling: mean`(激活 + readout 特征都 mean-pool) |
| p>n 欠拟合 | 特征 PCA 降到 **64 维** + **alpha 交叉验证**(`RidgeReconstructor(feat_pca, alpha=[...])`) |
| 样本太少 | **N=300**(n_train≈210 > 64) |
| 改写质量差 | 强化 `paraphrase_strong`(强调保事实)和 `semantic_drift`(改成"写一个完全不同主题")prompt |

## 通用教训(写给全量实验)

- **AR 必须先有正的基线 FVE,paraphrase 干预才有意义**。先验证 `C0 FVE > 0`,否则一切 ΔFVE 是噪声。
- **pooling 选择是一等超参**:reconstruction 目标要匹配"中间文本能描述的粒度"(整体语义 → mean;
  特定位置 → last)。论文用 last-token 是因为有**训练过的** AV/AR 紧耦合;训练-free 代理下 mean 更稳。
- **小样本下用 PCA 压特征维 + CV alpha**,否则 p>n 直接退化成均值预测。
- 负对照(semantic-drift)必须**校验真的改了意思**(sim_zz' 显著低);0.5B 模型经常不听话。
- 多 AR 档对比(bow/readout/semantic)的 ΔFVE 谱只有在基线 FVE 为正时才可解读。

## 复现诊断

```bash
# Run 1(故障)的根因诊断脚本逻辑见 git 历史 / 本文件 §根因分析;
# 改进版:
make -s -C . >/dev/null 2>&1 || true
python scripts/run_local_pilot.py --config experiments/exp02_open/config.yaml --force
python scripts/compare_ar.py     --config experiments/exp02_open/config.yaml
```
