# 论文事实依据 (Natural Language Autoencoders)

来源: https://transformer-circuits.pub/2026/nla/index.html （Anthropic, transformer-circuits 2026）
本文件记录从论文抽取、用于约束本项目设计的事实。**关键:NLA 在激活空间运作,不是文本→文本。**

## 数据流

```
激活 h_l ──AV──▶ 自然语言解释 z ──AR──▶ 重建激活 ĥ_l
```

被自编码的对象是目标模型 M 的**残差流激活向量** `h_l`;中间文本 `z` 是 bottleneck。

## AV (Activation Verbalizer)
- 与目标模型 M **同架构的 LLM**,由 M 复制初始化后微调。
- 输入:单个激活 `h_l`(配一个固定 prompt + 表示激活的特殊 token);输出:自然语言解释文本 `z`。
- 每个激活采样**一组**解释:`z ∼ AV_φ(· | h_l)`。
- 解释含有 confabulation(编造),风格为「短段落 + 加粗主题标题」(来自初始化用的 Claude 摘要)。

## AR (Activation Reconstructor)
- 与 M **同架构但截断到前 `l` 层**的 LLM,外加一个**学习到的 affine map**。
- 输入:解释文本 `z`;输出:重建的激活 `ĥ_l`。

## 初始化 / 训练
- AV、AR 均为 M 的副本,用 **Claude Opus 4.5 生成的摘要**做 text-summarization proxy 任务微调。
- 训练目标:最小化激活重建误差,数据为「pretraining-like text」语料上采集的激活。

## 目标模型 M
- 论文在 **Claude Haiku 3.5、Haiku 4.5、Opus 4.6** 上构建 NLA;case study 主要用 **Opus 4.6**。
- 用于 Opus 4.6 上线前 pre-deployment audit。

## 激活采集
- 「compute layer-l activations h_l from the final token of randomly truncated pretraining-like text snippets」
- 层 `l`:中到后层(middle-to-late);残差流;**最后一个 token**;随机截断片段。
- 论文未公开:具体层号、语料名称、样本量、held-out 切分。

## 重建指标
- **FVE = 1 − ℒ / 𝔼‖h_l − h̄_l‖²**(Fraction of Variance Explained)。
- FVE=0 → 预测均值;FVE=1 → 完美重建。论文 NLA 达 **0.6–0.8 FVE**。
- ⚠️ 重建在**激活空间**度量,非文本 BLEU/ROUGE。

## 已有干预实验(最接近本项目)
- 编辑解释文本(`rabbit→mouse, habit→house, carrots→cheese`),算 steering 向量 `Δ = ã_edit − ã_orig` 注入残差流,看模型补全是否从 "rabbit" 变 "mouse/house"。
- 注意:这是**改语义的 edit**,**不是语义保持的 paraphrase**。→ 本项目的 paraphrase 干预填补此空白。

## 对本项目设计的影响
1. 「输入 x」= 激活向量 `h_l`,**不是文本**。
2. outcome = **FVE / 激活 cosine / 激活 MSE**,不是文本相似度。
3. 「数据集」= 跑 M 在 pretraining-like 文本上采集的**激活集**(公开替代: FineWeb / The Pile / C4)。
4. semantic manipulation check `sim(z,z')` 仍在**文本**上做(解释文本的语义相似度)。
5. 干预条件(paraphrase 系列)仍作用在解释文本 `z` 上 —— 这部分设计不变。
6. 论文的 rabbit→mouse edit 可作为 semantic-drift 条件的**已知方向校验**(known-direction validity check)。

## 实操约束 / 本项目复现配置
- AV/AR 是 Claude 模型的微调副本,**非公开**。本项目走**开源复现路线**:
  - **M = `meta-llama/Llama-3.1-8B`**(base,非 Instruct):32 层、hidden 4096,取**中后层 l≈20**(pilot 扫 {16,20,24})残差流、最后 token 的 `h_l`。
  - **AV** = Llama-3.1-8B 副本微调(M2 训练);**AR** = Llama-3.1-8B 截断到前 l 层 + 学习的 affine map(→ R⁴⁰⁹⁶)。
  - **激活语料 = FineWeb**(`HuggingFaceFW/fineweb`,`sample-10BT` 子集,streaming);留 ~2 万文档做 held-out eval,与 AV/AR 训练文档不重叠。
- 配置见 `configs/models.yaml` 与 `configs/default.yaml`。
