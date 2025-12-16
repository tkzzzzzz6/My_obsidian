---
title: "IEEE ICASSP 2026 || Barcelona, Spain || 3-8 May 2026"
source: "https://cmsworkshops.com/ICASSP2026/papers/author_response.php"
author:
published:
created: 2025-12-16
description:
tags:
  - "算法与数据结构"
---

## 📝 投稿说明 | Submission Instructions

**English**: Submit your reviewer response by writing text only in the box at the bottom of the page. One response should be written, addressing as many reviews as you choose. You may identify each review by the 4-character code in the red boxes.

**中文**: 请在页面底部的文本框中撰写您对审稿人的回复。应撰写一份回复，回应您选择的尽可能多的评审意见。您可以通过红框中的4位字符代码识别每个评审。

---

## 📄 论文信息 | Paper Information

### 论文编号 | Paper Number
**7252**

### 论文标题 | Paper Title
**TOWARD ROBUST SPATIAL MULTI-OMICS INTEGRATION: SPATIAL MULTI-OMICS INTEGRATION VIA ALIGNMENT AND SCHEDULED DIFFUSION**

**中文翻译**: 迈向稳健的空间多组学整合：通过对齐和调度扩散进行空间多组学整合

### 作者信息 | Authors
- Ke Tan, Sichuan Agricultural University (四川农业大学)
- Guihua Yu, Sichuan Agricultural University (四川农业大学)
- Tingting Li, Sichuan Agricultural University (四川农业大学)
- Haoran Chen, Sichuan Agricultural University (四川农业大学)
- JianChao Liu, Sichuan Agricultural University (四川农业大学)
- Mengzhu Wang, Hebei University of Technology (河北工业大学)
- Hau-sing So, University of Macau (澳门大学)

---

## 👥 评审意见 | Reviewers

---

### 📋 评审人 4069 | Reviewer 4069

#### ✅ 评分概览 | Rating Overview

**1. 是否在会议范围内且与ICASSP 2026相关？ | Is the work within the scope of the conference and relevant to ICASSP 2026?**
- **评分 | Rating**: Clearly within scope
- **中文**: 明确在范围内

**2. 稿件技术上是否正确？ | Is the manuscript technically correct?**
- **评分 | Rating**: Some minor concerns that should be easily corrected without altering the contribution or conclusions
- **中文**: 存在一些小问题，应该可以轻松纠正，不会改变贡献或结论

**3. 技术贡献是否新颖？ | Is the technical contribution novel?**
- **评分 | Rating**: Substantial novelty, with clearly identifiable new methods/concepts
- **中文**: 具有实质性新颖性，有明确可识别的新方法/概念

**4. 实验验证水平是否充分？ | Is the level of experimental validation sufficient?**
- **评分 | Rating**: Limited but convincing
- **中文**: 有限但令人信服

**5. 技术贡献是否重要？ | Is the technical contribution significant?**
- **评分 | Rating**: Substantial contribution, with a clear potential for impact
- **中文**: 实质性贡献，具有明确的影响潜力

**6. 参考文献是否合适，没有重大遗漏？ | Are the references appropriate, without any significant omissions?**
- **评分 | Rating**: Complete list of references without any significant omissions
- **中文**: 完整的参考文献列表，没有重大遗漏

**7. 是否有不相关的参考文献？ | Are there any references that do not appear to be relevant?**
- **评分 | Rating**: All references are directly relevant to the contribution of the manuscript
- **中文**: 所有参考文献都与稿件的贡献直接相关

**8. 稿件结构是否合理且写作是否清晰？ | Is the manuscript properly structured and clearly written?**
- **评分 | Rating**: Some minor structural, language, or other issues of exposition that would be easily rectified
- **中文**: 存在一些小的结构、语言或其他表述问题，可以轻松纠正

#### 💬 评审意见详情 | Comments to the Author(s)

**Summary:**
> This paper introduces TRSMI, a spatial multi-omics integration framework combining a near-identity encoder, temperature-controlled cross-graph alignment, and multi-scale diffusion. The problem is relevant and the overall design is coherent. Experimental results are promising, but several aspects require clearer explanation and stronger empirical support.

**中文翻译-摘要**：
> 本文介绍了TRSMI，一个空间多组学整合框架，结合了近恒等编码器、温度控制的跨图对齐和多尺度扩散。问题是相关的，整体设计是连贯的。实验结果很有希望，但有几个方面需要更清晰的解释和更强的实证支持。

**Strengths:**
> (1) Addresses an important and timely topic in spatial multi-omics.
> (2) Provides a complete pipeline with encouraging initial performance.

**中文翻译-优点**：
> (1) 解决了空间多组学中一个重要且及时的话题。
> (2) 提供了一个完整的流程，具有令人鼓舞的初步性能。

**Major Issues to Address:**

**1. Several components resemble existing methods, and the paper should explicitly clarify what is new:**
> (1) RPR near-identity encoder resembles residual/identity-preserving GCN variants (e.g., CED, GCNII). Clarify the unique contribution of RPR.
> (2) Temperature-controlled cross-graph alignment is similar to soft attention alignment in SpatialGlue and related cross-modal matching. Explain how this differs and why it is needed.
> (3) Multi-scale APPNP diffusion + scheduled gate is close to existing multi-scale propagation schemes. Further justify what is novel in the scheduled global-to-local transition.

**中文翻译-主要问题1**：
> **几个组件与现有方法相似，论文应明确阐明什么是新的：**
> (1) RPR近恒等编码器类似于残差/恒等保持GCN变体（例如，CED，GCNII）。请阐明RPR的独特贡献。
> (2) 温度控制的跨图对齐类似于SpatialGlue和相关跨模态匹配中的软注意力对齐。请解释其不同之处以及为什么需要它。
> (3) 多尺度APPNP扩散+调度门接近现有的多尺度传播方案。请进一步证明调度的全局到局部转换中的新颖性。

**2. The following details are critical for reproducibility but insufficiently described:**
> (1) Graph fusion: selection and normalization of w\_S,w\_F; whether the fused adjacency is symmetrized or renormalized.
> (2) Top-k sparsification: how k is chosen and whether sparsification affects alignment stability.
> (3) Diffusion parameters: rationale for multi-scale APPNP settings and the update rule for the scheduled gate β.
> (4) Prototype-based clustering: initialization of prototypes and update procedures.
> (5) EMA regularization: definition of the EMA reference graph and decay settings.

**中文翻译-主要问题2**：
> **以下细节对可重复性至关重要，但描述不足：**
> (1) 图融合：w\_S、w\_F的选择和归一化；融合的邻接矩阵是否对称化或重新归一化。
> (2) Top-k稀疏化：如何选择k以及稀疏化是否影响对齐稳定性。
> (3) 扩散参数：多尺度APPNP设置的理由以及调度门β的更新规则。
> (4) 基于原型的聚类：原型的初始化和更新过程。
> (5) EMA正则化：EMA参考图的定义和衰减设置。

**3. Recommendations:**
> Add standard deviations or confidence intervals to support reported improvements. Include runtime and memory comparisons to demonstrate scalability. Provide more qualitative visualizations to show spatial boundary preservation and alignment quality.

**中文翻译-主要问题3**：
> **建议：**
> 添加标准差或置信区间以支持所报告的改进。包括运行时间和内存比较以展示可扩展性。提供更多定性可视化以显示空间边界保持和对齐质量。

---

### 📋 评审人 1DFB | Reviewer 1DFB

#### ✅ 评分概览 | Rating Overview

**1. 是否在会议范围内且与ICASSP 2026相关？ | Is the work within the scope of the conference and relevant to ICASSP 2026?**
- **评分 | Rating**: Clearly within scope
- **中文**: 明确在范围内

**2. 稿件技术上是否正确？ | Is the manuscript technically correct?**
- **评分 | Rating**: Some minor concerns that should be easily corrected without altering the contribution or conclusions
- **中文**: 存在一些小问题，应该可以轻松纠正，不会改变贡献或结论

**3. 技术贡献是否新颖？ | Is the technical contribution novel?**
- **评分 | Rating**: Substantial novelty, with clearly identifiable new methods/concepts
- **中文**: 具有实质性新颖性，有明确可识别的新方法/概念

**4. 实验验证水平是否充分？ | Is the level of experimental validation sufficient?**
- **评分 | Rating**: Lacking in some respect
- **中文**: 在某些方面缺乏

**5. 技术贡献是否重要？ | Is the technical contribution significant?**
- **评分 | Rating**: Insufficient contribution for a full-length regular paper, but suitable for short paper
- **中文**: 对于全长常规论文贡献不足，但适合短论文

**6. 参考文献是否合适，没有重大遗漏？ | Are the references appropriate, without any significant omissions?**
- **评分 | Rating**: A largely complete list of references with only minor omissions that would not affect the novelty of the submission
- **中文**: 基本完整的参考文献列表，只有小的遗漏，不会影响投稿的新颖性

**7. 是否有不相关的参考文献？ | Are there any references that do not appear to be relevant?**
- **评分 | Rating**: Some of the references are of limited relevance
- **中文**: 一些参考文献的相关性有限

**8. 稿件结构是否合理且写作是否清晰？ | Is the manuscript properly structured and clearly written?**
- **评分 | Rating**: Some minor structural, language, or other issues of exposition that would be easily rectified
- **中文**: 存在一些小的结构、语言或其他表述问题，可以轻松纠正

#### 💬 评审意见详情 | Comments to the Author(s)

**1)**
> This manuscript is within the scope and relevant to ICASSP as it introduces new techniques for spatial multi-omics.

**中文翻译**：
> 该稿件在ICASSP的范围内且相关，因为它引入了空间多组学的新技术。

**2)3)**
> The technical contents are new and appear to move the field in a stronger direction with their updated architecture over previous approaches along with the use of diffusion.

**中文翻译**：
> 技术内容是新的，并且通过其相对于先前方法的更新架构以及扩散的使用，似乎将该领域推向了更强的方向。

**4)**
> The metrics reported are averages of metrics without showing the actual metrics. Although these present similar information in some cases this is misleading especially as the actual metrics have not been presented anywhere. There is also no justification for presenting these averages, as the reference 23 presents individual metrics.
>
> Additionally, it would be good to present the mean +- std. of these results over multiple runs to properly show how much better the proposed method is over others by removing chance, unless the models weight initialisation is not random, although this is not stated.

**中文翻译**：
> 报告的指标是指标的平均值，而没有显示实际指标。尽管在某些情况下这些呈现了类似的信息，但这是误导性的，特别是因为实际指标没有在任何地方呈现。也没有理由呈现这些平均值，因为参考文献23呈现了单独的指标。
>
> 此外，最好呈现这些结果在多次运行中的平均值±标准差，以通过消除偶然性来正确显示所提出的方法比其他方法好多少，除非模型的权重初始化不是随机的，尽管这没有说明。

**6)7)**
> Reference 23's placement is misleading as it implies these averages are used in that work. All other references are good and help support the paper.

**中文翻译**：
> 参考文献23的位置具有误导性，因为它暗示这些平均值在该工作中使用。所有其他参考文献都很好，有助于支持论文。

**8)**
> The manuscript is mostly well written. However, there is cases of abbreviations never being expanded or being introduced after their first use. There is also excessive use of the emdash, bolding and use of italics.

**中文翻译**：
> 稿件大部分写得很好。但是，有些缩写从未展开或在首次使用后才引入。还过度使用了破折号、粗体和斜体。

**Overall:**
> It is recommended that proper transparent validation is completed to properly support all claims of SOTA performance.

**中文翻译-总体**：
> 建议完成适当的透明验证，以适当支持所有SOTA性能的声明。

---

### 📋 评审人 60DD | Reviewer 60DD

#### ✅ 评分概览 | Rating Overview

**1. 是否在会议范围内且与ICASSP 2026相关？ | Is the work within the scope of the conference and relevant to ICASSP 2026?**
- **评分 | Rating**: Clearly within scope
- **中文**: 明确在范围内

**2. 稿件技术上是否正确？ | Is the manuscript technically correct?**
- **评分 | Rating**: Some minor concerns that should be easily corrected without altering the contribution or conclusions
- **中文**: 存在一些小问题，应该可以轻松纠正，不会改变贡献或结论

**3. 技术贡献是否新颖？ | Is the technical contribution novel?**
- **评分 | Rating**: Substantial novelty, with clearly identifiable new methods/concepts
- **中文**: 具有实质性新颖性，有明确可识别的新方法/概念

**4. 实验验证水平是否充分？ | Is the level of experimental validation sufficient?**
- **评分 | Rating**: Limited but convincing
- **中文**: 有限但令人信服

**5. 技术贡献是否重要？ | Is the technical contribution significant?**
- **评分 | Rating**: Moderate contribution, with the possibility of an impact on the field
- **中文**: 中等贡献，有可能对该领域产生影响

**6. 参考文献是否合适，没有重大遗漏？ | Are the references appropriate, without any significant omissions?**
- **评分 | Rating**: A largely complete list of references with only minor omissions that would not affect the novelty of the submission
- **中文**: 基本完整的参考文献列表，只有小的遗漏，不会影响投稿的新颖性

**7. 是否有不相关的参考文献？ | Are there any references that do not appear to be relevant?**
- **评分 | Rating**: All references are directly relevant to the contribution of the manuscript
- **中文**: 所有参考文献都与稿件的贡献直接相关

**8. 稿件结构是否合理且写作是否清晰？ | Is the manuscript properly structured and clearly written?**
- **评分 | Rating**: Some minor structural, language, or other issues of exposition that would be easily rectified
- **中文**: 存在一些小的结构、语言或其他表述问题，可以轻松纠正

#### 💬 评审意见详情 | Comments to the Author(s)

**Summary:**
> The paper proposes TRSMI, a graph-signal-processing–inspired framework for integrating spatial multi-omics data (RNA, ATAC, ADT). It models each modality as a graph signal on spatial spots and combines near-identity propagation, soft cross-graph alignment, and multi-scale APPNP with scheduled global-to-local gating. Experiments on three public benchmarks show improved performance on clustering, boundary-sensitive, and classification metrics, supported by ablations and a small hyperparameter study.

**中文翻译-摘要**：
> 该论文提出了TRSMI，一个受图信号处理启发的框架，用于整合空间多组学数据（RNA、ATAC、ADT）。它将每个模态建模为空间点上的图信号，并结合近恒等传播、软跨图对齐和带有调度的全局到局部门控的多尺度APPNP。在三个公共基准上的实验显示，在聚类、边界敏感和分类指标上的性能有所改进，并得到消融实验和小型超参数研究的支持。

**Strengths:**
> (1) Addresses a timely and important problem of spatial multi-omics integration with explicit attention to spatial boundaries and microenvironments.
> (2) Presents a coherent GSP-based design (near-identity encoder, alignment, scheduled multi-scale diffusion) tailored to denoising while preserving boundaries.
> (3) Demonstrates strong empirical results on three benchmarks, with ablations showing complementary roles of the main modules and hyperparameter sweeps indicating reasonable robustness.

**中文翻译-优点**：
> (1) 解决了空间多组学整合的及时且重要的问题，明确关注空间边界和微环境。
> (2) 提出了一个连贯的基于GSP的设计（近恒等编码器、对齐、调度的多尺度扩散），专为去噪同时保持边界而定制。
> (3) 在三个基准上展示了强大的实证结果，消融实验显示了主要模块的互补作用，超参数扫描表明了合理的鲁棒性。

**Weaknesses:**
> (1) It is not clearly explained what is truly new compared to existing graph-based or multi-omics methods that use similar components.
> (2) The paper does not clearly show comparisons with the latest and strongest baselines, so the "state-of-the-art" claim is not fully convincing.

**中文翻译-缺点**：
> (1) 与使用类似组件的现有基于图的或多组学方法相比，没有清楚地解释什么是真正新的。
> (2) 论文没有清楚地显示与最新和最强基线的比较，因此"最先进"的声明并不完全令人信服。

---

## 📬 委员会意见 | Committee Comments

##### Provide your Response here

Provite your text-only response here. You may identify each review by the 4-character code in the red boxes.

**中文说明**: 请在此处提供您的仅文本回复。您可以通过红框中的4位字符代码识别每个评审。

**字数限制 | Word Limit**: 0 / 400 words