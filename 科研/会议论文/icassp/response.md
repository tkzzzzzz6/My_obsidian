We sincerely thank the reviewers (4069, 1DFB, and 60DD) for their constructive and encouraging feedback. We are pleased that all reviewers find the problem timely and relevant to ICASSP, the methodology technically sound, and the overall contribution novel and promising. Below we clarify the key novelty aspects and address concerns regarding reproducibility and experimental validation.

**Novelty clarification.**  
While individual components of TRSMI relate to prior work, our novelty lies in how they are _coupled and scheduled_ to address instability, over-smoothing, and boundary erosion in spatial multi-omics integration.  
(1) The proposed RPR encoder is not a generic residual or deep GCN variant (e.g., GCNII). It is intentionally shallow and near-identity, using a zero-initialized enhancement branch and row-wise ℓ2 normalization to constrain feature drift and preserve spatial boundaries, while deferring stronger diffusion to a global module.  
(2) The temperature-controlled cross-graph alignment is designed to stabilize training rather than simply compute soft correspondences. The temperature scaling and short warm-up detach explicitly prevent early noisy alignments, which we find critical in heterogeneous and imbalanced spatial omics, distinguishing it from static soft-attention alignment (e.g., SpatialGlue).  
(3) The scheduled multi-scale APPNP introduces a time-dependent global-to-local transition via a learnable gate, differing from existing static multi-scale propagation schemes by explicitly controlling training dynamics to balance global coherence and local boundary sharpening.

**Reproducibility and implementation details.**  
We agree that clearer descriptions are important for reproducibility. In the final version, we will explicitly detail: (i) graph fusion weights and normalization/symmetrization, (ii) top-k sparsification strategy and its effect on alignment stability, (iii) multi-scale APPNP parameter choices and the scheduled gate update, (iv) prototype initialization and updates, and (v) the EMA reference graph and decay settings.

**Experimental validation.**  
We acknowledge the concerns regarding evaluation transparency. In the final version, we will report individual metrics in addition to averaged scores, include mean ± standard deviation over multiple runs, and add runtime and memory comparisons to better demonstrate scalability. We will also clarify the positioning of our baselines and, where appropriate, include comparisons with additional recent and strong methods or refine the “state-of-the-art” claim to reflect the evaluated settings more precisely. Given the four-page ICASSP format, we will further improve the clarity and interpretability of the existing qualitative results and discussions to better highlight spatial boundary preservation and alignment behavior.

We believe these clarifications and additions will substantially strengthen the paper, and we thank the reviewers again for their valuable suggestions.

我们衷心感谢评审专家（4069、1DFB和60DD）提出的富有建设性且鼓舞人心的反馈意见。我们很高兴所有评审专家都认为该问题具有时效性，与ICASSP相关，所采用的方法在技术上是可靠的，整体贡献具有新颖性和前景。下面我们将阐明关键的创新点，并回应有关可复现性和实验验证的问题。

**创新点说明**：虽然TRSMI的各个组成部分与先前的研究有关联，但我们的创新之处在于这些部分的**耦合和调度方式**，以此解决空间多组学整合中的不稳定性、过度平滑和边界侵蚀问题。（1）所提出的RPR编码器并非通用的残差或深度GCN变体（例如GCNII）。它特意设计为浅层且接近恒等映射的结构，通过零初始化增强分支和行-wise ℓ2归一化来约束特征漂移并保留空间边界，同时将更强的扩散操作推迟到全局模块中进行。（2）温度控制的跨图对齐旨在稳定训练过程，而不仅仅是计算软对应关系。我们发现，温度缩放和短暂的预热分离能明确防止早期出现的噪声对齐，这在异质且不平衡的空间组学中至关重要，使其有别于静态软注意力对齐（例如SpatialGlue）。（3）调度式多尺度APPNP通过一个可学习的门控引入了时间相关的全局到局部过渡，与现有的静态多尺度传播方案不同，它通过明确控制训练动态来平衡全局一致性和局部边界锐化。

**可复现性和实现细节**：我们认同更清晰的描述对于可复现性至关重要。在最终版本中，我们将明确详细说明：（i）图融合权重以及归一化/对称化处理；（ii）top-k稀疏化策略及其对对齐稳定性的影响；（iii）多尺度APPNP的参数选择和调度门控更新；（iv）原型初始化和更新；（v）EMA参考图和衰减设置。

**实验验证**：我们认可关于评估透明度的担忧。在最终版本中，除了平均分数外，我们还将报告各个指标，包括多次运行的均值±标准差，并添加运行时间和内存的比较，以更好地展示可扩展性。此外，还将提供更多的定性可视化结果，进一步说明空间边界保留情况和对齐质量。

我们相信这些说明和补充内容将显著提升论文的质量，再次感谢评审专家提出的宝贵建议。