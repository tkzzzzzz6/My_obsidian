## ABSTRACT
Spatially resolved multi-omics promises systems-level insight into cellular state, regulation, and communication, yet robustly integrating heterogeneous modalities while preserving spatial boundaries remains challenging. We present a lightweight framework that couples a \emph{near-identity} projection–residual–propagation encoder for stable, boundary-preserving representation learning with \emph{temperature-controlled} soft cross-graph alignment using a short warm-up detach to prevent early misalignment. To reconcile global topology with local refinement, we employ \emph{multi-scale} APPNP diffusion and a \emph{time-scheduled} global-to-local gate that progressively shifts emphasis from global coherence to boundary sharpening. A small MLP fuses modality-specific embeddings; graph decoders enforce modality faithfulness; a prototype-aware contrastive objective compacts clusters; and a mild EMA regularizer stabilizes learned feature graphs. The propagation cost is $O(E d)$ and we sparsify correspondences to mitigate $O(N^2)$ alignment overhead. Across three public benchmarks spanning distinct tissues and platforms, the proposed method achieves state-of-the-art performance on boundary-sensitive and information-theoretic clustering metrics, while maintaining favorable runtime and memory profiles. Ablations confirm the complementary roles of near-identity encoding, warm-up alignment, and scheduled multi-scale diffusion. These results highlight a simple, scalable path toward accurate and interpretable spatial multi-omics integration.
空间分辨多组学有望从系统层面深入了解细胞状态、调控和通讯，但在保留空间边界的同时稳健整合异质模态仍然面临挑战。我们提出了一个轻量级框架，该框架将用于稳定、保边界表示学习的“近似恒等”投影-残差-传播编码器，与采用短时预热分离以防止早期错位的“温度控制”软跨图对齐相结合。为了协调全局拓扑与局部细化，我们采用“多尺度”APPNP扩散和“时间调度”的全局到局部门控，该门控逐渐将重点从全局一致性转向边界锐化。一个小型多层感知器（MLP）融合模态特异性嵌入；图解码器确保模态保真度；原型感知对比目标压缩聚类；适度的指数移动平均（EMA）正则化器稳定学习到的特征图。传播成本为$O(Ed)$，并且我们对对应关系进行稀疏化以减轻$O(N^2)$的对齐开销。在涵盖不同组织和平台的三个公共基准测试中，所提出的方法在边界敏感和信息论聚类指标上实现了最先进的性能，同时保持了良好的运行时间和内存占用。消融实验证实了近似恒等编码、预热对齐和调度多尺度扩散的互补作用。这些结果凸显了一条简单、可扩展的路径，可实现准确且可解释的空间多组学整合。
## Introduction
Spatial transcriptomics (ST) extends single-cell RNA sequencing (scRNA-seq) into the spatial domain by capturing transcripts \emph{in situ} while preserving the two-dimensional coordinates of their originating cells or spots~\cite{marx2021method,stahl2016visualization,rao2021exploring,moses2022museum}. In contrast to dissociation-based scRNA-seq, this preservation of spatial context enables direct mapping of gene-expression patterns within intact tissue architecture, typically at or near single-cell resolution (platform dependent)~\cite{asp2020spatially,wang2023advances}. By jointly retaining molecular readouts and spatial localization, ST reveals how heterogeneous cell populations are organized across anatomical boundaries and gradients, and how local microenvironments modulate cellular states through short-range cell--cell interactions. These spatially grounded measurements provide a principled basis for downstream analyses such as spatial domain delineation, cell-type mapping, and the inference of intercellular communication, laying the foundation for the integrative spatial multi-omics methods developed in this work.
空间转录组学（ST）通过原位捕获转录本，同时保留其来源细胞或斑点的二维坐标，将单细胞RNA测序（scRNA-seq）扩展到了空间领域[marx2021method,stahl2016visualization,rao2021exploring,moses2022museum]。与基于解离的单细胞RNA测序不同，这种空间背景的保留使得能够在完整的组织结构中直接绘制基因表达模式，通常达到或接近单细胞分辨率（取决于平台）[asp2020spatially,wang2023advances]。通过同时保留分子读数和空间定位，空间转录组学揭示了异质细胞群如何跨解剖学边界和梯度组织，以及局部微环境如何通过短距离细胞间相互作用调节细胞状态。这些基于空间的测量为下游分析（如空间域划分、细胞类型映射和细胞间通讯推断）提供了原则性基础，也为这项工作中开发的整合空间多组学方法奠定了基础。

Spatially resolved \emph{multi}-omics integrates transcript, chromatin, epigenetic, and protein layers within the same tissue to yield systems-level views of cellular state, regulation, and communication. These modalities are now measurable \emph{in situ} via scRNA-seq, ATAC-seq, CUT\&Tag, and multimodal platforms such as CITE-seq and Stereo-CITE-seq~\cite{zhu2021joint,Stoeckius2017large-scale,liao2023integrated,zhao2025Stereo-seqv2}. Integration, however, is challenging: modalities differ in dimensionality (e.g., ADT vs.\ RNA), carry modality-specific noise and chemistry-driven batch effects~\cite{Stuart2019comprehensive,Hao2021integrated}, and—in spatial settings—must align to anatomical coordinates while preserving local neighborhoods~\cite{marx2021method}. Moreover, single-cell and spatial datasets are sparse, heterogeneous, and non-i.i.d., unlike the large homogeneous corpora typical of NLP/CV~\cite{Lopez2018deep,Argelaguet2021computational}, limiting direct strategy transfer. These factors motivate methods for robust cross-modal alignment that jointly respect spatial geometry and modality-specific idiosyncrasies.空间分辨的多组学技术整合了同一组织内的转录本、染色质、表观遗传和蛋白质层面的信息，从而获得细胞状态、调控和通讯的系统级视图。如今，这些模态可通过单细胞RNA测序（scRNA-seq）、转座酶可接近染色质测序（ATAC-seq）、CUT&Tag技术以及CITE-seq和Stereo-CITE-seq等多模态平台在原位进行测量[参考文献{zhu2021joint,Stoeckius2017large-scale,liao2023integrated,zhao2025Stereo-seqv2}]。然而，整合工作面临诸多挑战：不同模态在维度上存在差异（例如，抗体衍生标签（ADT）与RNA），存在模态特异性噪声和化学驱动的批次效应[参考文献{Stuart2019comprehensive,Hao2021integrated}]，而且在空间场景中，必须与解剖坐标对齐，同时保留局部邻域结构[参考文献{marx2021method}]。此外，单细胞和空间数据集具有稀疏性、异质性和非独立同分布性，这与自然语言处理（NLP）/计算机视觉（CV）中典型的大型同质语料库不同[参考文献{Lopez2018deep,Argelaguet2021computational}]，这限制了直接的策略迁移。这些因素促使人们研发能够稳健进行跨模态对齐的方法，此类方法需同时兼顾空间几何结构和模态特异性特质。

Despite rapid progress, integrative analysis of spatially resolved multi-omics remains nascent. Existing approaches typically fall into two categories: (i) methods that integrate multiple modalities but \emph{do not explicitly couple} them to spatial coordinates—such as MOFA+~\cite{Argelaguet2020mofa+}, TotalVI~\cite{gayoso2019totalvi}, MultiVI~\cite{Ashuach2023multivi}, CiteFuse~\cite{kim2020citefuse}, and PAST~\cite{Li2023past}—and (ii) spatial methods that model neighborhood structure but remain largely \emph{single-modality}, including STAGATE~\cite{Dong2022stagate} and DeepST~\cite{xu2022deepst}. More recently, a new wave of algorithms explicitly couples spatial and molecular information; for example, SpatialGlue~\cite{Long2024SpatialGlue} leverages graph neural networks with dual attention for modality-aware alignment, and PRAGA~\cite{huang2024pragaprototypeawaregraphadaptive} employs dynamic graph construction with probabilistic denoising for robust cross-modal representation learning. These advances underscore the promise of spatial multi-omics integration while highlighting the need for more generalizable frameworks that can accommodate heterogeneous noise and dimensional scales, transfer across platforms, and remain robust to complex tissue architectures.
尽管取得了快速进展，但空间分辨多组学的整合分析仍处于起步阶段。现有方法通常分为两类：（i）整合多种模态但未明确将其与空间坐标耦合的方法，例如MOFA+~\cite{Argelaguet2020mofa+}、TotalVI~\cite{gayoso2019totalvi}、MultiVI~\cite{Ashuach2023multivi}、CiteFuse~\cite{kim2020citefuse}和PAST~\cite{Li2023past}；（ii）对邻域结构进行建模但主要仍是单模态的空间方法，包括STAGATE~\cite{Dong2022stagate}和DeepST~\cite{xu2022deepst}。最近，一批新的算法明确地将空间信息和分子信息耦合起来；例如，SpatialGlue~\cite{Long2024SpatialGlue}利用具有双重注意力的图神经网络实现模态感知对齐，而PRAGA~\cite{huang2024pragaprototypeawaregraphadaptive}则采用带有概率去噪的动态图构建，以实现稳健的跨模态表示学习。这些进展凸显了空间多组学整合的前景，同时也表明需要更具通用性的框架，以适应异质噪声和不同维度尺度，在不同平台间进行迁移，并对复杂的组织架构保持稳健性。

Nevertheless, current approaches exhibit recurring limitations. Spatial structure and cross-modal coupling are often modeled separately or with fixed single-scale diffusion, leading to over-smoothing and loss of boundaries. Cross-modal alignment tends to rely on static anchors or dense similarities without stabilization (e.g., temperature control or warm-up), making it brittle under noise, modality imbalance, and chemistry-specific artifacts. Graph construction is typically fixed ($k$, metric), susceptible to hubness, and rarely adapted across tissues. Many models enforce a single shared latent space without explicit reconstruction, weakening modality faithfulness and interpretability. Finally, scalability ($O(N^2)$ correspondences) and robustness to batch/platform shifts, resolution mismatches, and partially observed modalities remain open challenges.
然而，当前的方法存在一些反复出现的局限性。空间结构和跨模态耦合通常被分开建模，或者采用固定的单尺度扩散进行建模，这会导致过度平滑和边界丢失。跨模态对齐往往依赖于静态锚点或密集的相似度，而没有进行稳定性处理（例如温度控制或预热），这使其在噪声、模态不平衡和化学特异性伪影的情况下变得脆弱。图的构建通常是固定的（k值、度量标准），容易受到中心节点问题的影响，且很少能跨组织进行调整。许多模型强制使用单一的共享潜在空间，却没有进行显式的重建，这削弱了模态的真实性和可解释性。最后，在可扩展性（O(N²)的对应关系）以及对批次/平台变化、分辨率不匹配和部分观测模态的鲁棒性方面，仍然存在尚未解决的挑战。
![[Pasted image 20250916180935.png]]

To address these challenges, we present \textbf{TRSMI}, a simple and scalable framework for robust spatial multi-omics integration. TRSMI combines (i) a \emph{near-identity} projection–residual–propagation encoder (zero-initialized enhancement with row-wise $\ell_2$ normalization) for stable, boundary-preserving representations; (ii) \emph{temperature-controlled} soft cross-graph alignment with a short warm-up detach to prevent early misalignment; and (iii) \emph{multi-scale} APPNP diffusion with a \emph{time-scheduled} global-to-local gate that first enforces global coherence and then sharpens local structure. A lightweight MLP fuses modality-specific embeddings, graph decoders maintain modality faithfulness, a prototype-aware contrastive objective compacts clusters, and a mild EMA regularizer stabilizes learned feature graphs. Collectively, these choices yield stable alignment that respects spatial geometry and cross-modal coupling, multi-scale diffusion that mitigates over-smoothing with $O(E d)$ propagation while sparsifying correspondences to limit $O(N^2)$ cost, and an interpretable, modality-faithful fusion via joint reconstruction and clustering. Across three public benchmarks, TRSMI achieves state-of-the-art performance on boundary-sensitive and information-theoretic metrics.
为了应对这些挑战，我们提出了**TRSMI**，这是一个简单且可扩展的鲁棒空间多组学整合框架。TRSMI 结合了（i）一个近似恒等投影-残差-传播编码器（具有行式ℓ₂归一化的零初始化增强），用于生成稳定、保留边界的表征；（ii）温度控制的软跨图对齐，带有短暂的预热分离机制，以防止早期错位；（iii）多尺度APPNP 扩散，配备时间调度的全局到局部门控，该门控首先确保全局一致性，然后锐化局部结构。一个轻量级的多层感知器（MLP）融合模态特异性嵌入，图解码器保持模态真实性，一个原型感知的对比目标压缩聚类，以及一个温和的指数移动平均（EMA）正则化器稳定学习到的特征图。总的来说，这些选择实现了稳定的对齐，尊重空间几何结构和跨模态耦合；多尺度扩散通过O(E d)传播减轻过度平滑，同时稀疏化对应关系以限制O(N²)成本；并通过联合重建和聚类实现可解释、忠实于模态的融合。在三个公共基准测试中，TRSMI 在边界敏感指标和信息论指标上均达到了最先进的性能。
## Method

### Problem Formulation and Preliminaries
Spatial multi-modal omics jointly measure molecular profiles and spatial context across $N$ tissue spots~\cite{stahl2016visualization}. 
Let $\mathcal{S}=\{(x_i,y_i)\}_{i=1}^{N}$ denote coordinates and $\mathcal{F}_M=\{\mathbf{f}_i^m\in\mathbb{R}^{D_m}\}_{i=1,m=1}^{N,M}$ denote features from $M$ modalities (e.g., RNA, ATAC, ADT~\cite{wang2009RNA-seq,Buenrostro2015ATAC-seq,Stoeckius2017large-scale}). 
Our goal is to learn a joint embedding $\mathcal{Z}\in\mathbb{R}^{N\times d}$ that preserves cross-modal semantics and spatial structure:
\begin{equation}
\label{eq:joint_embedding}
\mathcal{Z} = \Phi(\mathcal{F}_M,\mathcal{S}).
\end{equation}
where $\Phi$ denotes the integration function modeling both cross-modal dependencies and spatial relationships~\cite{Hao2021integrated}.Such embeddings support downstream tasks including cell-type identification, spatial domain detection, tumor microenvironment (TME) analysis, and cell--cell interaction inference~\cite{Hao2021integrated,liu2024SpatialMultiomicsDeciphering}.
空间多模态组学联合测量了N个组织位点的分子谱和空间背景\cite{stahl2016visualization}。令S = {(xᵢ, yᵢ)}ᵢ₌₁ᴺ表示坐标，Fₘ = {fₘᵢ ∈ ℝᴰₘ}ᵢ₌₁,ₘ₌₁ᴺ,ᴹ表示来自M种模态的特征（例如，RNA、ATAC、ADT\cite{wang2009RNA-seq,Buenrostro2015ATAC-seq,Stoeckius2017large-scale}）。我们的目标是学习一个联合嵌入Z ∈ ℝᴺˣᵈ，以保留跨模态语义和空间结构：
\begin{equation} \label{eq:joint_embedding} \mathcal{Z} = \Phi(\mathcal{F}M,\mathcal{S}). \end{equation}
其中Φ表示同时建模跨模态依赖关系和空间关系的整合函数\cite{Hao2021integrated}。这种嵌入支持下游任务，包括细胞类型识别、空间域检测、肿瘤微环境（TME）分析以及细胞间相互作用推断\cite{Hao2021integrated,liu2024SpatialMultiomicsDeciphering}。
### Graph Construction and RPR Encoders
For each modality $m$, we build a spatial $k$-NN graph $A_S$ on $\mathcal{S}$ and a feature $k$-NN graph $A_F^{m}$ on $F^{m}\!\in\!\mathbb{R}^{N\times D_m}$ (after PCA/LSI/CLR when applicable). Graphs are symmetrized and degree-normalized. We then form a fused adjacency:
$$\begin{equation}
\hat{A}^{m} = w_S^{m} A_S + w_F^{m} A_F^{m},\qquad w_S^{m},w_F^{m}\ge 0.
\end{equation}$$
\textbf{RPR encoder.}Motivated by CED’s conservative near-identity design~\cite{zhang2023CEDNetCascadeEncoderdecoder}, we adopt a one-layer graph encoder with a zero-initialized enhancement branch and a small learnable gate $\alpha$:
$$\begin{equation}
\begin{aligned}
Z^{m} &= \mathrm{GCN}_e^{m}(F^{m},\hat{A}^{m})\\ 
&= \hat{A}^{m}\!\left(F^{m}W_e^{m} + \alpha\,\Delta(F^{m}W_e^{m})\right),\\
&\quad
W_e^{m}\!\in\!\mathbb{R}^{D_m\times d},
\end{aligned}
\end{equation}$$
where $\Delta(\cdot)$ is a zero-initialized linear branch. Row-wise $\ell_2$ normalization is applied before/after propagation to stabilize training and preserve boundaries. The encoder is intentionally shallow to avoid over-smoothing; deeper diffusion is deferred to a global module.
对于每种模态$m$，我们在$S$上构建一个空间$k$-近邻图$A_S$，并在$F_m \in \mathbb{R}^{N \times D_m}$上构建一个特征$k$-近邻图$A_m^F$（在适用的情况下，经过PCA/LSI/CLR处理后）。这些图会进行对称化和度归一化处理。然后，我们构建融合邻接矩阵：$$$\hat{A}_m = w_m^S A_S + w_m^F A_m^F，其中w_m^S, w_m^F \geq 0$$$。$\textbf{RPR编码器}。受CED的保守近恒等设计启发[zhang2023CEDNetCascadeEncoderdecoder]，我们采用了一个单层图编码器，它具有一个零初始化的增强分支和一个小的可学习门控$\alpha$：$Z_m = GCN_m^e(F_m, \hat{A}_m) = \hat{A}_m (F_m W_m^e + \alpha \Delta(F_m W_m^e))$，其中$W_m^e \in \mathbb{R}^{D_m \times d}$，$\Delta(\cdot)$是一个零初始化的线性分支。在传播前后会应用行方向的$\ell_2$归一化，以稳定训练并保留边界。编码器特意设计得较浅，以避免过度平滑；更深层次的扩散则推迟到全局模块中进行。
### Cross-Graph Alignment and Multi-Scale Diffusion
Given two modality-specific encodings $Z^{1},Z^{2}\in\mathbb{R}^{N\times d}$ from Eq.~\eqref{eq:encoder}, we compute temperature-controlled soft correspondences. Let
$$\begin{equation}
\begin{aligned}
& Q = \mathrm{norm}(Z^{1}W_q),\\
& K=\mathrm{norm}(Z^{2}W_k),\\
&S = \mathrm{softmax}\!\big(QK^\top/\tau\big),
\end{aligned}
\end{equation}$$
where $W_q,W_k\in\mathbb{R}^{d\times d_a}$ and the softmax is row-wise. During a short warm-up, cross terms are detached (when updating $Z^1$ we use $Z^2\!\texttt{.detach()}$, and vice versa) to avoid unstable early alignments. We then augment each modality with its aligned counterpart and project back:
$$\begin{equation}
\widetilde{Z}^{1} = \big[\,Z^{1}\,\Vert\, SZ^{2}\,\big]W_{1},\qquad
\widetilde{Z}^{2} = \big[\,Z^{2}\,\Vert\, S^\top Z^{1}\,\big]W_{2},
\end{equation}$$
with $W_{1},W_{2}\in\mathbb{R}^{2d\times d}$. (In practice, $S$ is optionally row-sparsified by top-$k$ to reduce $O(N^2)$ cost.)
给定两个来自公式\eqref{eq:encoder}的特定模态编码Z₁、Z₂∈R^N×d，我们计算温度控制的软对应关系。令Q = norm(Z₁W_q)，K = norm(Z₂W_k)，S = softmax(QK⊤/τ)，其中W_q、W_k∈R^d×dₐ，且softmax按行进行。在一个短暂的预热阶段，交叉项会被分离（更新Z₁时，我们使用Z₂.detach()，反之亦然），以避免早期对齐不稳定。然后，我们用每个模态的对齐对应项对其进行增强并投影回去：Ẑ₁ = [Z₁ ∥ SZ₂]W₁，Ẑ₂ = [Z₂ ∥ S⊤Z₁]W₂，其中W₁、W₂∈R^2d×d。（在实际应用中，S可选择通过top-k进行行稀疏化，以降低O(N²)的成本。）

#### Local refinement with conservative updates.
A lightweight local branch aggregates neighbors (via $\hat{A}^{m}$) and mixes them with a small gate $\alpha$:
$$\begin{equation}
Z_{\mathrm{loc}} = Z_{\mathrm{base}} + \alpha\,Z_{\mathrm{graph}},\quad 0\le \alpha\le 1,
\end{equation}$$
where $Z_{\mathrm{base}}$ stacks $\widetilde{Z}^{1}$ and $\widetilde{Z}^{2}$ after a linear projection, and $Z_{\mathrm{graph}}$ concatenates self and neighbor features followed by a linear layer.

\textbf{Multi-scale APPNP diffusion.}
To preserve global cluster geometry while denoising, we fuse modality graphs to obtain $\widehat{A}_{g}$ (averaging and renormalization) and apply a multi-scale APPNP operator:
$$\begin{equation}
Z_{g} = \mathrm{MS\text{-}APPNP}(\widehat{A}_{g}, Z_{\mathrm{loc}}),
\end{equation}$$
where multiple APPNP propagations with different $(K,\alpha)$ are concatenated and linearly projected back to $d$.

\textbf{Time-scheduled global/local balance.}
We combine local and global features with a learnable gate $\beta$ that is monotonically cooled during training:
$$\begin{equation}
Z_{\ast} = (1-\beta)\,Z_{\mathrm{loc}} + \beta\,Z_{g},\qquad 0\le\beta\le 1.
\end{equation}$$
Early iterations favor global structure (larger $\beta$), and later iterations progressively refine local boundaries as $\beta$ decreases.

\subsection{Fusion, Reconstruction, and Prototype-Aware Clustering}
\label{sec:fuse_loss}

We fuse modality-specific encodings via a small MLP:
$$\begin{equation}
Z = \mathrm{MLP}\big(\mathrm{Concat}(Z^{1},\ldots,Z^{M})\big)\in\mathbb{R}^{N\times d}.
\end{equation}$$
To maintain modality faithfulness and spatial smoothness, we decode $Z$ back to each modality using the spatial graph $A_S$:
\begin{equation}
\label{eq:decoder}
\widehat{F}^{m} = A_S\,Z\,W^{m}_{d},\qquad W^{m}_{d}\in\mathbb{R}^{d\times D_m}.
\end{equation}
The reconstruction loss averages modality-wise MSE (with optional weights $w^{m}$):
$$\begin{equation}
\mathcal{L}_{\mathrm{rec}} = \frac{1}{M}\sum_{m=1}^{M} w^{m}\,\|F^{m}-\widehat{F}^{m}\|_{2}^{2}.
\end{equation}$$

Let $C=\{c_k\}_{k=1}^{K}$ be learnable prototypes (initialized by split-and-merge seeding over $Z$). With temperature $\tau_c$, a prototype-aware contrastive objective compacts clusters while resisting class imbalance:
$$\begin{equation}
\mathcal{L}_{\mathrm{clust}} = \frac{1}{K}\sum_{k=1}^{K}
\left[-\log\frac{\mathbb{E}_{i\in\mathcal{I}_k}\exp(\langle z_i,c_k\rangle/\tau_c)}
{\mathbb{E}_{j}\exp(\langle z_j,c_k\rangle/\tau_c)}\right],
\end{equation}$$
where $z_i$ is the $i$-th row of $Z$ and $\mathcal{I}_{k}$ are indices currently assigned to prototype $k$ (nearest-prototype assignment).

To stabilize training and improve interpretability, we regularize the learned feature graphs toward an exponential-moving-average (EMA) reference:
$$\begin{equation}
\mathcal{L}_{\mathrm{graph}}=\sum_{m=1}^{M}\big\|A_F^{m}-\mathrm{EMA}(A_F^{m})\big\|_{F}^{2}.
\end{equation}$$
The full objective is
$$\begin{equation}
\mathcal{L} = \mathcal{L}_{\mathrm{rec}} + \lambda_{\mathrm{cl}}\,\mathcal{L}_{\mathrm{clust}} + \lambda_{\mathrm{g}}\,\mathcal{L}_{\mathrm{graph}}.
\end{equation}$$一个轻量级的局部分支通过（via ^ A m）聚合邻居节点，并将它们与一个小的门控α混合：$$Z_loc = Z_base + α·Z_graph，0 ≤ α ≤ 1,$$其中Z_base在经过线性投影后堆叠˜Z₁和˜Z₂，而Z_graph在经过线性层后连接自身和邻居特征。

**多尺度APPNP扩散**。为了在去噪的同时保留全局聚类几何结构，我们融合模态图以获得Â_g（通过平均和重新归一化），并应用多尺度APPNP算子：$$Z_g = MS-APPNP(Â_g, Z_loc)，$$其中具有不同（K，α）的多个APPNP传播被连接起来，并线性投影回d维。

**时间调度的全局/局部平衡**。我们使用一个在训练过程中单调冷却的可学习门控β来组合局部和全局特征：$$Z* = (1 − β)·Z_loc + β·Z_g，0 ≤ β ≤ 1。$$早期迭代偏向全局结构（β较大），而随着β的减小，后期迭代逐渐细化局部边界。

\subsection{融合、重建和原型感知聚类} \label{sec:fuse_loss}

我们通过一个小型MLP融合特定模态的编码：$$Z = MLP(Concat(Z₁, …, Z_M)) ∈ ℝ^N×d。$$为了保持模态忠实性和空间平滑性，我们使用空间图A_S将Z解码回每个模态：
$$\begin{equation}  \widehat{F}^{m} = A_S,Z,W^{m}{d},\qquad W^{m}{d}\in\mathbb{R}^{d\times D_m}. \end{equation}$$
重建损失对模态-wise的MSE进行平均（带有可选权重wₘ）：$$L_rec = (1/M)·∑ₘ₌₁ᴹ wₘ·∥Fₘ −  ˆFₘ∥²₂。$$
令$C = {cₖ}ₖ₌₁ᴷ$为可学习的原型（通过对Z进行分割和合并种子初始化）。在温度$τ_c$下，一个原型感知的对比目标在压缩聚类的同时抵抗类别不平衡：$$L_{clust} = (1/K)·∑ₖ₌₁ᴷ[−log(E_{i∈Iₖ} exp(⟨zᵢ, cₖ⟩/τ_c) / Eⱼ exp(⟨zⱼ, cₖ⟩/τ_c))]，$$其中zᵢ是Z的第i行，Iₖ是当前分配给原型k的索引（最近原型分配）。

为了稳定训练并提高可解释性，我们将学习到的特征图正则化为指数移动平均（EMA）参考：$$L_{graph} = ∑ₘ₌₁ᴹ ∥AₘF − EMA(AₘF)∥²_F。$$完整的目标函数是
$$\begin{equation} \mathcal{L} = \mathcal{L}_{\mathrm{rec}} + \lambda_{\mathrm{cl}}\,\mathcal{L}_{\mathrm{clust}} + \lambda_{\mathrm{g}}\,\mathcal{L}_{\mathrm{graph}}. \end{equation}$$
## Experiments
![[Pasted image 20250916183515.png]]
  
![[Pasted image 20250916183534.png]]
PRAGA vs. TRSMI on MB (spatial maps and UMAPs). TRSMI shows clearer boundaries and compact clusters.
![[Pasted image 20250916183539.png]]![[Pasted image 20250916183547.png]]

 


We evaluate on three publicly available datasets spanning real tissues and a controlled simulation.
\textbf{Human Lymph Node (HLN).}
The HLN dataset~\cite{Long2024SpatialGlue} provides paired RNA, ADT, and spatial coordinates for \mbox{3,484} spots with expert cell-type annotations.
\textbf{Mouse Brain (MB).}
The MB dataset~\cite{zhang2023spatial} integrates ATAC–RNA and CUT\&Tag–RNA profiles from postnatal day 22 mouse brain, totaling \mbox{9,196} spatial spots. In the absence of ground-truth labels, we assess performance via cross-modal clustering consistency between RNA and ATAC.
\textbf{Simulation (Sim).}
The synthetic benchmark~\cite{Long2024SpatialGlue} contains simulated RNA, ATAC, and ADT modalities with \mbox{1,296} spatially annotated spots, enabling controlled evaluation of multimodal integration under known ground truth.
我们在三个公开可用的数据集上进行了评估，这些数据集涵盖了真实组织和受控模拟。\textbf{人类淋巴结（HLN）}：HLN数据集[Long2024SpatialGlue]提供了3484个点的配对RNA、ADT和空间坐标，以及专家标注的细胞类型。\textbf{小鼠大脑（MB）}：MB数据集[zhang2023spatial]整合了出生后22天小鼠大脑的ATAC-RNA和CUT&Tag-RNA图谱，共包含9196个空间点。在没有真实标签的情况下，我们通过RNA和ATAC之间的跨模态聚类一致性来评估性能。\textbf{模拟数据（Sim）}：合成基准数据集[Long2024SpatialGlue]包含模拟的RNA、ATAC和ADT模态，带有1296个空间注释点，能够在已知真实情况的条件下对多模态整合进行受控评估。

We report averaged metrics: \emph{Info-Avg} = (MI, NMI, AMI)/3, \emph{Clust-Avg} = (FMI, ARI, V-Measure)/3, 
and \emph{Class-Avg} = (F1, Jaccard, Completeness)/3, with the overall score defined as \emph{Overall-Avg} = (Info-Avg + Clust-Avg + Class-Avg)/3. All metrics follow prior work~\cite{huang2024pragaprototypeawaregraphadaptive}.
我们报告了平均指标：\emph{信息平均值} =（互信息（MI）、标准化互信息（NMI）、调整后的互信息（AMI））/3，\emph{聚类平均值} =（Fowlkes-Mallows指数（FMI）、调整后的兰德指数（ARI）、V测度）/3，以及\emph{分类平均值} =（F1分数、雅卡尔指数、完整性）/3，总体得分定义为\emph{总体平均值} =（信息平均值 + 聚类平均值 + 分类平均值）/3。所有指标均遵循先前的研究成果~\cite{huang2024pragaprototypeawaregraphadaptive}。

## Baseline Methods and Setup
We compare our method with several representative methods, including MOFA+\cite{Argelaguet2020mofa+}, CiteFuse\cite{kim2020citefuse}, SpatialGlue\cite{Long2024SpatialGlue}, STAGATE\cite{Dong2022stagate}, and PRAGA\cite{huang2024pragaprototypeawaregraphadaptive}.
我们将我们的方法与几种具有代表性的方法进行了比较，包括MOFA+\cite{Argelaguet2020mofa+}、CiteFuse\cite{kim2020citefuse}、SpatialGlue\cite{Long2024SpatialGlue}、STAGATE\cite{Dong2022stagate}和PRAGA\cite{huang2024pragaprototypeawaregraphadaptive}。

All models are trained on an NVIDIA RTX~3090 GPU with full-batch SGD, using learning rates of 0.01 for dual-modality and 0.001 for tri-modality tasks. Baseline methods follow their official default hyperparameters whenever available. RNA features are preprocessed with Scanpy (gene filtering, HVG selection, normalization, scaling), while ADT features are CLR-normalized. Graphs are constructed with neighborhood sizes $k=10/20$, and multi-scale diffusion is applied with $(K,\alpha)=\{(2,0.20),(4,0.15),(8,0.10)\}$. We employ an adaptively learned global gate (no fixed $\beta_0$) and a composite loss with hyperparameters $\alpha=0.9$, $w_{\text{cl}}=1$, $w_{\scriptstyle RNA}=5$, $w_{\scriptstyle ADT}=5$, and temperature $\tau=2$. 
% Exact replication may be non-trivial, as several baseline implementations are not fully open-sourced and preprocessing pipelines vary across datasets.

\subsection{Performance Analysis}

% As shown in Table~\ref{tab:all_avg}, TRSMI consistently outperforms baselines, with the largest gain on \textbf{MB} (+6.40 Overall-Avg) and stable improvements on \textbf{HLN} (+0.23) and \textbf{Simulation} (+0.52). 
% In addition, Fig.~\ref{fig:visualization_mb} qualitatively confirms these gains, with TRSMI producing clearer clusters and sharper boundaries than PRAGA.

As shown in Table~\ref{tab:all_avg}, \textbf{TRSMI} achieves the best Overall-Avg on all datasets, with the largest margin on \textbf{MB} (+6.40) and stable gains on \textbf{HLN} (+0.23) and \textbf{Sim} (+0.52). The MB result—evaluated via RNA--ATAC consistency—indicates superior cross-modal reconciliation while preserving spatial topology; the HLN gain suggests robustness to modality imbalance (low-dimensional ADT); and on Sim, TRSMI better recovers ground-truth spatial domains. 

Across metrics, TRSMI improves both information-theoretic scores (MI/NMI/AMI) and boundary-sensitive measures (F1/Jaccard), reflecting enhanced global structure and sharper local delineation. Qualitative maps in Fig.~\ref{fig:visualization_mb} corroborate these trends, showing clearer clusters and cleaner boundaries than PRAGA—attributable to near-identity RPR encoding, temperature-controlled warm-up alignment, and multi-scale APPNP with scheduled gating.


\subsubsection{Ablation Study}

% condensed version
% On MB, we ablate RPR, cross-graph alignment, multi-scale diffusion, and scheduled gating (Table~\ref{tab:ablation}). Each component yields incremental gains, while the full model achieves the best overall performance, confirming their complementary effects.

We conduct an ablation study on the MB dataset to evaluate the contributions of different components in our TRSMI framework (Table~\ref{tab:ablation}). The examined modules include RPR (row-propagation encoder), Align (cross-graph alignment), MS-APPNP (multi-scale diffusion), and Sched-$\beta$ (scheduled gating).While enabling a single component yields noticeable improvements over the baseline, the gains remain limited. In contrast, combining all modules delivers the best overall performance, confirming that global diffusion, cross-modal alignment, and scheduled gating complement each other in enhancing structure-preserving and boundary-sensitive metrics.

\begin{table}[h]
\centering
\small
\caption{Ablation on MB. \ding{52} enabled, \ding{56} disabled.}
\label{tab:ablation}
% Avg. denotes Overall-Avg with equal weights across Info-/Clust-/Class-Avg.
 % RPR: row-propagation encoder; Align: cross-graph alignment; MS-APPNP: multi-scale diffusion; Sched-$\beta$: scheduled global gate.
\setlength{\tabcolsep}{3pt}
\begin{tabular}{l|cccc|c}
\toprule
model & RPR & Align & MS-APPNP & Sched-$\beta$ & Avg. \\
\midrule
Baseline    & \ding{56} & \ding{56} & \ding{56} & \ding{56} & 44.11 \\
& \ding{52} & \ding{56} & \ding{56} & \ding{56} & 45.46 \\
& \ding{56} & \ding{52} & \ding{56} & \ding{56} & 49.94 \\
& \ding{56} & \ding{56} & \ding{52} & \ding{56} & 48.92 \\
& \ding{56} & \ding{56} & \ding{56} & \ding{52} & 48.70 \\
\midrule
TRSMI (Ours)        & \ding{52} & \ding{52} & \ding{52} & \ding{52} & 50.51 \\
\bottomrule
\end{tabular}
\end{table}

\subsubsection{Parameter Analysis}
We analyze three hyperparameters on HLN: $\alpha_1$ (diffusion at $K{=}2$), $\alpha_2$ (diffusion at $K{=}4$), and $\beta_0$ (initial global gate). We sweep $\alpha_1\in[0.10,0.20]$, $\alpha_2\in[0.05,0.10]$, and $\beta_0\in[1.5,2.5]$ with 10 evenly spaced points each, keeping other settings fixed. As shown in Fig.~\ref{fig:param_analysis} (top-left: $\alpha_1$, top-right: $\alpha_2$, bottom-left: $\beta_0$), performance is generally stable across these ranges; the best Overall appears at moderate diffusion strengths and mid-range $\beta_0$.

\begin{figure}[h]
  \centering
  \includegraphics[width=0.85\linewidth]{LaTex/figure/parameter_analysis.png}
  \caption{Parameter analysis on HLN. Top-left/Top-right/Bottom-left: Overall vs. $\alpha_1$ ($K{=}2$), $\alpha_2$ ($K{=}4$), and $\beta_0$, respectively.}
  \label{fig:param_analysis}
\end{figure}





