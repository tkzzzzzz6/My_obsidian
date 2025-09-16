## ABSTRACT
Spatially resolved multi-omics promises systems-level insight into cellular state, regulation, and communication, yet robustly integrating heterogeneous modalities while preserving spatial boundaries remains challenging. We present a lightweight framework that couples a \emph{near-identity} projection–residual–propagation encoder for stable, boundary-preserving representation learning with \emph{temperature-controlled} soft cross-graph alignment using a short warm-up detach to prevent early misalignment. To reconcile global topology with local refinement, we employ \emph{multi-scale} APPNP diffusion and a \emph{time-scheduled} global-to-local gate that progressively shifts emphasis from global coherence to boundary sharpening. A small MLP fuses modality-specific embeddings; graph decoders enforce modality faithfulness; a prototype-aware contrastive objective compacts clusters; and a mild EMA regularizer stabilizes learned feature graphs. The propagation cost is $O(E d)$ and we sparsify correspondences to mitigate $O(N^2)$ alignment overhead. Across three public benchmarks spanning distinct tissues and platforms, the proposed method achieves state-of-the-art performance on boundary-sensitive and information-theoretic clustering metrics, while maintaining favorable runtime and memory profiles. Ablations confirm the complementary roles of near-identity encoding, warm-up alignment, and scheduled multi-scale diffusion. These results highlight a simple, scalable path toward accurate and interpretable spatial multi-omics integration.
## Introduction
Spatial transcriptomics (ST) extends single-cell RNA sequencing (scRNA-seq) into the spatial domain by capturing transcripts \emph{in situ} while preserving the two-dimensional coordinates of their originating cells or spots~\cite{marx2021method,stahl2016visualization,rao2021exploring,moses2022museum}. In contrast to dissociation-based scRNA-seq, this preservation of spatial context enables direct mapping of gene-expression patterns within intact tissue architecture, typically at or near single-cell resolution (platform dependent)~\cite{asp2020spatially,wang2023advances}. By jointly retaining molecular readouts and spatial localization, ST reveals how heterogeneous cell populations are organized across anatomical boundaries and gradients, and how local microenvironments modulate cellular states through short-range cell--cell interactions. These spatially grounded measurements provide a principled basis for downstream analyses such as spatial domain delineation, cell-type mapping, and the inference of intercellular communication, laying the foundation for the integrative spatial multi-omics methods developed in this work.

Spatially resolved \emph{multi}-omics integrates transcript, chromatin, epigenetic, and protein layers within the same tissue to yield systems-level views of cellular state, regulation, and communication. These modalities are now measurable \emph{in situ} via scRNA-seq, ATAC-seq, CUT\&Tag, and multimodal platforms such as CITE-seq and Stereo-CITE-seq~\cite{zhu2021joint,Stoeckius2017large-scale,liao2023integrated,zhao2025Stereo-seqv2}. Integration, however, is challenging: modalities differ in dimensionality (e.g., ADT vs.\ RNA), carry modality-specific noise and chemistry-driven batch effects~\cite{Stuart2019comprehensive,Hao2021integrated}, and—in spatial settings—must align to anatomical coordinates while preserving local neighborhoods~\cite{marx2021method}. Moreover, single-cell and spatial datasets are sparse, heterogeneous, and non-i.i.d., unlike the large homogeneous corpora typical of NLP/CV~\cite{Lopez2018deep,Argelaguet2021computational}, limiting direct strategy transfer. These factors motivate methods for robust cross-modal alignment that jointly respect spatial geometry and modality-specific idiosyncrasies.

Despite rapid progress, integrative analysis of spatially resolved multi-omics remains nascent. Existing approaches typically fall into two categories: (i) methods that integrate multiple modalities but \emph{do not explicitly couple} them to spatial coordinates—such as MOFA+~\cite{Argelaguet2020mofa+}, TotalVI~\cite{gayoso2019totalvi}, MultiVI~\cite{Ashuach2023multivi}, CiteFuse~\cite{kim2020citefuse}, and PAST~\cite{Li2023past}—and (ii) spatial methods that model neighborhood structure but remain largely \emph{single-modality}, including STAGATE~\cite{Dong2022stagate} and DeepST~\cite{xu2022deepst}. More recently, a new wave of algorithms explicitly couples spatial and molecular information; for example, SpatialGlue~\cite{Long2024SpatialGlue} leverages graph neural networks with dual attention for modality-aware alignment, and PRAGA~\cite{huang2024pragaprototypeawaregraphadaptive} employs dynamic graph construction with probabilistic denoising for robust cross-modal representation learning. These advances underscore the promise of spatial multi-omics integration while highlighting the need for more generalizable frameworks that can accommodate heterogeneous noise and dimensional scales, transfer across platforms, and remain robust to complex tissue architectures.

Nevertheless, current approaches exhibit recurring limitations. Spatial structure and cross-modal coupling are often modeled separately or with fixed single-scale diffusion, leading to over-smoothing and loss of boundaries. Cross-modal alignment tends to rely on static anchors or dense similarities without stabilization (e.g., temperature control or warm-up), making it brittle under noise, modality imbalance, and chemistry-specific artifacts. Graph construction is typically fixed ($k$, metric), susceptible to hubness, and rarely adapted across tissues. Many models enforce a single shared latent space without explicit reconstruction, weakening modality faithfulness and interpretability. Finally, scalability ($O(N^2)$ correspondences) and robustness to batch/platform shifts, resolution mismatches, and partially observed modalities remain open challenges.
![[Pasted image 20250916180935.png]]

To address these challenges, we present \textbf{TRSMI}, a simple and scalable framework for robust spatial multi-omics integration. TRSMI combines (i) a \emph{near-identity} projection–residual–propagation encoder (zero-initialized enhancement with row-wise $\ell_2$ normalization) for stable, boundary-preserving representations; (ii) \emph{temperature-controlled} soft cross-graph alignment with a short warm-up detach to prevent early misalignment; and (iii) \emph{multi-scale} APPNP diffusion with a \emph{time-scheduled} global-to-local gate that first enforces global coherence and then sharpens local structure. A lightweight MLP fuses modality-specific embeddings, graph decoders maintain modality faithfulness, a prototype-aware contrastive objective compacts clusters, and a mild EMA regularizer stabilizes learned feature graphs. Collectively, these choices yield stable alignment that respects spatial geometry and cross-modal coupling, multi-scale diffusion that mitigates over-smoothing with $O(E d)$ propagation while sparsifying correspondences to limit $O(N^2)$ cost, and an interpretable, modality-faithful fusion via joint reconstruction and clustering. Across three public benchmarks, TRSMI achieves state-of-the-art performance on boundary-sensitive and information-theoretic metrics.
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
\end{equation}$$
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

We report averaged metrics: \emph{Info-Avg} = (MI, NMI, AMI)/3, \emph{Clust-Avg} = (FMI, ARI, V-Measure)/3, 
and \emph{Class-Avg} = (F1, Jaccard, Completeness)/3, with the overall score defined as \emph{Overall-Avg} = (Info-Avg + Clust-Avg + Class-Avg)/3. All metrics follow prior work~\cite{huang2024pragaprototypeawaregraphadaptive}.

\subsection{Baseline Methods and Setup}
We compare our method with several representative methods, including MOFA+\cite{Argelaguet2020mofa+}, CiteFuse\cite{kim2020citefuse}, SpatialGlue\cite{Long2024SpatialGlue}, STAGATE\cite{Dong2022stagate}, and PRAGA\cite{huang2024pragaprototypeawaregraphadaptive}.

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





