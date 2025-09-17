### 1) Synopsis of the paper

The paper introduces **TRSMI**, a framework for robust spatial multi-omics integration. The method combines a near-identity projection–residual–propagation encoder, temperature-controlled soft cross-graph alignment with warm-up detach, and multi-scale APPNP diffusion with scheduled global-to-local gating. Modality-specific embeddings are fused through a lightweight MLP, with graph decoders ensuring reconstruction and a prototype-aware contrastive loss compacting clusters. Experiments on three benchmarks (Human Lymph Node, Mouse Brain, Simulation) show state-of-the-art performance in boundary-sensitive and information-theoretic metrics, supported by ablation and parameter analyses.

---

### 2) Summary of Review

The manuscript addresses the problem of integrating spatially resolved multi-omics data, proposing a scalable and interpretable model. Strengths include its innovative combination of alignment and scheduled diffusion (see §2.3; Eq. (4)–(8)), thorough empirical validation (Table 1; Fig. 2), and clear ablation analysis (Table 2). Weaknesses include limited dataset diversity (only three benchmarks; §3.1), partial reproducibility (no code release noted; “No direct evidence found in the manuscript”), and lack of statistical testing on reported improvements (Table 1). The writing is clear and well-structured, though some claims about scalability lack detailed empirical support (e.g., runtime/memory results mentioned in Abstract but not quantified in Experiments).

---

### 3) Strengths

- **Novel architectural design**: The combination of near-identity RPR encoder, warm-up controlled alignment, and multi-scale APPNP with scheduled gating is conceptually well-motivated (see §2.2–§2.3; Eq. (3)–(8)).
    
- **Balanced global/local modeling**: Scheduled gating ensures both topology preservation and boundary sharpening (Eq. (8); §2.3).
    
- **Clear ablation study**: Contributions of individual components are isolated and quantified (Table 2, §3.3.1).
    
- **Strong empirical performance**: Consistent improvements across datasets and metrics (Table 1, §3.3).
    
- **Qualitative validation**: Visual comparisons show improved boundary delineation (Fig. 2, §3.3).
    
- **Parameter robustness**: Performance remains stable under hyperparameter variations (Fig. 3, §3.3.2).
    

---

### 4) Weaknesses

- **Dataset scope**: Evaluation is limited to three datasets, with only one synthetic case; broader tissue/platform diversity is not covered (§3.1).
    
- **Reproducibility**: No mention of released code, seeds, or preprocessing scripts; key for replication (“No direct evidence found in the manuscript”).
    
- **Statistical significance**: Improvements in Table 1 are not accompanied by variance estimates or significance testing (§3.3).
    
- **Scalability claims**: While Abstract mentions favorable runtime/memory, no explicit runtime or complexity benchmarks are reported in Experiments (§3).
    
- **Baselines**: Comparisons omit some recent deep generative multi-omics integration models (e.g., TotalVI, MultiVI) despite being introduced in Introduction (§1).
    
- **Interpretability**: While prototype-aware clustering is described (Eq. (12)), no case studies are shown on biological interpretability or downstream analysis (§2.4; §3).
    

---

### 5) Suggestions for Improvement

1. Expand experimental evaluation to additional spatial multi-omics datasets (e.g., other tissues or higher-resolution Stereo-seq data) to strengthen generalizability.
    
2. Release implementation code, including preprocessing, hyperparameter configurations, and random seeds, to enhance reproducibility.
    
3. Report variance (mean ± std) over multiple runs and provide statistical tests (e.g., paired t-test) for Table 1 to confirm significance of improvements.
    
4. Include explicit runtime and memory usage comparisons against baselines to substantiate claims of scalability.
    
5. Broaden baseline set to include additional recent generative or multimodal integration methods introduced in Introduction (e.g., TotalVI, MultiVI).
    
6. Provide biological interpretability analyses (e.g., recovered cell types or marker genes) to demonstrate downstream value beyond clustering accuracy.
    
7. Clarify the role and initialization of prototypes (Eq. (12)), and add visualizations of learned prototypes or embeddings.
    
8. Discuss potential failure modes, e.g., when modalities are partially missing or heavily imbalanced, and validate robustness experimentally.