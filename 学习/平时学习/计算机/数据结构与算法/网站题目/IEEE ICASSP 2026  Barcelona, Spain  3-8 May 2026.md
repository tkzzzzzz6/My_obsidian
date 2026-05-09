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
Submit your reviewer response by writing text only in the box at the bottom of the page.

One response should be written, addressing as many reviews as you choose. You may identify each review by the 4-character code in the red boxes.

Paper Information

### Paper Number

7252

### Paper Title

TOWARD ROBUST SPATIAL MULTI-OMICS INTEGRATION: SPATIAL MULTI-OMICS INTEGRATION VIA ALIGNMENT AND SCHEDULED DIFFUSION

### Authors

Ke Tan, Sichuan Agricultural University  
Guihua Yu, Sichuan Agricultural University  
Tingting Li, Sichuan Agricultural University  
Haoran Chen, Sichuan Agricultural University  
JianChao Liu, Sichuan Agricultural University  
Mengzhu Wang, Hebei University of Technology  
Hau-sing So, University of Macau

Reviewers

Reviewer 4069

Is the work within the scope of the conference and relevant to ICASSP 2026? →

Clearly within scope

Is the manuscript technically correct? →

Some minor concerns that should be easily corrected without altering the contribution or conclusions

Is the technical contribution novel? →

Substantial novelty, with clearly identifiable new methods/concepts

Is the level of experimental validation sufficient? →

Limited but convincing

Is the technical contribution significant? →

Substantial contribution, with a clear potential for impact

Are the references appropriate, without any significant omissions? →

Complete list of references without any significant omissions

Are there any references that do not appear to be relevant? →

All references are directly relevant to the contribution of the manuscript

Is the manuscript properly structured and clearly written? →

Some minor structural, language, or other issues of exposition that would be easily rectified

Comments to the Author(s)

Summary:
This paper introduces TRSMI, a spatial multi-omics integration framework combining a near-identity encoder, temperature-controlled cross-graph alignment, and multi-scale diffusion. The problem is relevant and the overall design is coherent. Experimental results are promising, but several aspects require clearer explanation and stronger empirical support.
Strengths:
(1) Addresses an important and timely topic in spatial multi-omics.
(2) Provides a complete pipeline with encouraging initial performance.
Major Issues to Address:
1. Several components resemble existing methods, and the paper should explicitly clarify what is new:
    (1) RPR near-identity encoder resembles residual/identity-preserving GCN variants (e.g., CED, GCNII). Clarify the unique contribution of RPR.
    (2) Temperature-controlled cross-graph alignment is similar to soft attention alignment in SpatialGlue and related cross-modal matching. Explain how this differs and why it is needed.
    (3) Multi-scale APPNP diffusion + scheduled gate is close to existing multi-scale propagation schemes. Further justify what is novel in the scheduled global-to-local transition.
2. The following details are critical for reproducibility but insufficiently described:
    (1) Graph fusion: selection and normalization of w\_S,w\_F; whether the fused adjacency is symmetrized or renormalized.
    (2) Top-k sparsification: how k is chosen and whether sparsification affects alignment stability.
    (3) Diffusion parameters: rationale for multi-scale APPNP settings and the update rule for the scheduled gate β.
    (4) Prototype-based clustering: initialization of prototypes and update procedures.
    (5) EMA regularization: definition of the EMA reference graph and decay settings.
3. Add standard deviations or confidence intervals to support reported improvements. Include runtime and memory comparisons to demonstrate scalability. Provide more qualitative visualizations to show spatial boundary preservation and alignment quality.

Reviewer 1DFB

Is the work within the scope of the conference and relevant to ICASSP 2026? →

Clearly within scope

Is the manuscript technically correct? →

Some minor concerns that should be easily corrected without altering the contribution or conclusions

Is the technical contribution novel? →

Substantial novelty, with clearly identifiable new methods/concepts

Is the level of experimental validation sufficient? →

Lacking in some respect

Is the technical contribution significant? →

Insufficient contribution for a full-length regular paper, but suitable for short paper

Are the references appropriate, without any significant omissions? →

A largely complete list of references with only minor omissions that would not affect the novelty of the submission

Are there any references that do not appear to be relevant? →

Some of the references are of limited relevance

Is the manuscript properly structured and clearly written? →

Some minor structural, language, or other issues of exposition that would be easily rectified

Comments to the Author(s)

1)
This manuscript is within the scope and relevant to ICASSP as it introduces new techniques for spatial multi-omics.

2)3)
The technical contents are new and appear to move the field in a stronger direction with their updated architecture over previous approaches along with the use of diffusion. 

4)
The metrics reported are averages of metrics without showing the actual metrics. Although these present similar information in some cases this is misleading especially as the actual metrics have not been presented anywhere. There is also no justification for presenting these averages, as the reference 23 presents individual metrics.

Additionally, it would be good to present the mean +- std. of these results over multiple runs to properly show how much better the proposed method is over others by removing chance, unless the models weight initialisation is not random, although this is not stated.

6)7)
Reference 23’s placement is misleading as it implies these averages are used in that work. All other references are good and help support the paper.

8)
The manuscript is mostly well written. However, there is cases of abbreviations never being expanded or being introduced after their first use. There is also excessive use of the emdash, bolding and use of italics.

Overall, it is recommended that proper transparent validation is completed to properly support all claims of SOTA performance.

Reviewer 60DD

Is the work within the scope of the conference and relevant to ICASSP 2026? →

Clearly within scope

Is the manuscript technically correct? →

Some minor concerns that should be easily corrected without altering the contribution or conclusions

Is the technical contribution novel? →

Substantial novelty, with clearly identifiable new methods/concepts

Is the level of experimental validation sufficient? →

Limited but convincing

Is the technical contribution significant? →

Moderate contribution, with the possibility of an impact on the field

Are the references appropriate, without any significant omissions? →

A largely complete list of references with only minor omissions that would not affect the novelty of the submission

Are there any references that do not appear to be relevant? →

All references are directly relevant to the contribution of the manuscript

Is the manuscript properly structured and clearly written? →

Some minor structural, language, or other issues of exposition that would be easily rectified

Comments to the Author(s)

Summary: The paper proposes TRSMI, a graph-signal-processing–inspired framework for integrating spatial multi-omics data (RNA, ATAC, ADT). It models each modality as a graph signal on spatial spots and combines near-identity propagation, soft cross-graph alignment, and multi-scale APPNP with scheduled global-to-local gating. Experiments on three public benchmarks show improved performance on clustering, boundary-sensitive, and classification metrics, supported by ablations and a small hyperparameter study.
Strengths:
(1)	Addresses a timely and important problem of spatial multi-omics integration with explicit attention to spatial boundaries and microenvironments.
(2)  Presents a coherent GSP-based design (near-identity encoder, alignment, scheduled multi-scale diffusion) tailored to denoising while preserving boundaries.
(3) Demonstrates strong empirical results on three benchmarks, with ablations showing complementary roles of the main modules and hyperparameter sweeps indicating reasonable robustness.

Weaknesses
(1) It is not clearly explained what is truly new compared to existing graph-based or multi-omics methods that use similar components.
(2) The paper does not clearly show comparisons with the latest and strongest baselines, so the “state-of-the-art” claim is not fully convincing.

Committee Comments

##### Provide your Response here

Provite your text-only response here. You may identify each review by the 4-character code in the red boxes.

0 / 400 words