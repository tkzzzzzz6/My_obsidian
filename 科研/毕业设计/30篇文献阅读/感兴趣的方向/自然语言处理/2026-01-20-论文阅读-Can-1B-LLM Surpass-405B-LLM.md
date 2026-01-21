---
title: 2026-01-20-论文阅读-Can-1B-LLM Surpass-405B-LLM?
date: 2026-01-20
tags:
  - 论文阅读
  - TTS(推理侧缩放)
  - LLM
---
# 《Can 1B LLM Surpass 405B LLM? Rethinking Compute-Optimal Test-Time Scaling》

## 一、论文基本信息
- [原文链接](https://www.alphaxiv.org/abs/2502.06703?chatId=019be074-e9d0-7eed-bdf2-01fe34e3ea54),[翻译链接](https://hjfy.top/arxiv/2502.06703)
- 作者:Runze Liu1,2,∗, Junqi Gao1,3, Jian Zhao4, Kaiyan Zhang2, Xiu Li2, Biqing Qi1,†, Wanli Ouyang1 and
Bowen Zhou1,2,†

> 关键词:Test-Time Scaling (TTS),过程奖励模型(PRM),小语言模型(SLMs) 。  

## 二、研究背景与问题定义

### 研究背景
近年来，大语言模型（LLMs）在复杂推理任务上取得了显著进步。随着 OpenAI o1 等模型的出现，研究界开始关注如何通过增加推理侧的计算投入来突破模型原有性能的上限，这一方向被称为测试时缩放（Test-Time Scaling, TTS）。目前的 TTS 主要分为两大路径：一是通过强化学习训练模型生成长思维链（Internal TTS）；二是在预训练模型保持不变的情况下，利用外部搜索和验证机制（External TTS）来寻找最优解答。

然而，现有的外部 TTS 研究仍面临诸多挑战。首先，虽然增加计算量能提升性能，但如何根据不同的问题难度、策略模型（Policy Model）和验证模型（PRM）来分配有限的算力，实现“计算最优（Compute-Optimal）”仍缺乏系统性分析。其次，当前的验证器（PRM）在面对非其训练分布的模型输出时，往往会出现“分布外（OOD）”失效问题，导致推理过程被错误的奖励信号误导。最后，业界对于小型模型（如 1B 或 3B 规模）在极端计算缩放下的潜力上限尚不明确。

### 问题定义
1. 策略优化问题：在面对不同的策略模型、过程奖励模型（PRM）以及不同难度的任务时，如何选择并配置最优的 TTS 路径（如 Best-of-N、Beam Search 或 DVTS）以实现效率与性能的最佳平衡？
2. 性能边界问题：通过极端的测试时计算缩放，小型语言模型在处理复杂数学和竞赛级任务（如 MATH-500, AIME24）时，性能提升的极限在哪里？它们是否真的能够通过算力补偿，在逻辑推理能力上跨越参数规模的鸿沟，甚至超越 405B 等级的巨型模型或现有的顶尖推理模型（如 o1-preview）？

## 三、核心方法与设计


## 四、实验


## 五、创新点、贡献与改进空间


## 六、我的思考


## 七、其他
### 可跟进的文献  


