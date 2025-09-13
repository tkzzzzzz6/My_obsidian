第一段：这个任务的意义，比如空间多组学相较于单细胞等任务的优势。 第二段：这个问题的描述，常见的解决方案，等。这一段开始谭课就有点冗余了

# Introduction

帮我撰写我们论文的introduction部分呢,总体分为5段

**宏观背景引入 (第1段)** → **研究方向聚焦 (第2段)** → **阐述普遍挑战 (第3段)** → **评述现有方法及局限性 (第4段)** → **凝练待解决的核心问题 (第5段)**

具体来讲

- 第一段:这个任务的意义
- 第二段:问题的描述，常见的解决方案等。
- 第三段:讲常见的方法以及各个方法的优势
- 第四段:应该是讲当前方法的问题和局限性
- 第五段就是讲创新,明确总结出我们要解决的根本性的、悬而未决的挑战,定义了本文的研究动机和目标
Medical image segmentation serves as a cornerstone of computer-aided diagnosis, enabling precise delineation of organs, lesions, and vascular structures to support disease screening, clinical diagnosis, and treatment planning. In ophthalmology, the segmentation of the optic disc (OD) and optic cup (OC) from retinal fundus images is particularly critical for glaucoma detection and monitoring, as the cup-to-disc ratio (CDR) derived from these structures is a key biomarker for assessing glaucoma risk. In recent years, deep learning has achieved remarkable breakthroughs in medical image segmentation, with performance on multiple public benchmarks approaching or even surpassing that of human experts. Nevertheless, when deployed in real-world clinical environments, these models still encounter severe challenges due to highly variable and complex imaging conditions.
医学图像分割在计算机辅助诊断中具有核心地位，能够精确刻画器官、病灶和血管等解剖结构，为疾病筛查、临床诊断和治疗规划提供关键支持。在眼科，眼底图像的视盘（Optic Disc, OD）与视杯（Optic Cup, OC）分割是青光眼检测和随访中的重要环节，其计算的杯盘比（CDR）是青光眼风险评估的重要指标。近年来，深度学习方法在医学图像分割领域取得突破性进展，多个公开数据集上的性能已接近甚至超越人类专家。然而，当这些模型应用于临床真实环境时，仍面临复杂多变条件带来的严重挑战。

---
A major challenge in deploying medical image segmentation models across clinical centers is the issue of \textbf{domain shift}: models trained on labeled data from one site (source domain) often fail to maintain stable performance on data from another site (target domain). Such degradation primarily arises from variations in imaging devices, acquisition protocols, and patient populations. To alleviate domain shift, two major research directions have been explored, namely \textbf{domain adaptation (DA)} and \textbf{domain generalization (DG)}. DA methods typically rely on joint training with both source and target data~\cite{tzeng2017adda,ganin2016dann,guan2021survey}, while DG approaches aim to enhance model generalizability through multi-source training or data augmentation strategies~\cite{li2018dg,motiian2017dg,zhou2022dg}. However, both paradigms generally require access to source data during training, which is often impractical in medical scenarios due to privacy concerns and restrictions on cross-institutional data sharing~\cite{perone2019uda,liu2021medsurvey}.
一个突出的难题是**域偏移（domain shift）**：当模型在某一中心的有标注数据（源域）上训练后，往往难以在另一中心（目标域）的数据上保持稳定性能。这种性能退化主要源于不同设备厂商、成像协议和人群差异等带来的数据分布差别。为缓解域偏移，研究者提出了多种解决思路，包括领域适应（Domain Adaptation, DA）和领域泛化（Domain Generalization, DG）。DA 通常依赖源域与目标域的联合训练，DG 则通过多源数据增强模型的泛化性。然而，这两类方法通常要求在训练阶段访问源域数据，在医学场景中受到数据隐私、跨中心共享受限等现实约束，难以广泛应用。

---
In recent years, \textbf{test-time adaptation (TTA)} has emerged as an effective strategy to mitigate domain shift. TTA relies solely on unlabeled target-domain samples during inference to perform lightweight model updates, enabling cross-domain adaptation without access to source data. A representative method is \textbf{TENT}, which minimizes prediction entropy by updating batch normalization (BN) affine parameters, offering a simple yet efficient solution~\cite{wang2021tent}. Subsequent works have extended BN-driven strategies, such as \textbf{DUA}~\cite{mirza2022dua} and \textbf{MedBN}~\cite{park2024medbn}, which enhance the dynamic adaptability and robustness of BN statistics. Beyond BN optimization, more flexible mechanisms have been introduced. For example, \textbf{DIGA} employs dynamic instance-guided adaptation to avoid costly backpropagation while remaining effective for dense prediction tasks~\cite{wang2023diga}; \textbf{DLTTA} leverages adaptive learning rates to accelerate convergence and improve stability~\cite{yang2022dltta}; and \textbf{DomainAdaptor} provides a modular framework for versatile adaptation across tasks~\cite{zhang2023domainadaptor}. In addition, stabilization-oriented methods~\cite{niu2023stabletta} and regularization-driven approaches such as \textbf{DeY-Net}~\cite{wen2023denoisingtta} have demonstrated robustness in dynamic environments, while pixel-level contrastive learning has further improved the discriminability of feature representations. Overall, these approaches contribute complementary strengths in efficiency, flexibility, and robustness, laying a solid foundation for applying TTA to medical image segmentation.

近年来，**测试时自适应（Test-Time Adaptation, TTA）** 逐渐成为缓解域偏移的有效策略。TTA 在推理阶段仅依赖未标注的目标域数据，对模型进行轻量更新，无需访问源域即可实现跨域适配。典型方法如 **TENT** 通过最小化预测熵更新 BN 层参数，具有简洁高效的优势；后续工作如 **DUA** 与 **MedBN** 则进一步提升了 BN 统计量的动态适配性与稳健性。除 BN 驱动的优化外，研究者还提出了更灵活的机制，如 **DIGA** 的动态实例引导、**DLTTA** 的自适应学习率调节，以及 **DomainAdaptor** 的模块化框架，这些方法在保证适配性能的同时提升了效率与灵活性。此外，稳定化方法和结合去噪正则化的 **DeY-Net** 在动态环境下展现了鲁棒性，而像素级对比学习则进一步增强了特征判别性。总体而言，现有方法在效率、适应性和稳健性方面各具优势，为 TTA 在医学图像分割中的应用奠定了坚实基础。

---
尽管已有方法取得了进展，但仍存在显著不足。首先，许多方法严重依赖伪标签，当预测不可靠时容易引入噪声并导致误差传播，尤其在视杯等小结构上表现脆弱。其次，单一的优化目标（如熵最小化或一致性）往往缺乏语义约束，导致特征空间判别性不足。再次，大多数 TTA 方法使用固定学习率，难以在不同目标域条件下保持稳定；即使是先进的方法如 **GraTa**，虽然通过梯度对齐与动态学习率改善了优化稳定性，但其在像素级语义表征方面仍显不足。另一方面，**SPCL**（语义原型对比学习）能够增强特征判别性，但其原始形式依赖源域监督，不完全适用于严格的 source-free TTA 场景。因此，目前仍缺乏一种既能保证优化稳定性，又能提升特征判别性的 TTA 框架。

---


针对上述问题，本文提出了一种融合 **GraTa 与 SPCL 优势**的 框架，用于稳健的 OD/OC 分割。具体而言，
- 我们引入了**动态原型记忆库**：以源域初始化类原型，并在测试时利用高置信度像素进行动量更新，从而实现语义一致的像素级对比学习，提升特征表征的判别性。
- 我们设计了**基于梯度信息的自适应学习率（GIALR）**，通过考察梯度方向一致性与幅度稳定性，动态调节更新步长，在保证优化稳定性的同时避免模型漂移。
- 我们的方法在五个公开眼底图像数据集（RIM-ONE-r3, REFUGE, ORIGA, REFUGE-Validation, Drishti-GS）上进行了广泛验证，结果显示在多种域偏移场景下均显著优于现有最先进方法。
本文的工作为医学图像分割中的 TTA 提供了一种新的解决思路，兼顾了优化稳定性与特征判别性，展现了在临床应用中的巨大潜力。

# Method
背景 → 输入/输出定义 → 数学公式化 → 下游应用价值

## Problem Formulation and Preliminaries

设源域数据集为 $\mathcal{D}_s=\{(x_i^s,y_i^s)\}_{i=1}^{N_s}$，目标域数据集为 $\mathcal{D}_t=\{x_j^t\}_{j=1}^{N_t}$，其中 $x$ 表示眼底图像，$y$ 表示分割标注。分割模型记为 $f_\theta:\mathbb{R}^{H\times W\times 3}\to \mathbb{R}^{K\times H\times W}$，在源域 $\mathcal{D}_s$ 上预训练后部署于目标域 $\mathcal{D}_t$。  
与域自适应 或域泛化 不同，**测试时自适应（TTA）**  在 *source-free* 设定下工作，即推理阶段无法访问源域样本，只能利用目标域的无标注数据。为了满足实时性约束，我们仅允许更新少量参数（如 BN 仿射参数与预测头），并对 BN 统计量在目标域上重新估计。

## 3.2 语义原型引导的像素级对比学习

在密集预测任务中，直接依赖伪标签往往会导致噪声的累积和传播，从而削弱模型的适应能力。为缓解这一问题，我们受到 **语义原型对比学习（SPCL）** 的启发，提出了一种基于 **动态原型记忆库** 的像素级对比学习框架。该方法通过引入类原型作为语义锚点，有效提升了特征空间的结构化与判别性。
在眼底图像 OD/OC 分割中，直接依赖伪标签常在边界区域传播噪声，降低自适应效果。为此，我们受到 **语义原型对比学习（SPCL）** 的启发，设计了一种 **动态原型记忆机制**，通过引入语义原型作为全局锚点，提升特征的判别性与鲁棒性。

## 3.2 语义原型引导的像素级对比学习

在眼底图像 OD/OC 分割中，直接依赖伪标签常在边界区域传播噪声，降低自适应效果。为此，我们受到 **语义原型对比学习（SPCL）** 的启发，设计了一种 **动态原型记忆机制**，通过引入语义原型作为全局锚点，提升特征的判别性与鲁棒性。

首先，利用源域标注数据计算类别特征均值初始化原型 \cite{pei2024multi}，避免冷启动偏移。随后，在目标域自适应过程中，根据预测概率筛选高置信像素集合 $\mathcal{C}$，若某类像素数超过阈值 $\tau_{\text{pix}}$，则以 EMA 更新：

pk←mpk+(1−m)zˉk,∥pk∥2=1,p_k \leftarrow m p_k + (1-m)\bar z_k,\quad \|p_k\|_2=1,

其中 $\bar z_k$ 为该类高置信像素均值，$m$ 为动量系数。

在此基础上，每个像素特征 $z_i$ 与原型计算温度化余弦相似度：

si,k=⟨z^i,p^k⟩τ,s_{i,k}=\frac{\langle \hat z_i,\hat p_k\rangle}{\tau},

并以伪标签 $\hat y_i$ 监督，构建 SPCL 损失：

$$
Lspcl=∣C∣∑i∈C−log⁡exp⁡(si,y^i)∑k=Kexp⁡(si,k).\mathcal{L}_{\text{spcl}}=\tfrac{1}{|\mathcal{C}|}\sum_{i\in \mathcal{C}} -\log \frac{\exp(s_{i,\hat y_i})}{\sum_{k=1}^{K}\exp(s_{i,k})}.
$$

具体而言，我们构建了一个维护一个三类原型库 $M={p_k}_{k=1}^3$（背景、OD、OC)来存储各类别的语义原型特征，并在适配过程中持续更新：

- **原型初始化：** 首先在源域中计算每类特征的均值 $p_k^0$，并进行 $\ell_2$ 归一化，以保证数值稳定性。  
- **动态更新：** 对于每个目标域样本 $x_t$，根据预测置信度筛选出高置信像素集合 $\mathcal{C}$。随后对每一类 $k$ 计算该类高置信像素的均值特征 $\bar z_k$。若该类像素数量 $|\mathcal{C}_k| \ge \tau_{\text{pix}}$，则采用指数滑动平均（EMA）进行更新：
$$
  p_k \leftarrow m p_k + (1-m)\bar z_k,\quad \|p_k\|_2=1,
$$
  其中 $m \in [0,1)$ 为动量系数。  
- **像素级对比学习：** 对于高置信像素特征 $z_i$，计算其与原型的温度化余弦相似度：
  $$
  s_{i,k}=\frac{\langle \hat z_i,\hat p_k\rangle}{\tau}, \quad 
  \hat z_i=\frac{z_i}{\|z_i\|_2},\ \hat p_k=\frac{p_k}{\|p_k\|_2}.
  $$
  在此基础上，定义对比损失函数：
  $$
  \mathcal{L}_{\text{spcl}}=\tfrac{1}{|\mathcal{C}|}\sum_{i\in \mathcal{C}} -\log \frac{\exp(s_{i,\hat y_i})}{\sum_{k=1}^{K}\exp(s_{i,k})}.
  $$

通过这种机制，模型能够在语义空间内拉近同类像素与其对应原型的距离，并将异类特征有效区分开。相比于依赖噪声伪标签的直接监督，该方法在面对预测不确定性时更加稳健，从而显著提升了特征表征的判别性。

## 3.3 基于梯度的自适应学习率（GIALR）
以往 TTA 方法常采用单一目标（如熵最小化 [16]）并使用固定学习率，容易导致更新方向不稳定。受 GraTa 方法 [36] 启发，我们提出 **基于梯度一致性的学习率调度**：

- **辅助梯度：** 原图上的熵损失
$$
\mathcal{L}_{\text{ent}}(\theta;x_t) = - \tfrac{1}{HW}\sum_{u,v}\sum_{c} P_{uv}^{(c)}\log P_{uv}^{(c)}.
$$
- **伪梯度：** 强增强 $\tilde x_t$ 与弱增强 $\{x_t^{(j)}\}$ 的一致性损失
$$
\mathcal{L}_{\text{con}}(\theta';x_t)=\text{CE}\Big(\tfrac{1}{J}\sum_j P_{\theta'}(x_t^{(j)}), P_{\theta'}(\tilde x_t)\Big).
$$
- **梯度对齐：** 计算 $g_p=\nabla_\theta \mathcal{L}_{\text{con}}$ 与 $g_a=\nabla_\theta \mathcal{L}_{\text{ent}}$ 的余弦相似度
$$
c=\frac{\langle g_p,g_a\rangle}{\|g_p\|_2\|g_a\|_2}.
$$
- **自适应学习率：**
$$
\eta=\beta \cdot \tfrac{1}{4}(c+1)^2.
$$

当 $c$ 较大时说明梯度方向一致，应增大学习率加快收敛；当 $c$ 较小时说明冲突严重，则减小学习率以保证稳定性 [32,34]。

## 3.4 总体优化目标与算法
最终目标函数为：
$$
\mathcal{L}_{\text{tot}} = \mathcal{L}_{\text{con}} + \lambda_{\text{spcl}}\mathcal{L}_{\text{spcl}}.
$$
参数更新规则为：
$$
\theta \leftarrow \theta - \eta \nabla_\theta \mathcal{L}_{\text{tot}}, \quad \theta\in\Theta_{\text{adapt}}.
$$

**算法流程（Algorithm 1）：**  
1. 计算 $L_{\text{ent}}$，得到辅助更新 $\theta'$；  
2. 计算 $L_{\text{con}}$，获得伪梯度 $g_p$；  
3. 根据 $c$ 计算自适应学习率 $\eta$；  
4. 以 $\mathcal{L}_{\text{tot}}$ 更新 $\theta$；  
5. 更新原型记忆库。  

该方案在保证优化稳定性的同时，增强了特征的判别性。

## 3.5 实现细节
在实现中，仅更新 BN 仿射参数与预测头，其余参数冻结；BN 统计在目标域重新估计 [16]。  
超参数设定：弱增强视角数 $J=6$，温度系数 $\tau\in[0.05,0.2]$，动量 $m=0.9$，像素阈值 $\tau_{\text{pix}}=30$，权重 $\lambda_{\text{spcl}}=0.2$，缩放因子 $\beta=10^{-4}$。  
与 GraTa 相比，本方法增加了 SPCL 分支，但由于仅更新少量参数，整体计算与显存开销仍然可控，能够满足临床场景下的实时需求。

# Experiments
## Datasets
参考:
We evaluate our proposed GraTa and other state-of-the-art TTA methods on the joint optic disc (OD) and cup (OC) segmentation task, which comprises five public datasets collected from different medical centres, denoted as domain A (RIM-ONE-r3 (Fumero et al. 2011)), B (REFUGE (Orlando et al. 2020)), C (ORIGA (Zhang et al. 2010)), D (REFUGE-Validation/Test (Orlando et al. 2020)), and E (Drishti-GS (Sivaswamy et al. 2014)). These datasets consist of 159, 400, 650, 800, and 101 images, respectively. For each image, we cropped a region of interest (ROI) centered at the OD with a size of 800 × 800, and each ROI is further resized to 512 × 512 and normalized by min-max normalization following (Hu, Liao, and Xia 2022). We utilize the Dice score metric (DSC) for evaluation.
我们在联合视盘（OD）和视杯（OC）分割任务上评估了我们提出的GraTa以及其他最先进的TTA方法，该任务包含来自不同医疗中心的五个公开数据集，分别记为域A（RIM-ONE-r3（Fumero等人，2011））、B（REFUGE（Orlando等人，2020））、C（ORIGA（Zhang等人，2010））、D（REFUGE-Validation/Test（Orlando等人，2020））和E（Drishti-GS（Sivaswamy等人，2014））。这些数据集分别包含159、400、650、800和101张图像。对于每张图像，我们裁剪出一个以视盘为中心、大小为800×800的感兴趣区域（ROI），每个感兴趣区域进一步调整为512×512的大小，并按照（Hu、Liao和Xia，2022）的方法进行最小-最大归一化。我们使用骰子评分指标（DSC）进行评估。
我们论文:
我们选择了联合视盘（OD）和视杯（OC）分割任务相关的来自不同医疗中心的五个公开数据集分别记为域A（RIM-ONE-r3）、B（REFUGE（Orlando等人，2020））、C（ORIGA（Zhang等人，2010））、D（REFUGE-Validation/Test（Orlando等人，2020））和E（Drishti-GS（Sivaswamy等人，2014））,这些数据集分别包含159、400、650、800和101张图像。每张图像都被裁剪出一个以视盘为中心、大小为800×800的感兴趣区域（ROI），每个感兴趣区域进一步调整为512×512的大小，并按照（Hu、Liao和Xia，2022）的方法进行最小-最大归一化。我们使用骰子评分指标（DSC）进行评估.

我们选取了来自不同医疗中心的五个公开眼底图像数据集：域A（RIM-ONE-r3）、域B（REFUGE，Orlando _et al._，2020）、域C（ORIGA，Zhang _et al._，2010）、域D（REFUGE-Validation/Test，Orlando _et al._，2020）以及域E（Drishti-GS，Sivaswamy _et al._，2014）对各个方法在联合视盘（OD）与视杯（OC）分割任务进行了评估，。上述数据集分别包含 159、400、650、800 和 101 张图像。每张图像都被裁剪出一个以视盘为中心、大小为 $800\times 800$ 的感兴趣区域（ROI）并被统一调整为 $512\times 512$，依据 Hu, Liao 和 Xia (2022) 提出的方式进行最小-最大归一化处理。最终，我们采用 **Dice 相似系数（DSC）** 作为性能评价指标。


We evaluate all methods on the joint optic disc (OD) and optic cup (OC) segmentation task using five publicly available fundus image datasets collected from different medical centers:  RIM-ONE-r3,REFUGE \cite{orlando2020refuge}), ORIGA \cite{zhang2010origa}), REFUGE-Validation/Test \cite{orlando2020refuge}, and Drishti-GS \cite{sivaswamy2014drishti}. These datasets contain 159, 400, 650, 800, and 101 images, respectively. For each image, a region of interest (ROI) of size $800\times 800$ centered on the optic disc is first cropped and then resized to $512\times 512$. All images are subsequently normalized using the min–max strategy proposed by Hu, Liao, and Xia \cite{hu2022minmax}. The performance of all methods is quantitatively assessed using the Dice similarity coefficient (DSC).

## Baseline Methods and Setup
我们在每个源域上分别训练了一个xx.xx基线模型，遵循ProSFDA 的数据预处理和增强流程，然后在所有剩余的域（目标域）上进行评估。最终性能报告为所有跨域设置（总共4×5个）的平均值。为了进行公平比较，与Yang等人（2022年）的做法一致,我们对每个测试批次采用批量大小为1的单迭代适应。在部署MASR时，我们使用Adam优化预训练模型的仿射变换参数，同时根据Wang等人（2021年）的方法，从目标测试数据中重新估计批归一化（BN）统计量。缩放因子β根据经验设置为xx.xx。

REFUGE挑战赛的官方训练/测试集与验证集被有意分离开，并将它们视为两个独立的域。



遵循 ProSFDA \cite{hu2022minmax} 的数据预处理与增广流程，我们在五个眼底数据集上进行单源到多目标的跨域评估：对每个源域单独训练 ResUNet-34 基线模型，并在其余 4 个目标域上进行测试时自适应（TTA），最终报告 5×4=20 组结果的平均值。我们与No Adapt以及DUA \cite{mirza2022dua}、DIGA \cite{wang2023diga}、MedBN \cite{park2024medbn}、SAR \cite{niu2023stabletta}、DeTTA \cite{wen2023denoisingtta} 与 GraTa \cite{chen2024grata}等方法进行对比。为公平比较，所有实验均在 NVIDIA 3090 GPU 上进行,按 Yang 等（2022）\cite{yang2022dltta} 对每个测试批次采用 batch size=1 的单迭代适应；部署 MASR 时，仅优化 BN 仿射参数并使用 Adam（lr=1e-4）更新，同时按 Wang 等（2021）\cite{wang2021tent} 在目标域上重新估计 BN 统计量。主伪损失由一致性项与语义原型对比项加权组成，权重 β=1.0。基线方法的超参数遵循其原始论文中报告的默认配置。

Following the data preprocessing and augmentation pipeline of ProSFDA \cite{hu2022minmax}, 
we conduct cross-domain evaluation in a single-source-to-multiple-target setting on five fundus datasets. 
Specifically, a ResUNet-34 baseline model is trained on each source domain individually and adapted at test time to the remaining four target domains, 
yielding $5 \times 4 = 20$ domain adaptation results whose average performance is reported. We compare our method against \textit{No Adapt} and several state-of-the-art TTA approaches, including 
DUA \cite{mirza2022dua}, DIGA \cite{wang2023diga}, MedBN \cite{park2024medbn}, 
SAR \cite{niu2023stabletta}, DeTTA \cite{wen2023denoisingtta}, and GraTa \cite{chen2024grata}. 
For fair comparison, all experiments are conducted on an NVIDIA RTX 3090 GPU. 
Following Yang \textit{et al.} \cite{yang2022dltta}, we adopt a batch size of one and perform a single adaptation step per test batch. 
When deploying MASR, only the BN affine parameters are optimized using Adam with a learning rate of $1\times 10^{-4}$, 
and BN statistics are recalibrated on the target domain as suggested by Wang \textit{et al.} \cite{wang2021tent}. 
The pseudo-supervised objective combines a consistency term and a semantic prototype contrastive term with equal weighting ($\beta=1.0$). 
For all baseline methods, hyperparameters follow the default configurations reported in their original papers.



## Performance Analysis



### Performance Results

表~\ref{tab:DSC}报告了在五个目标领域的定量比较结果。总体而言，MASR始终优于“无适应”基线和所有竞争的TTA方法，取得了75.28的最高平均DSC。总体来看，MASR在平均水平上以+0.91%的优势超过了强大的梯度对齐方法GraTa，在REFUGE-Val/Test域上的增益最为显著（+1.16%）。这些改进证实，我们对语义正则化和梯度幅度控制的整合带来了更稳定且更有效的适应，在领域偏移情况下既提供了鲁棒性，又实现了更优的分割质量。

### Ablation Study

### Parameter Analysis
我现在正在进行我们的参数分析实验,实验的要求是通过对
SPCL权重 (0.5, 1.0, 1.5, 2.0)：语义原型对比损失的权重
原型动量 (0.7, 0.9, 0.95)：原型记忆库的更新动量
GIALR权重组合：方向权重 vs 幅值权重 (0.1,0.9), (0.3,0.7), (0.5,0.5), (0.7,0.3), (0.9,0.1)
三个参数进行灵敏度分析,生成的是各个域对应TTA测试以后对其他域的dsc值的平均值的折线图,一共5个域,REFUGE_Valid
RIM_ONE_r3
REFUGE
ORIGA
和Drishti_GS,生成的可视化图只需要一张

### 2. 伪标签筛选参数 (pseudo_label_sensitivity.py)

测试伪标签筛选的关键参数：

- 绝对阈值 (0.02, 0.05, 0.08, 0.1)：OD/OC的绝对概率阈值

- 分位数阈值 (0.4, 0.6, 0.7, 0.8)：自适应分位数选择




# 提示词

1. 帮我将这部分内容翻译为地道的英文会议计算机论文表达,并使用latex格式提供带上典型文献引用（DOI/ArXiv）呢,文献提供bib格式,还是提供latex格式,使用\cite引用
