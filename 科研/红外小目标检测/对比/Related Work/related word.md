### paper1

### A. Single-frame Infrared Small Target Detection

SIRST detection has been extensively investigated for decades. The traditional paradigm achieves SIRST detection by measuring the discontinuity between targets and backgrounds. Typical methods include filtering-based methods [6], [9], local contrast measure based methods [10]–[15], and low-rank based methods [7], [8], [16]–[19].

Considering real scenes are much more complex with dramatic changes in target size, shape, and clutter background, it is difficult to use handcrafted features and fixed hyper-parameters to handle such variations. To address this problem, recent CNN-based methods learn trainable features in a data-driven manner. Thanks to the large quantity of data and the powerful model fitting capability of CNNs, these methods achieve better performance than traditional ones.

Existing CNN-based methods can be divided into detection based methods and segmentation based methods. Liu et al. [20] first introduced a generic target detection framework for infrared small target detection. They designed a multi-layer perception (MLP) network with 5 layers for infrared small target detection. Then, McIntosh et al. [21] fine-tuned several generic target detection network (e.g., Faster-RCNN [22] and Yolo-v3 [23]) and used the optimized eigen-vectors as input to achieve improved performance.

Recently, segmentation-based methods have attracted increasing attention. That is because these methods can produce both pixel-level classification and localization outputs. Dai et al. [24] proposed the first segmentation-based network (i.e., ACM). They designed an asymmetric contextual module to aggregate features from shallow layers and deep layers. Then, Dai et al. [31] further improved their ACM by introducing a dilated local contrast measure. Specifically, a feature cyclic shift scheme was designed to achieve a trainable local contrast measure. Moreover, Wang et al. [32] decomposed the infrared target detection problem into two opposed sub-problems (i.e., miss detection and false alarm) and used a conditional generative adversarial network (CGAN) to achieve the trade-off between miss detection and false alarm for infrared small target detection.

Although the performance is continuously improved by recent networks, the loss of small targets in deep layers still remains. This problem ultimately results in the poor robustness to dramatic scene changes (e.g., clutter background, targets with different SCR, shape, and size).

### B. Datasets for SIRST Detection

Existing open-source dataset in infrared small target detection is scarce, most traditional methods are evaluated on their in-house datasets. Only a few infrared small target datasets are released by CNN-based methods [24], [32].

Wang et al. [32] built the first big and open SIRST dataset. This dataset includes 10000 training images and 100 test images. However, many targets in this dataset do not meet the definition of society of photo-optical instrumentation engineers (SPIE) [33] and have obvious synthesized traces with illogical annotations. These problems may lead to the inapplicability toward SIRST detection.

Dai et al. [24] built the first real SIRST dataset with high-quality images and labels. However, the number of images in NUAA-SIRST is 427 (256 for training), which cannot well cover dramatic scene changes in infrared small target detection. Moreover, these real infrared data are all manually labelled with many inaccurately labeled pixels.

Although these open-sourced datasets greatly prompt the prosperity of SIRST detection, their limited data capacity, data variety, and poor annotation hinder the further development of this field. Synthesized data can be easily generated to achieve higher variety and annotation quality at very low cost (i.e., time and money). Hence, we developed a new **NUDT-SIRST dataset** with numerous categories of target, various target sizes, diverse clutter backgrounds, and accurate annotations. The superiority of our dataset is evaluated in Section V.