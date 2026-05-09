---
title: "I-AND vs MEX_2026牛客寒假算法基础集训营1"
source: "https://ac.nowcoder.com/acm/contest/120561/I"
author:
published:
created: 2026-02-03
description: "牛客竞赛是专业的编程比赛和算法训练平台，包括ACM校赛、ICPC、CCPC、信息学奥赛、NOIP、NOI等编程比赛提高训练营。适合初级小白编程入门训练，包含NOIP普及组提高组赛前集训、ACM区域赛前多校训练营。"
tags:
  - "clippings"
---
## 题目描述$\hspace{15pt}$小苯有一个可重整数集合 $\Bbb{S}$，初始为空。现在他可以进行任意次以下操作，给 $\Bbb{S}$ 中加入一些元素，具体的：  
$\hspace{15pt}\bullet$ 他可以从区间 $[l, r]$ 中选择若干个（至少一个）**不同**的整数，将这些整数的 $\rm AND$（按位与）加入集合 $\Bbb{S}$。  
$\hspace{15pt}$他可以做任意次上述操作，请问 $\Bbb{S}$ 的 $\rm MEX$ 最大可以达到多少。  
  
【名词解释】  
$\hspace{15pt}$AND：指位运算中的按位与（Bitwise AND），对两个整数的二进制表示按位进行与运算。如果您需要更多位运算相关的知识，可以参考 [**OI-Wiki的相关章节**](https://oi-wiki.org/math/bit/#%E4%B8%8E%E6%88%96%E5%BC%82%E6%88%96)。  
$\hspace{15pt}$MEX：整数集合的 $\operatorname{MEX}$ 定义为没有出现在集合中的最小非负整数。例如，$\operatorname{MEX}(1,2,3) =0$、$\operatorname{MEX}(0,2,5) =1$。

## 输入描述:

$\hspace{15pt}$每个测试文件均包含多组测试数据。第一行输入一个整数 $T\left(1\leqq T\leqq 10^5\right)$ 代表数据组数，每组测试数据描述如下：  
  
$\hspace{15pt}$在一行上输入两个整数 $l, r\left(0 \leqq l \leqq r < 2^{30}\right)$，表示区间。

## 输出描述:

$\hspace{15pt}$对于每一组测试数据，新起一行输出一个整数，表示最大的 $\rm MEX$。

示例1

## 输入

[复制](https://ac.nowcoder.com/acm/contest/120561/) 

3
3 4
1 2
0 10

## 输出

[复制](https://ac.nowcoder.com/acm/contest/120561/) 

1
3
11

## 说明

$\hspace{15pt}$对于第一组测试数据，我们只能选择 $3$ 和 $4$ 将他们的 $\rm AND$（即 $0$）加入 $\Bbb{S}$，此时 $\Bbb{S}$ 的 $\rm MEX=1$。