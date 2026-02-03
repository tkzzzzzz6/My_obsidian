---
title: "C-Array Covering_2026牛客寒假算法基础集训营1"
source: "https://ac.nowcoder.com/acm/contest/120561/C"
author:
published:
created: 2026-02-03
description: "牛客竞赛是专业的编程比赛和算法训练平台，包括ACM校赛、ICPC、CCPC、信息学奥赛、NOIP、NOI等编程比赛提高训练营。适合初级小白编程入门训练，包含NOIP普及组提高组赛前集训、ACM区域赛前多校训练营。"
tags:
  - "clippings"
---
## 题目描述$\hspace{15pt}$给定长度为 $n$ 的数组 $a_1,a_2,\dots,a_n$，其中第 $i$ 个数的值为 $a_i$。  
  
$\hspace{15pt}$小苯希望数组中所有数字的总和尽可能大，为此他可以做任意次如下操作：  

$\hspace{23pt}\bullet\,$选择一对下标 $l, r\left(1 \leqq l < r \leq n\right)$，接着将 $(l,r)$ 区间（注意是开区间）内的所有数都变为区间端点值的较大者。

$\hspace{30pt}$形式化地，对所有 $j\left(l<j<r\right)$，均执行：$a_j:=\max(a_l,a_r)$（其中 $:=$ 表示赋值操作）。

  
$\hspace{15pt}$你的任务就是求出数组总和的最大值。

## 输入描述:

$\hspace{15pt}$每个测试文件均包含多组测试数据。第一行输入一个整数 $T\left(1 \leqq T \leqq 10^5\right)$ 代表数据组数，每组测试数据描述如下：  
  
$\hspace{15pt}$第一行输入一个正整数 $n\left(1 \leqq n \leqq 5 \times 10^5\right)$，表示数组 $a$ 的长度。  
$\hspace{15pt}$第二行输入 $n$ 个正整数 $a_1,a_2,\dots,a_n\left(1 \leqq a_i \leqq 10^9\right)$，表示最初时所有数字的值。  
  
$\hspace{15pt}$除此之外，保证单个测试文件的 $n$ 之和不超过 $5 \times 10^5$。

## 输出描述:

$\hspace{15pt}$对于每一组测试数据，新起一行输出一个正整数，表示在可以进行任意次操作的情况下，所有数字之和的最大值。

示例1

## 输入

[复制](https://ac.nowcoder.com/acm/contest/120561/) 

2
3
5 1 5
4
1 1 1 1

## 输出

[复制](https://ac.nowcoder.com/acm/contest/120561/) 

15
4

## 说明

$\hspace{15pt}$对于第一组测试数据，可以选择 $l=1,r=3$ 操作一次，操作完后数组为 $\{5,5,5\}$，求和为 $15$。我们可以证明，无法通过其他操作使得总和更大。