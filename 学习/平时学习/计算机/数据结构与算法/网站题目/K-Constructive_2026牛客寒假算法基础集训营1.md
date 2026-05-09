---
title: "K-Constructive_2026牛客寒假算法基础集训营1"
source: "https://ac.nowcoder.com/acm/contest/120561/K"
author:
published:
created: 2026-02-03
description: "牛客竞赛是专业的编程比赛和算法训练平台，包括ACM校赛、ICPC、CCPC、信息学奥赛、NOIP、NOI等编程比赛提高训练营。适合初级小白编程入门训练，包含NOIP普及组提高组赛前集训、ACM区域赛前多校训练营。"
tags:
  - "clippings"
---
## 题目描述$\hspace{15pt}$给定一个正整数 $n$，你需要构造一个长度为 $n$ 的数组 $a_1, a_2, \dots, a_n$，满足以下条件：  
$\hspace{23pt}\bullet\,$数组中的每个元素 $a_i$ 都是正整数；  
$\hspace{23pt}\bullet\,$所有元素的乘积等于所有元素的和；  
$\hspace{23pt}\bullet\,$数组中的元素互不相同。  
$\hspace{15pt}$小苯想知道，对于给定的 $n$，是否存在这样的数组。如果存在，请构造字典序最小的解；如果不存在报告无解。  
  
【名词解释】  
$\hspace{15pt}$数组的字典序比较：从左到右逐个比较两个数组的元素。如果在某个位置上元素不同，比较这两个元素的大小，元素小的数组字典序也小。如果一直比较到其中一个数组结束，则长度较短的数组字典序更小。例如，$\{2,3,4,5\}$ 的字典序小于 $\{2,4,3,5\}$，也小于 $\{3,2,4,5\}$。  

## 输入描述:

$\hspace{15pt}$每个测试文件均包含多组测试数据。第一行输入一个整数 $T\left(1 \leqq T \leqq 100\right)$ 代表数据组数，每组测试数据描述如下：  
  
$\hspace{15pt}$在一行上输入一个整数 $n\left(1 \leqq n \leqq 100\right)$，表示数组的长度。  
  
$\hspace{15pt}$除此之外，保证单个测试文件中所有 $n$ 之和不超过 $10^3$。

## 输出描述:

$\hspace{15pt}$对于每组测试数据，如果不存在满足条件的数组，直接输出 $\texttt{NO}$，否则：  
$\hspace{23pt}\bullet\,$第一行输出 $\texttt{YES}$；  
$\hspace{23pt}\bullet\,$第二行输出 $n$ 个正整数，表示字典序最小的满足条件的数组，元素间用单个空格分隔；

示例1

## 输入

[复制](https://ac.nowcoder.com/acm/contest/120561/) 

2
1
2

## 输出

[复制](https://ac.nowcoder.com/acm/contest/120561/) 

YES
1
NO

## 说明

$\hspace{15pt}$对于第一组测试数据，数组 $\{1\}$ 显然满足条件。  
$\hspace{15pt}$对于第二组测试数据，可以证明找不到满足条件的数组。