---
title: "D-Sequence Coloring_2026牛客寒假算法基础集训营1"
source: "https://ac.nowcoder.com/acm/contest/120561/D"
author:
published:
created: 2026-02-03
description: "牛客竞赛是专业的编程比赛和算法训练平台，包括ACM校赛、ICPC、CCPC、信息学奥赛、NOIP、NOI等编程比赛提高训练营。适合初级小白编程入门训练，包含NOIP普及组提高组赛前集训、ACM区域赛前多校训练营。"
tags:
  - "clippings"
---
## 题目描述$\hspace{15pt}$给定一个长度为 $n$ 的数字序列 $a_1, a_2, \dots, a_n$，每一个元素都有一个颜色，且初始时要么为白色，要么为黑色，使用其值的大小表示其颜色：若 $a_i>0$，则第 $i$ 个元素为白色；否则为黑色。  
  
$\hspace{15pt}$小苯可以在第 $0$ 秒时选择其中至多 $k$ 个**白色**元素染红，随后，从第 $1$ 秒开始，每秒都会发生如下事件：  

$\hspace{23pt}\bullet\,$对于第 $i$ 个元素 $a_i$，如果它是**红色**的，那么将其右侧 $a_i$ 个元素中的**白色**元素也染红。

$\hspace{30pt}$形式化地，对于所有红色元素所在的位置 $i\left(1 \leqq i \leqq n\right)$，将第 $i+1$ 到第 $\min\left(n,i+a_i\right)$ 个元素中的**白色**元素也染红；

$\hspace{23pt}\bullet\,$所有的红色元素同步进行这一操作（即每一秒开始时已经是红色的元素，会在该秒内尝试染红其它元素）；  
$\hspace{23pt}\bullet\,$**黑色元素并不会、也无需被染红**。  
  
$\hspace{15pt}$请你帮助小苯以最优策略染色最初的数字，使得所有的白色元素都被染红的时间尽可能短（可以为 $0$ 秒）。求出这个最短时间，或报告无论如何都无法将所有的白色元素都染红。  

## 输入描述:

$\hspace{15pt}$每个测试文件均包含多组测试数据。第一行输入一个整数 $T\left(1 \leqq T \leqq 10^5\right)$ 代表数据组数，每组测试数据描述如下：  
  
$\hspace{15pt}$第一行两个整数 $n,k\left(1 \leqq k \leqq n \leqq 5 \times 10^5\right)$，表示数字序列的长度、最开始可以染红的元素个数。  
$\hspace{15pt}$第二行 $n$ 个整数 $a_1, a_2, \dots, a_n\left(0 \leqq a_i \leqq n\right)$，表示序列中的元素。其中 $a_i>0$ 表示白色，$a_i=0$ 表示黑色。  
  
$\hspace{15pt}$除此之外，保证单个测试文件的 $n$ 之和不超过 $5 \times 10^5$。  

## 输出描述:

$\hspace{15pt}$对于每一组测试数据，新起一行。如果可以将所有的白色元素都染红，输出一个整数表示最短的全染红时间；否则输出 $-1$。  

示例1

## 输入

[复制](https://ac.nowcoder.com/acm/contest/120561/) 

3
6 2
2 0 1 1 0 1
5 1
1 1 1 1 1
5 1
1 0 1 0 1

## 输出

[复制](https://ac.nowcoder.com/acm/contest/120561/) 

2
4
-1

## 说明

$\hspace{15pt}$对于第一组测试数据，我们一开始选择染红：$i=1$ 和 $i=6$ 这两个位置的元素即可，即 $\{{\color{red}2},0,1,1,0,{\color{red}1}\}$，随后：  
$\hspace{23pt}\bullet\,$第一秒后，序列变为 $\{{\color{red}2},0,{\color{red}1},1,0,{\color{red}1}\}$；  
$\hspace{23pt}\bullet\,$第二秒后，序列变为 $\{{\color{red}2},0,{\color{red}1},{\color{red}1},0,{\color{red}1}\}$。  
$\hspace{15pt}$此时，序列中所有的白色元素均已被染红，耗时 $2$ 秒，可以证明不存在更优的方案，因此输出 $2$。