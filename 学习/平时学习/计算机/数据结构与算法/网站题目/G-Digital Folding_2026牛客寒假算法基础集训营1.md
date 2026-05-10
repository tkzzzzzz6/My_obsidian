---
title: "G-Digital Folding_2026牛客寒假算法基础集训营1"
source: "https://ac.nowcoder.com/acm/contest/120561/G"
author:
published:
created: 2026-02-03
description: "牛客竞赛是专业的编程比赛和算法训练平台，包括ACM校赛、ICPC、CCPC、信息学奥赛、NOIP、NOI等编程比赛提高训练营。适合初级小白编程入门训练，包含NOIP普及组提高组赛前集训、ACM区域赛前多校训练营。"
tags:
  - "clippings"
---
## 题目描述$\hspace{15pt}$小苯发现了一种特殊的数字运算，称为"数字折叠"。对于一个正整数 $x$，定义其 "折叠数" 为：  

$\hspace{23pt}\bullet\,$将 $x$ 的十进制数位翻转并去除前导 $0$，$x$ 的值更改为翻转后得到的新数。

$\hspace{30pt}$例如，$123$ 操作后会变为 $321$，而 $120$ 会变为 $21$。

$\hspace{15pt}$现在小苯拿到了一个区间 $[l, r]$，他想知道如果将区间中所有的整数 $i\left(l \leqq i \leqq r \right)$ 的折叠数都求出，那么其中的最大值是多少。你的任务就是求出这个最大值。

## 输入描述:

$\hspace{15pt}$每个测试文件包含多组测试数据。第一行输入一个整数 $T\left(1 \leqq T \leqq 10^4\right)$ 代表数据组数，每组测试数据描述如下：  
  
$\hspace{15pt}$在一行上输入两个整数 $l, r\left(1 \leqq l \leqq r \leqq 10^{15}\right)$，表示询问的区间。

## 输出描述:

$\hspace{15pt}$对于每一组测试数据，新起一行输出一个整数，表示区间中所有数的 "折叠数" 的最大值。

示例1

## 输入

[复制](https://ac.nowcoder.com/acm/contest/120561/) 

3
1 20
1000 10000
1 999

## 输出

[复制](https://ac.nowcoder.com/acm/contest/120561/) 

91
9999
999

## 说明

$\hspace{15pt}$对于第一组测试数据，折叠数最大的数字是 $19$，其折叠数是 $91$。