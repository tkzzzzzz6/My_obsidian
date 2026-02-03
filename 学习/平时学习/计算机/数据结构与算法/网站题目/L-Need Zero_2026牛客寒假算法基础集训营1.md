---
title: "L-Need Zero_2026牛客寒假算法基础集训营1"
source: "https://ac.nowcoder.com/acm/contest/120561/L"
author:
published:
created: 2026-02-03
description: "牛客竞赛是专业的编程比赛和算法训练平台，包括ACM校赛、ICPC、CCPC、信息学奥赛、NOIP、NOI等编程比赛提高训练营。适合初级小白编程入门训练，包含NOIP普及组提高组赛前集训、ACM区域赛前多校训练营。"
tags:
  - "clippings"
---
## 题目描述$\hspace{15pt}$小苯拿到了一个正整数 $n$，现在他希望 $n$ 的个位数是 $0$，为此他**必须**执行下述操作恰好一次：  
$\hspace{23pt}\bullet\,$选择一个正整数 $x\left(1 \leqq x \leqq 10^5\right)$，并执行 $n:=n\times x$（其中 $:=$ 表示赋值操作）。  
$\hspace{15pt}$你的任务就是帮助小苯找出**最小**的 $x$。我们可以证明，一定存在合法的答案。

## 输入描述:

$\hspace{15pt}$输入一个正整数 $n\left(1 \leqq n \leqq 10^5\right)$，表示小苯拿到的数字。

## 输出描述:

$\hspace{15pt}$输出一个正整数，表示最小的合法解 $x$（可以证明在题目的限定范围内一定有解）。

示例1

## 输入

[复制](https://ac.nowcoder.com/acm/contest/120561/) 

125

## 输出

[复制](https://ac.nowcoder.com/acm/contest/120561/) 

2

## 说明

$\hspace{15pt}$在这个样例中，对于 $n=125$，我们只需要选择 $x=2$，就可以将 $n$ 变为 $125\times 2=250$，满足其个位数为 $0$。显然 $2$ 是最小的正整数解。

示例2

## 输入

[复制](https://ac.nowcoder.com/acm/contest/120561/) 

10

## 输出

[复制](https://ac.nowcoder.com/acm/contest/120561/) 

1

## 说明

$\hspace{15pt}$在这个样例中，对于 $n=10$，我们选择 $x=1$ 即可，操作后 $n$ 不变，满足条件。显然 $1$ 是最小的正整数解。