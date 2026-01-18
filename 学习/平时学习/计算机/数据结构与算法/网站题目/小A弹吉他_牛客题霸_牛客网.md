---
title: "小A弹吉他_牛客题霸_牛客网"
source: "https://www.nowcoder.com/practice/1ea8f581cc1a4db9b098faf82a7b0850?channelPut=tracker2"
author:
published:
created: 2025-11-22
description: "小 A 不想成为不被需要的人，所以她决定要努力练习吉他。但是她现在被一个作业题卡着了，为了。题目来自【牛客题霸】"
tags:
  - "算法与数据结构"
---
> 小 A 不想成为不被需要的人，所以她决定要努力练习吉他。但是她现在被一个作业题卡着了，为了珍惜练习时间，她现在向你求助。  

$\hspace{15pt}$对于给定的正整数 $m$，小 A 需要选一个正整数 $n$，并构造一个长度为 $n$ 的正整数序列 $\{a_1,a_2,\dots,a_n\}$ 使得它们的和等于 $m$ 。  
$\hspace{15pt}$小 A 想知道，对于所有可能的构造，令 $t_i$ 表示 $a_1,a_2,\dots,a_n$ 中等于 $i$ 的元素个数，最大的 $\operatorname{mex}\big\{t_1,t_2,\dots,t_{114^{514}}\big\}$ 是多少。请你求出这个值。  
  
$\hspace{15pt}$**本题有多组数据，请你对每组数据的 $m$ 都求出相应的结果。**  
  
$\hspace{15pt}$整数序列的 $\operatorname{mex}$ 定义为没有出现在序列中的最小非负整数。例如 $\operatorname{mex} \{1,2,3 \} =0$ 、$\operatorname{mex} \{0,2,5\} =1$ 。

### 输入描述：

$\hspace{15pt}$每个测试文件均包含多组测试数据。第一行输入一个整数 $T \left(1 \le T \le 10^5\right)$ 代表数据组数，每组测试数据描述如下：  
  
$\hspace{15pt}$在一行上输入一个正整数 $m \left(1 \le m \le 10^{18}\right)$ 代表规定的数组和。

### 输出描述：

$\hspace{15pt}$对于每一组测试数据，在单独的一行上输出一个整数，表示最大的 $\operatorname{mex}$ 。

## 示例1

输入：3
1
5
12

输出：2
3
4

说明：$\hspace{15pt}$对于第一组数据，其唯一可能的构造是选取 $n=1$ 且 $a=\{1\}$，这样 $t_1=1;\ t_2=t_3=\cdots=t_{114^{514}}=0$，因此 $\operatorname{mex}\big\{t_1,t_2,\dots,t_{114^{514}}\big\} =2$。  
$\hspace{15pt}$对于第二组数据，其中一种可能的构造是选取 $n=3$ 且 $a=\{1,2,2\}$，这样 $t_1=1;\ t_2=2;\ t_3=t_4=\cdots=t_{114^{514}}=0$，因此 $\operatorname{mex}\big\{t_1,t_2,\dots,t_{114^{514}}\big\} =3$。可以证明这是更优的。