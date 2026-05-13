---
title: "H-Blackboard_2026牛客寒假算法基础集训营1"
source: "https://ac.nowcoder.com/acm/contest/120561/H"
author:
published:
created: 2026-02-03
description: "牛客竞赛是专业的编程比赛和算法训练平台，包括ACM校赛、ICPC、CCPC、信息学奥赛、NOIP、NOI等编程比赛提高训练营。适合初级小白编程入门训练，包含NOIP普及组提高组赛前集训、ACM区域赛前多校训练营。"
tags:
  - "clippings"
---
## 题目描述$\hspace{15pt}$小红在黑板上写了一个计算式，具体来讲是 $n$ 个数字 $a_1,a_2,\dots,a_n$，其中间由 $n-1$ 个加号（$\texttt{`+'}$）连接组成：$a_1+a_2+\cdots+a_n$。  
$\hspace{15pt}$现在小苯想去擦去黑板上的一些 $\texttt{`+'}$ 运算符，但他擦得很不干净，只擦去了加号中的横线（$\texttt{`-'}$），剩下的部分就是一个竖线（$\texttt{`|'}$）了。巧合的是，$|$ 恰好也是一个运算符：按位或 $\rm or$。  
$\hspace{15pt}$小苯想知道，有多少种不同的擦黑板方式，能使得按照新算式进行计算，结果和擦黑板前的算式计算结果相同，请你帮他算一算。注意，不擦黑板也是一种方案。由于答案可能很大，请将答案对 $998\,244\,353$ 取模后输出。  
  
【注】  
$\hspace{15pt}$特别的，**在本题中我们认为 $\rm or$ 运算符的优先级大于加法运算 $\texttt{`+'}$**。  
$\hspace{15pt}$两种擦黑板方式不同当且仅当存在至少一个运算符位置，其在其中一个方式中为 $\texttt{`+'}$，而在另一个方式中为 $\texttt{`|'}$（即按位或 $\rm or$）。  
  
【名词解释】  
$\hspace{15pt}$$\operatorname{or}$：指位运算中的按位或（Bitwise OR），对两个整数的二进制表示按位进行或运算。如果您需要更多位运算相关的知识，可以参考 [**OI-Wiki的相关章节**](https://oi-wiki.org/math/bit/#%E4%B8%8E%E6%88%96%E5%BC%82%E6%88%96)。  

## 输入描述:

$\hspace{15pt}$每个测试文件均包含多组测试数据。第一行输入一个整数 $T\left(1 \leqq T \leqq 100\right)$ 代表数据组数，每组测试数据描述如下：  
  
$\hspace{15pt}$第一行输入一个整数 $n\left(1 \leqq n \leqq 2\times 10^5\right)$，表示黑板上的数字个数。  
$\hspace{15pt}$第二行输入 $n$ 个整数 $a_1, a_2, \dots, a_n\left(0 \leqq a_i \lt 2^{31}\right)$，表示黑板上的数字。  
  
$\hspace{15pt}$除此之外，保证单个测试文件的 $n$ 之和不超过 $2 \times 10^5$。  

## 输出描述:

$\hspace{15pt}$对于每一组测试数据，新起一行输出一个整数，表示不同的擦黑板方案个数对 $998\,244\,353$ 取模后的答案。  

示例1

## 输入

[复制](https://ac.nowcoder.com/acm/contest/120561/) 

3
2
1 2
3
1 2 3
4
1 2 0 4

## 输出

[复制](https://ac.nowcoder.com/acm/contest/120561/) 

2
2
8

## 说明

$\hspace{15pt}$对于第一组测试数据，对于 $1 \texttt{+} 2$ 这个计算式：  
$\hspace{23pt}\bullet\,$我们可以选择擦中间的 $\texttt{`+'}$，就能得到：$1\ \rm{|}\ 2=3$，原式运算结果也为 $3$，因此是一种方案；  
$\hspace{23pt}\bullet\,$不擦黑板显然是一种方案。  
$\hspace{15pt}$综上，共有两种方案。