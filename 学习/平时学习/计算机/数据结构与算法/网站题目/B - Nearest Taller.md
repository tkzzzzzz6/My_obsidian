---
title: "B - Nearest Taller"
source: "https://atcoder.jp/contests/abc433/tasks/abc433_b"
author:
  - "[[AtCoder Inc.]]"
published: 2025-11-22
created: 2025-11-22
description: "AtCoder is a programming contest site for anyone from beginners to experts. We hold weekly programming contests online."
tags:
  - "算法与数据结构"
---
B - Nearest Taller  
B - 最接近的更高者

---

Score: $200$ points 得分： $200$ 分

### Problem Statement 问题陈述

There are $N$ people standing in a row from left to right. The $i$ -th person from the left $(1\le i\le N)$ is called person $i$. The height of person $i$ $(1\le i\le N)$ is $A_i$.  
从左到右排成一排有 $N$ 个人。从左边数第 $i$ 个人 $(1\le i\le N)$ 被称为 $i$ 。第 $i$ 个人 $(1\le i\le N)$ 的身高是 $A_i$ 。

For each $i=1,2,\ldots,N$, determine whether there exists a person to the left of person $i$ who is taller than person $i$, and if so, find the person standing closest to person $i$ among them.  
对于每个 $i=1,2,\ldots,N$ ，确定在 $i$ 左边是否有人比 $i$ 更高，如果有，找出这些人中离 $i$ 最近的人。

### Constraints 约束

- $1\le N\le 100$
- $1\le A_i\le 100$
- All input values are integers.  
	所有输入值都是整数。

---

### Input 输入

The input is given from Standard Input in the following format:  
输入来自标准输入，格式如下：

```
NN
A1A_1 A2A_2 …\ldots ANA_N
```

### Output 输出

Output $N$ lines.输出 $N$ 行。

The $i$ -th line $(1\le i\le N)$ should contain $-1$ if there is no person to the left of person $i$ who is taller than person $i$, and otherwise, the number representing the person standing closest to person $i$ among such people.  
如果 $i$ 左边没有人比 $i$ 高，则第 $i$ 行 $(1\le i\le N)$ 应包含 $-1$ ；否则，应包含代表这些人中离 $i$ 最近的人的编号。

---

### Sample Input 1 示例输入 1 份

```
4
4 3 2 5
```

### Sample Output 1 示例输出 1 份

```
-1
1
2
-1
```
- There is no person to the left of person $1$. Thus, output $-1$ on the first line.  
	人 $1$ 左边没有人。因此，第一行输出 $-1$ 。
- Among the people to the left of person $2$, only person $1$ is taller than person $2$. Thus, output $1$ on the second line.  
	在 $2$ 左边的人中，只有 $1$ 比 $2$ 高。因此，第二行输出 $1$ 。
- Among the people to the left of person $3$, persons $1,2$ are taller than person $3$, and the person standing closest to person $3$ is person $2$. Thus, output $2$ on the third line.  
	在 $3$ 左边的人中， $1,2$ 比 $3$ 高，离 $3$ 最近的人是 $2$ 。因此，第三行输出 $2$ 。
- There is no person to the left of person $4$ who is taller than person $4$. Thus, output $-1$ on the fourth line.  
	在 $4$ 左边没有人比 $4$ 高。因此，第四行输出 $-1$ 。

---

### Sample Input 2 示例输入 2 复制

```
3
7 7 7
```

### Sample Output 2 示例输出 2Copy

```
-1
-1
-1
```

There may be multiple people with the same height.  
可能有多人身高相同。

---

### Sample Input 3 示例输入 3 副本

```
6
31 9 17 10 2 9
```

### Sample Output 3 示例输出 3 副本

```
-1
1
1
3
4
4
```
