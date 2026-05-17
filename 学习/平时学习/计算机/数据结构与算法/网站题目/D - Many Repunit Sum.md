---
title: "D - Many Repunit Sum"
source: "https://atcoder.jp/contests/abc444/tasks/abc444_d"
author:
  - "[[AtCoder Inc.]]"
published: 2026-02-07
created: 2026-02-07
description: "AtCoder is a programming contest site for anyone from beginners to experts. We hold weekly programming contests online."
tags:
  - "clippings"
---
### Problem Statement

For $i=1,2,\dots,N$, let $B_i$ denote the integer formed by concatenating $A_i$ ones.  
More formally, $B_i=\sum_{j=0}^{A_i-1}{10^j}$.  
Find $\sum_{i=1}^{N}{B_i}$.

### Constraints

- $1 \leq N \leq 2 \times 10^5$
- $1 \leq A_i \leq 2 \times 10^5$
- All input values are integers.

---

### Input

The input is given from Standard Input in the following format:

$N$  
$A_1$ $A_2$ $\ldots$ $A_N$  

### Output

Output the answer in one line.

---

### Sample Input 1Copy

Copy

4
3 3 3 3

### Sample Output 1Copy

Copy

444

$B_1=B_2=B_3=B_4=111$, so $B_1+B_2+B_3+B_4=444$.

---

### Sample Input 2Copy

Copy

3
30 10 20

### Sample Output 2Copy

Copy

111111111122222222223333333333

The answer may be very large.

---

### Sample Input 3Copy

Copy

10
1 2 3 4 5 6 7 8 9 10

### Sample Output 3Copy

Copy

1234567900