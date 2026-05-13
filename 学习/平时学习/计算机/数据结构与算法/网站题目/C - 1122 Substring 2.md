---
title: "C - 1122 Substring 2"
source: "https://atcoder.jp/contests/abc433/tasks/abc433_c"
author:
  - "[[AtCoder Inc.]]"
published: 2025-11-22
created: 2025-11-22
description: "AtCoder is a programming contest site for anyone from beginners to experts. We hold weekly programming contests online."
tags:
  - "算法与数据结构"
---
Score : $300$ points

### Problem Statement

You are given a string $S$ consisting of digits.

A string $T$ is called a **1122-string** if it satisfies all of the following conditions. (The definition is the same as in Problem F.)

- $T$ is a non-empty string consisting of digits.
- $|T|$ is even, where $|T|$ denotes the length of string $T$.
- All characters from the $1$\-st through the $\frac{|T|}2$\-th character of $T$ are the same digit.
- All characters from the $(\frac{|T|}2+1)$\-th through the $|T|$\-th character of $T$ are the same digit.
- Adding $1$ to the digit of the $1$\-st character of $T$ gives the digit of the $|T|$\-th character.

For example, `1122`, `01`, and `444555` are 1122-strings, but `1222` and `90` are not 1122-strings.

Find the number of **substrings** of $S$ that are 1122-strings.

Two substrings are counted separately if they are extracted from different positions, even if they are identical as strings.

### Constraints

- $S$ is a string consisting of digits with length between $1$ and $10^6$, inclusive.

---

### Input

The input is given from Standard Input in the following format:

$S$

### Output

Output the number of non-empty substrings of $S$ that are 1122-strings.

---

### Sample Input 1



1122

### Sample Output 1



2

The following two substrings satisfy the condition.

- `12` extracted from the $2$\-nd through $3$\-rd characters of $S$
- `1122` extracted from the $1$\-st through $4$\-th characters of $S$

Thus, output $2$.

---

### Sample Input 2



7788788

### Sample Output 2



3

Note that two substrings are counted separately if they are extracted from different positions, even if they are identical as strings.

---

### Sample Input 3



2025

### Sample Output 3



0

There may be no substring that is a 1122-string.

---

### Sample Input 4



1112222334445556555

### Sample Output 4



11