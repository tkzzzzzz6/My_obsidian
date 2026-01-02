#set page(
  paper: "a4",
  margin: (top: 18mm, bottom: 18mm, left: 16mm, right: 16mm),
  header: [
    #align(right)[*by~~SICAU~~ttkwqe*]
    #line(length: 100%)
  ],
  header-ascent: 8pt,
)

#set text(size: 9pt)
#set raw(tab-size: 2)
#show raw.where(block: true): it => block(
  fill: rgb(230, 240, 255),
  stroke: (paint: rgb(150, 180, 230), thickness: 1pt),
  radius: 3pt,
  inset: (x: 8pt, y: 5pt),
  width: 95%,
  breakable: true,
)[
  #set text(font: "consolas", size: 8pt, weight: "bold")
  #align(left)[#it]
]
#set heading(numbering: "1.", outlined: true)
#align(center)[
  #text(size: 18pt, weight: "bold")[算法分析与设计---程序设计题复习资料]
]
#v(6pt)

#let problem(
  title,
  difficulty: "",
  tags: (),
  source: "",
  body,
) = [
  #heading(level: 2)[#title]
  #body
  #line(length: 95%)
]

#place(center)[
  #line(
    length: 95%,
    angle: 90deg,
    stroke: (paint: luma(120), thickness: 0.6pt),
  )
]
#columns(2, gutter: 10pt)[
= 基础知识
#problem("欧几里得算法求最大公约数")[
  *题意:*
  求两个正整数的最大公约数。
  
  *输入:*
  输入两个正整数a和b。

  *输出:*
  输出a和b的最大公约数。

  ```cpp
  function gcd(a,b)
      while b != 0 do
          t := b
          b := a mod b
          a := t
  ```
]

#problem("顺序查找")[
  *题意:*
  在一个数组中查找指定元素的位置。

  *输入:*
  输入一个数组A及其长度n，以及要查找的元素key。

  *输出:*
  返回元素key在数组中的位置，如果不存在则返回0。

  ```cpp
  function linear_search(A,n,key)
      for i <- 1 to n do
          if A[i] = key then
              return i
      return 0
  ```
]

#problem("Hanoi塔问题")[
  *题意:*
  将n个从小到大的盘子从源柱子移动到目标柱子,共A,B,C三根柱子,A移动到C。

  *输入:*
  输入盘子数量n。

  *输出:*
  输出移动盘子的步骤。

  ```cpp
  function hanoi(n,a,b,c)
      if n = 1 then
          move disk from a to c
      else
          hanoi(n-1,a,c,b)
          move disk from a to c
          hanoi(n-1,b,a,c)

  ```
]

#problem("插入排序")[
  *题意:*
  将一个数组按升序排列。

  *输入:*
  输入一个数组A及其长度n。

  *输出:*
  输出排序后的数组。

  ```cpp
  function insertion_sort(A,n)
      for j <- 2 to n do
          key := A[j]
          i := j - 1
          while i > 0 and A[i] > key do
              A[i + 1] := A[i]
              i := i - 1
          A[i + 1] := key

  ```
]

#problem("二分归并排序")[
  *题意:*


  *输入:*


  *输出:*


  ```cpp


  ```
]

#problem("找第二小的数算法")[
  *题意:*


  *输入:*


  *输出:*


  ```cpp


  ```
]

#problem("素数测试算法")[
  *题意:*


  *输入:*


  *输出:*


  ```cpp


  ```
]

= 分治策略
#problem("二分查找")[
  *题意:*


  *输入:*


  *输出:*


  ```cpp


  ```
]

#problem("分治法总体伪码描述")[
  *题意:*


  *输入:*


  *输出:*


  ```cpp


  ```
]

#problem("")[
  *题意:*


  *输入:*


  *输出:*


  ```cpp


  ```
]

= 动态规划

= 贪心算法

= 回溯与分支限界

= 线性规划

= 网络流算法

= 可能考的数据结构题



]
