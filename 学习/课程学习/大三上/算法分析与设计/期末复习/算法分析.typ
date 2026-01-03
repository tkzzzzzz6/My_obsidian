#set page(
  paper: "a4",
  margin: (top: 18mm, bottom: 18mm, left: 16mm, right: 16mm),
  columns: 2,
  background: [
    #place(center, dy: 50pt)[
      #line(
        length: 88.5%,
        angle: 90deg,
        stroke: (paint: luma(120), thickness: 1pt, dash: "dashed"),
      )
    ]
  ],
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
  #set text(font: "fira code", size: 8pt, weight: "bold")
  #align(left)[#it]
]
#set heading(numbering: "1.", outlined: true)
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

= 基础知识
  #problem("欧几里德算法求最大公约数(辗转相除法)")[
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
    将一个数组按照升序(降序)排列

    *输入:*
    输入一个数组A及其长度n。

    *输出:*
    输出排序后的数组。

    ```cpp
    function merge_sort(A,left,right)
        if left < right then
            mid := (left + right) / 2
            merge_sort(A,left,mid)
            merge_sort(A,mid+1,right)
            merge(A,left,mid,right)

    function merge(A,left,mid,right)
        n1 := mid - left + 1
        n2 := right - mid
        create arrays L[1..n1 + 1] and R[1..n2 + 1]
        for i <- 1 to n1 do
            L[i] := A[left + i - 1]
        for j <- 1 to n2 do
            R[j] := A[mid + j]
        L[n1 + 1] := INFINITY
        R[n2 + 1] := INFINITY
        i := 1
        j := 1
        for k <- left to right do
          if L[i] <= R[j] then
                A[k] := L[i]
                i := i + 1
            else
                A[k] := R[j]
                j := j + 1
            if i > n1 then
                for m <- j to n2 do
                    A[k + 1] := R[m]
                    k := k + 1
            else if j > n2 then
                for m <- i to n1 do
                    A[k + 1] := L[m]
                    k := k + 1


    ```
  ]

  #problem("找第二大(小)的数算法")[
    *题意:*
    查找一个数组中第二大(小)的数。


    *输入:*
    一个数组A及其长度n。

    *输出:*
    输出数组中第二大(小)的数。

    1. 顺序查找
    ```cpp
    function find_second_largest(A,n)
        MIN_VALUE := -INFINITY
        if n < 2 then
            return "Array must have at least two elements"
        largest := MIN_VALUE
        second_largest := MIN_VALUE
        for i <- 1 to n do
            if A[i] > largest then
                second_largest := largest
                largest := A[i]
            else if A[i] > second_largest and A[i] != largest then
                second_largest := A[i]
        if second_largest = MIN_VALUE then
            return "No second largest element"
        return second_largest
    ```
    2. 锦标赛法
    ```cpp
    function find_second_largest_tournament(A,n)
        create a tournament tree
        for i<-1 to n do
            insert A[i] into the tournament tree
        largest := root of the tournament tree
        candidates := elements that lost to largest
        second_largest := -INFINITY
        for each candidates do
            if candidate > second_largest then
                second_largest := candidate
        return second_largest
    ```
  ]

  #problem("素数测试算法")[
    *题意:*
    检查一个数是否为素数。

    *输入:*
    输入一个正整数n。

    *输出:*
    输出n是否为素数。

    ```cpp
    function is_prime(n)
        if n <= 1 then
            return false
        for i <- 2 to sqrt(n) do
            if n mod i = 0 then
                return false
        return true

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
