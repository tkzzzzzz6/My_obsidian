#set page(
  paper: "a4",
  margin: (top: 18mm, bottom: 18mm, left: 16mm, right: 16mm),
  columns: 2,
  header: [
    #align(right)[*by SICAU ttkwqe*]
    #line(length: 100%)
  ],
  header-ascent: 8pt,
)

#set columns(gutter: 10pt)

#set text(size: 10pt)
#set raw(tab-size: 2)
#show raw.where(block: true): it => block(
  fill: rgb(230, 240, 255),
  stroke: (paint: rgb(150, 180, 230), thickness: 1pt),
  radius: 3pt,
  inset: (x: 8pt, y: 5pt),
  width: 100%,
  breakable: true,
)[
  #set text(font: "Consolas", size: 11pt)
  #align(left)[#it]
]
#set heading(numbering: "1.", outlined: true)
#set columns(1)
#align(center)[
  #text(size: 18pt, weight: "bold")[算法分析与设计-程序设计题库]
]
#v(6pt)
#set columns(2)

#let problem(
  title,
  difficulty: "",
  tags: (),
  source: "",
  body,
) = [
  #heading(level: 2)[#title]
  #body
  #line(length: 100%)
]

= 基础知识
#problem("欧几里得算法求最小公倍数")[
  *题意*:

  *输入:*

  *输出:*

  *伪代码:*
  ```cpp
  function gcd(a,b)
      while b != 0 do
          t := b
          b := a mod b
          a := t
  ```
]

#problem("顺序查找")[
  *题意*:

  *输入:*

  *输出:*

  *伪代码:*
  ```cpp

  ```
]

  #problem("Hanoi塔问题")[
    *题意*:

    *输入:*

    *输出:*

    *伪代码:*
    ```cpp

    ```
  ]
