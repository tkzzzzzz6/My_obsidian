#set page(
  paper: "a4",
  margin: (top: 18mm, bottom: 18mm, left: 16mm, right: 16mm),
  columns: 2,
  header: [#align(right)[*by SICAU ttkwqe*]],
  header-ascent: 8pt,
)

#set columns(gutter: 10pt)

#set text(size: 10pt)
#set heading(numbering: "1.", outlined: true)
#let problem(
  title,
  difficulty: "",
  tags: (),
  source: "",
  body,
) = [
  #heading(level: 2)[#title]
  #if difficulty != "" [*难度:* #difficulty #linebreak()]
  #if source != "" [*来源:* #source #linebreak()]
  #if tags.len() > 0 [*标签:* #tags #linebreak()]
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




