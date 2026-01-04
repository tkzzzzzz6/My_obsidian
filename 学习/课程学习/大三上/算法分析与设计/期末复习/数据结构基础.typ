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
  footer: [
    #align(right)[
      #context [
        #text("第 ")
        #counter(page).display("1")
        #text(" / ")
        #text(str(counter(page).final().at(0)))
        #text(" 页")
      ]
    ]
  ],
  header-ascent: 8pt,
)

#set text(size: 8pt)
#set raw(tab-size: 2)
#show raw.where(block: true): it => block(
  fill: rgb(230, 240, 255),
  stroke: (paint: rgb(150, 180, 230), thickness: 1pt),
  radius: 3pt,
  inset: (x: 8pt, y: 5pt),
  width: 95%,
  breakable: true,
)[
  #set text(font: "fira code", size: 6.8pt, weight: "bold")
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

数据结构相关

= 排序

#problem("直接插入排序")[
  *题意:*


  *输入:*


  *输出:*


  ```cpp


  ```
]

#problem("冒泡排序")[
  *题意:*


  *输入:*


  *输出:*


  ```cpp


  ```
]


= 查找

#problem("顺序查找")[
  *题意:*


  *输入:*


  *输出:*


  ```cpp


  ```
]

#problem("二分查找")[
  *题意:*


  *输入:*


  *输出:*


  ```cpp


  ```
]

= 栈和队列

#problem("链栈出栈")[
  *题意:*


  *输入:*


  *输出:*


  ```cpp


  ```
]

#problem("链栈入栈")[
  *题意:*


  *输入:*


  *输出:*


  ```cpp


  ```
]
#problem("顺序栈的入栈、出栈")[
  *题意:*


  *输入:*


  *输出:*


  ```cpp


  ```
]
#problem("循环队的出队、入队")[
  *题意:*


  *输入:*


  *输出:*


  ```cpp


  ```
]
#problem("双链表中在第i个位置上添加元素")[
  *题意:*


  *输入:*


  *输出:*


  ```cpp


  ```
]
#problem("双链表中删除第i个位置上的元素")[
  *题意:*


  *输入:*


  *输出:*


  ```cpp


  ```
]


