比如将任务划分成折线图/t-sne/可视化图等多个任务
按照几个subtask去做
level 1:多图判断好坏，A,B,C,D
level 2:单图判断关键点，这个可以设置几个打分点，比如看出折线图逐渐降低，算1分，然后看出收敛不平滑，算1分，看出在某一个epoch波动了算3分这样。Leve 2主要是评估阅读图是否全面
level 3：多图做阅读理解，GT参照论文，至于是不是ABCD再讨论。level 3可以用大模型评估，我们给出评分细则，比如描述结果正确给1分，描述的重点和GT的重点每吻合一个，给1分。Level 3主要是图->文字的解读是否完整

围绕的点就是当前VLM的机制，比如可能只关注几个重点，尤其是图像，而不能看到全局其他非重要问题，
https://scholar.google.com/citations?user=ln4hmCwAAAAJ&hl=en
关于VLM的模型在图表方向的benchmark
![[2d400a42e69aef4fd516c922cfa5357d.png]]
文生图：https://arxiv.org/abs/2503.07265
Reasoning: https://arxiv.org/abs/2505.14552
General Knowledge: https://www.arxiv.org/abs/2502.14739
能做出让qwen用上的benchmark就算成功
要能被qwen用都不用中了，arxiv都是胜利