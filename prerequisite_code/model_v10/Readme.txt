修改初始版本代码，增加了head attention部分。
修改了原代码中求topic_relation_score部分的错误，现在的score为正数。（以前求出来的是负数，错的）
修改loss函数，将L2正则化部分改为整个模型的权值

-------------------------------------------------------
v7：在TutorialBank数据集上进行实验，v2数据集（即只考虑对应维基页面上文本，不考虑链接到的文本）。修改如下：
1、修改相似度函数，增加欧氏距离相似函数和曼哈顿距离相似函数
lambda = 0.01（优参）

---------------------------------------------------------
v10: 在v7基础上
数据集不变。
修改代码，使可以输出最终结果中每个mention的start和end index，以确定对应实际的单词。还有mention对应的先序词。以及score。