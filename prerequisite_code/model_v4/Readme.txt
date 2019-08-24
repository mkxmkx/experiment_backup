修改初始版本代码，增加了head attention部分。
修改了原代码中求topic_relation_score部分的错误，现在的score为正数。（以前求出来的是负数，错的）
修改loss函数，将L2正则化部分改为整个模型的权值
