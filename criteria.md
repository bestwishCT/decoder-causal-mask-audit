运行前标准：逐项回答D1-D4，保留论文条件与假设；数值与公式可离线复算；区分论文观测、教学算术与作者判断；引文范围存在不等于语义支持。检查分母、单位、方向、因果、表头。precision/recall与TP/FP/FN不适用时明确标记，不造指标；输出错误必须保留。单次无对照，不宣称提示增益。
第3页称预测 i 只依赖小于 i 的输出，第5页却允许关注当前位置；用右移目标输入解释二者为何一致。
区分 decoder self-attention、encoder self-attention、encoder-decoder attention 的 Q/K/V 来源与可见范围。
教学例目标[A,B,EOS]、输入[BOS,A,B]，写出按行query按列key的允许矩阵；等logits时各行权重是什么？屏蔽值为何在softmax前设负无穷？
把未来 logits 设0是否等于屏蔽？把softmax后未来权重置0但不归一化会怎样？该掩码是否意味着训练必须逐token串行？