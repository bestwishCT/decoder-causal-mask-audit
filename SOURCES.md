# 原文与来源

Vaswani et al. Attention Is All You Need. NeurIPS 2017，arXiv:1706.03762v7。https://arxiv.org/abs/1706.03762v7 。PDF SHA256：bdfaa68d8984f0dc02beaca527b76f207d99b666d31d1da728ee0728182df697。

本包仅含所引页面的文字片段和核查用页图，不含整篇PDF。物理页码与所引页脚页码对应。pypdf抽取文本；第4页分式重建为QK^T/sqrt(d_k)、脚注求和重建为sum，第7页公式重建为显式幂；第5页抽取的WQ_i等下标断行保留并以页图确认形状。重建在冻结前完成，不改变论文含义。部分页只取相关节，前后上下文保留。

第74题d_model定义为作者补充背景，原给定P7没有明确文字定义。第75题交叉熵/单tokenPPL是教学定义，平滑分布约定是题设。BLEU的modified n-gram precision及brevity penalty定义背景来源：Papineni et al. BLEU: a Method for Automatic Evaluation of Machine Translation, ACL 2002，https://aclanthology.org/P02-1040/ 。此定义没有伪装成给模型的P8原文。
