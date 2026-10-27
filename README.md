# 能看自己为什么仍然自回归：右移目标与因果掩码

关注当前位置是关注右移输入中的当前位置，并未看到待预测标签。 这篇围绕一个新阅读问题，将D1–D4逐项核对。相同论文可以提出不同问题；本次调用与旧表格读数任务独立，不复用旧回答。

## 问题、方法与实测

完整问题见[task.txt](task.txt)，运行前标准见[criteria.md](criteria.md)，冻结哈希见[protocol-lock.json](protocol-lock.json)。只给所选论文片段，逐行编号，要求D1–D4 JSON；服务无历史无工具，先dry-run逐字核对message后真实调用一次。

模型：chenkun/gpt-5-mini；completion id：chatcmpl-EdW2j96WfDfbghVVzqjxhGdNx03mB；开始时间UTC：2026-10-27T07:37:52.513522+00:00；总耗时39.264秒；prompt/completion/total tokens：1350/5348/6698。费用null，服务未返回逐次货币扣费。响应中的账户余额脱敏，模型content逐字保留；未伪造费用或实验。

机械验收：range_errors，10段合法范围，错误=['D1:missing_field:limitations', 'D2:missing_field:limitations', 'D3:missing_field:limitations', 'D4:missing_field:limitations']。语义判定见每题核查；不将合法范围当作支持性证明。

输入[request.txt](request.txt)、预检[preflight.json](preflight.json)、响应[response.json](response.json)、原答[output.json](output.json)、元数据[metadata.json](metadata.json)、检查[validation.json](validation.json)均公开。

## 四道子题：原文、原答与作者核查

### D1：第3页称预测 i 只依赖小于 i 的输出，第5页却允许关注当前位置；用右移目标输入解释二者为何一致。

**结论**：关注当前位置是关注右移输入中的当前位置，并未看到待预测标签。

**论文原文**：物理页3，P3:16–19。保留前后句如下：

```text
around each of the sub-layers, followed by layer normalization. We also modify the self-attention
sub-layer in the decoder stack to prevent positions from attending to subsequent positions. This
masking, combined with fact that the output embeddings are offset by one position, ensures that the
predictions for position i can depend only on the known outputs at positions less than i.
```

**模型原答（原样）**：

```text
问题：第3页说“对位置 i 的预测只依赖小于 i 的已知输出”，而第5页又说 decoder 的 self-attention “允许每个位置关注到该位置及之前的位置（up to and including that position）”。二者如何一致？
回答：这两句在模型内部是兼容的，关键在于“输出嵌入向右移动一位（offset by one position）”＋“因果/前瞻掩码（causal mask）”。具体说明如下——
推导：在训练时，decoder 的输入序列是目标序列右移一位（例如用 BOS 填在最前面）：即当模型要预测目标序列中第 i 个真实 token 时，decoder 在位置 i 看到的输入恰好是目标序列的第 i−1 个 token（因为所有输出嵌入都向右偏移了一个位置）。decoder 的 self-attention 被允许在 decoder 序列的位置维度上“attend 到该位置及其之前的位置”（P5 行29-31），但由于位置 i 的输入本身已经是前一时刻的已知 token（i−1），即使它可以“attend 到自身（该位置）”，该“自身”也只包含了已知的第 i−1 个输出信息。因此，结合掩码（prevent positions from attending to subsequent positions，P3 行17-19 与 P5 行29-31），预测位置 i 无法获得关于真实第 i 或更后位置的任何信息，因而“预测 i 只依赖小于 i 的输出”与“允许关注当前位置”并不矛盾。
限制（limitations）：以上结论基于文中关于“输出嵌入右移一位”和“在 decoder 中屏蔽未来位置”的描述（见证据）。文中并未在这些段落中给出逐步执行示例或对其它可能排列（例如不同右移策略）进行实验对比；因此若采用不同的工程实现或不同的对齐约定，需小心验证行为是否保持一致。

模型 limitations字段：缺失，见answer内的限制段落
```

**作者推导**：目标[A,B,EOS]，输入[BOS,A,B]；输入位置2持有A、预测B。允许注意输入位置1/2只看BOS与A，因此预测y_i仅用y_<i；未来位置屏蔽才能阻止泄露。

**逐题核查**：原答解释正确。但四题均把limitations塞进answer而非独立字段，机械模式失败，不能报告全部通过。


### D2：区分 decoder self-attention、encoder self-attention、encoder-decoder attention 的 Q/K/V 来源与可见范围。

**结论**：三种attention的Q/K/V来源和可见范围必须分开。

**论文原文**：物理页5，P5:19–33。保留前后句如下：

```text
The Transformer uses multi-head attention in three different ways:
• In "encoder-decoder attention" layers, the queries come from the previous decoder layer,
and the memory keys and values come from the output of the encoder. This allows every
position in the decoder to attend over all positions in the input sequence. This mimics the
typical encoder-decoder attention mechanisms in sequence-to-sequence models such as
[38, 2, 9].
• The encoder contains self-attention layers. In a self-attention layer all of the keys, values
and queries come from the same place, in this case, the output of the previous layer in the
encoder. Each position in the encoder can attend to all positions in the previous layer of the
encoder.
• Similarly, self-attention layers in the decoder allow each position in the decoder to attend to
all positions in the decoder up to and including that position. We need to prevent leftward
information flow in the decoder to preserve the auto-regressive property. We implement this
inside of scaled dot-product attention by masking out (setting to −∞) all values in the input
of the softmax which correspond to illegal connections. See Figure 2.
```

**模型原答（原样）**：

```text
问题：区分三类 attention（decoder 自注意、encoder 自注意、encoder-decoder attention）的 Q/K/V 来源与每类可见范围。
回答（逐类说明）：
1) Encoder self-attention：
   - Q、K、V 的来源：全部来自 encoder 中上一层（“在自注意层中所有的 keys, values and queries 来自同一处，在本文中是 encoder 前一层的输出”）。（见证据）
   - 可见范围（可 attend 的位置）：每个 encoder 的位置可以 attend 到 encoder 前一层的所有位置（没有因果屏蔽，完全可见）。（见证据）
2) Decoder self-attention（在 decoder 中的 self-attention 子层）：
   - Q、K、V 的来源：全部来自 decoder 中上一层（即 decoder 自己的前一层输出），与 encoder 自注意类似地是“self”来源。（见证据）
   - 可见范围：每个 decoder 位置只能 attend 到该序列中“该位置及其之前的位置（up to and including that position）”，同时必须阻止关注未来位置（文中通过在 softmax 输入上掩码来实现，see P5 行29-33；P3 行17-19 也说明了这种掩码与右移嵌入结合以保证自回归性质）。
3) Encoder-decoder attention（decoder 中的第三个子层）：
   - Q、K、V 的来源：queries 来自 decoder 的前一层输出；keys 和 values 来自 encoder 的输出（文中称 memory keys and values come from the output of the encoder）。（见证据）
   - 可见范围：这使得 decoder 中的每个位置可以 attend 到输入序列（encoder 输出）中的所有位置，换言之对 encoder 端没有因果屏蔽（可以查看 encoder 整个输入）。（见证据）
限制（limitations）：上述说明严格基于文中句子对 Q/K/V 来源与屏蔽策略的描述（见证据）。文中没有在这些段落提供逐元素矩阵表示或不同实现细节（例如在实现中如何排列多头的具体索引），也未对不同 masking 实现（例如在计算顺序或数值稳定性层面）的性能差异给出实验对比，因此关于实现细节的推断为理论性说明而非文中直接给出的实验结论。

模型 limitations字段：缺失，见answer内的限制段落
```

**作者推导**：encoder自注意Q/K/V来自encoder同一输入表示，可看全源；decoder自注意来自decoder同一输入表示，只看≤当前输入位置；cross-attention的Q来自decoder，K/V来自encoder输出，可看全源。第一层输入可来自embedding；“上一层”不应误解成全部来自encoder。

**逐题核查**：原答基本正确；论文用层的概述术语，实际cross-attention的query来自当前decoder子层输入，不需把层/子层索引误写成强制实现。


### D3：教学例目标[A,B,EOS]、输入[BOS,A,B]，写出按行query按列key的允许矩阵；等logits时各行权重是什么？屏蔽值为何在softmax前设负无穷？

**结论**：允许矩阵是含对角线的下三角；先掩码再softmax。

**论文原文**：物理页5，P5:29–33。保留前后句如下：

```text
• Similarly, self-attention layers in the decoder allow each position in the decoder to attend to
all positions in the decoder up to and including that position. We need to prevent leftward
information flow in the decoder to preserve the auto-regressive property. We implement this
inside of scaled dot-product attention by masking out (setting to −∞) all values in the input
of the softmax which correspond to illegal connections. See Figure 2.
```

**模型原答（原样）**：

```text
问题与设定：给定目标序列 [A, B, EOS]，decoder 的输入为右移后的 [BOS, A, B]。请写出“按行 query、按列 key 的允许（可访问）矩阵”；在 logits 相等时各行 attention 权重为何？为何在 softmax 前把屏蔽位置设为 −∞？
回答：
1) 索引与对应关系（为清楚解释，先定义位置索引）：
   - decoder 输入位置 1: BOS（用于预测目标第1个 token A）
   - decoder 输入位置 2: A   （用于预测目标第2个 token B）
   - decoder 输入位置 3: B   （用于预测目标第3个 token EOS）
2) 允许矩阵（行 = query 位置，列 = key 位置；1 表示允许 attend，0 表示被屏蔽）：
   - 行1（query 在位置1）: [1, 0, 0]
   - 行2（query 在位置2）: [1, 1, 0]
   - 行3（query 在位置3）: [1, 1, 1]
   解释：decoder 的每个位置可以 attend 到该位置及之前的位置（P5 行29-31），因此第1行只能看第1列，第2行可以看第1和第2列，第3行可以看第1、2、3列。
3) 当 logits（未做 softmax 的打分）在允许位置上相等时，各行的 attention 权重（softmax 输出）如下（推导）：
   - 推导：softmax 在一行上对允许项归一化；若允许项数量为 k 且它们的 logits 都相等，则每一项的权重均为 1/k。
   - 因此：
     * 行1：只有一项，被赋权 1（1/1）
     * 行2：两项，各为 1/2
     * 行3：三项，各为 1/3
4) 为何在 softmax 前把屏蔽位置设为 −∞？（P5 行32-33 给出该实现细节）
   - 推导：softmax 的定义为 softmax(z)i = exp(z_i) / sum_j exp(z_j)。若将某些不合法连接对应的 logits 设为 −∞，则 exp(−∞) = 0，因此这些位置在 softmax 后的权重恰好为 0，从而“屏蔽”掉它们对输出的任何贡献；这同时保留了对合法位置归一化的性质（即合法位置的权重之和为 1）。因此，在 logits 输入处设置 −∞ 是一种直接且精确地把非法连接的权重置为 0 的方法（并且便于与矩阵化并行运算结合）。
限制（limitations）：文中仅陈述了使用 −∞ 的实现方式（P5 行32-33），并未给出关于数值稳定性细节或在浮点实现中如何近似 −∞ 的说明；关于在实际实现中使用非常小的负数代替 −∞ 的效果与实验对比并未在这些段落中给出，因此上述关于 exp(−∞)=0 的推导在工程实现上需要按数值表示注意稳定性问题，这一点超出所给证据范围。

模型 limitations字段：缺失，见answer内的限制段落
```

**作者推导**：行query列key：[[1,0,0],[1,1,0],[1,1,1]]。等logits时权重[[1,0,0],[1/2,1/2,0],[1/3,1/3,1/3]]。exp(-∞)=0，分母只含合法位置；至少一个合法项才能正常归一化。

**逐题核查**：原答矩阵、权重和分母正确；并非所有位置都mask，BOS/对角线提供合法项。


### D4：把未来 logits 设0是否等于屏蔽？把softmax后未来权重置0但不归一化会怎样？该掩码是否意味着训练必须逐token串行？

**结论**：未来logit置0并非屏蔽，softmax后截断且不归一化也改变输出。

**论文原文**：物理页5，P5:29–33。保留前后句如下：

```text
• Similarly, self-attention layers in the decoder allow each position in the decoder to attend to
all positions in the decoder up to and including that position. We need to prevent leftward
information flow in the decoder to preserve the auto-regressive property. We implement this
inside of scaled dot-product attention by masking out (setting to −∞) all values in the input
of the softmax which correspond to illegal connections. See Figure 2.
```

**模型原答（原样）**：

```text
问题：把未来 logits 直接设为 0 是否等价于用 −∞ 屏蔽？若在 softmax 后把未来权重置为 0 但不再归一化会怎样？该掩码是否意味着训练必须逐 token 串行（不能并行）？
回答：
1) 将未来 logits 设为 0 ≠ 等价于将其设为 −∞（文中所用的屏蔽方式）。理由（推导）：
   - softmax 是按行做 exp 然后归一化的操作；若把非法位置的 logit 设为 −∞，exp(−∞)=0，使这些位置在归一化后权重恰为 0，且合法位置的权重在归一化时仍和为 1（这是 P5 行32-33 所描述的做法）。
   - 若把非法位置的 logit 设为 0，而合法位置的 logits 不是都很负，那么 exp(0) = 1，这会给非法位置分配非零质量，从而改变归一化分母和合法位置的最终权重分配。因此一般情况 0 与 −∞ 不是等价的（只有在非常特殊的情形下，譬如所有合法位置的 logits 极为负，使得 exp(合法)≈0，相对关系才可能近似）。
2) 若在 softmax 后把未来位置的权重直接置为 0 但不重新归一化，会产生什么后果：
   - 结果：该行 attention 权重之和将小于 1（因为把部分质量直接丢弃而未重新分配或归一化）。这会导致 attention 输出向量是合法位置的加权和但整体范数被缩小；模型接收到的上下文信息的幅度（scale）会发生变化，进而可能影响后续层的数值与学习行为（例如梯度尺度）。文中未就这种做法给出描述或实验比较，因此无法从给定证据判断其训练效果或是否可行。
3) 掩码是否意味着训练必须逐 token 串行？
   - 结论（推导）：否。文中将“输出嵌入右移一位”与“在 decoder self-attention 中屏蔽未来位置”结合，来保证自回归性质（P3 行17-19，P5 行29-31）。但这些掩码是在 attention 的 logits 矩阵上进行的元素级操作（P5 行32-33），attention 本身由向量矩阵乘法和一次性 softmax 计算得到（P3 行2-4 描述了使用堆叠的 self-attention 与逐位置的 point-wise 前馈层）。因此可以在整个序列的所有位置上并行地计算 queries、keys、values，构造完整的 logits 矩阵，然后一次性应用掩码（把非法位置设为 −∞）并做 softmax，从而并行得到所有位置的注意力分布。换言之，掩码并不强制训练必须逐 token 串行：它仅限制信息流向，但并不阻止对整个序列并行地计算被屏蔽后的注意力矩阵（文中通过描述掩码在 softmax 输入上的应用间接支持这一点）。
限制（limitations）：文中在这些段落没有明确用“并行计算”或“并行训练”的字眼来阐述掩码与训练并行性的具体实现/性能成本；上述关于可以并行计算的结论是基于对 attention 矩阵化计算与掩码作用的推理（结合 P3 行2-4 与 P5 行32-33 的描述）。此外，文中未对“把未来 logits 设为 0”或“在 softmax 后零掉再不归一化”这类替代实现的数值稳定性或训练行为做实验比较，因此对这些替代方法的实际效果与收敛特性无法从给定证据中断言，需要额外实验验证。

模型 limitations字段：缺失，见answer内的限制段落
```

**作者推导**：所有三项原logit为0时，置0仍给每项1/3；第一行后截断只剩[1/3,0,0]，行和1/3而非1。若合法logit=-100、未来置0，质量几乎全给未来，错误更严重；只有合法总exp远大于未来项数时，置0未来的影响才可能很小。并行teacher forcing仍可同时计算各位置；生成阶段通常依赖已生成前缀。

**逐题核查**：原答D4出现反向例外：“所有合法极负”会让未来占优，不会接近屏蔽。另外“不重归一化使范数缩小”不能当任意向量的一般断言；本设定相对重新归一化的合法加权和是标量缩放，比较未屏蔽完整和时存在抵消。原答不修改。

## 离线可复算算例

运行 `python offline_calc.py`，仅使用Python标准库，无网络、无GPU、不训练模型。以下为实际脚本输出；论文事实、题设教学输入与代数推导在各题中已区分。

```json
{
  "teaching_only": true,
  "rows": "query",
  "columns": "key",
  "target": [
    "A",
    "B",
    "EOS"
  ],
  "input": [
    "BOS",
    "A",
    "B"
  ],
  "allowed": [
    [
      1,
      0,
      0
    ],
    [
      1,
      1,
      0
    ],
    [
      1,
      1,
      1
    ]
  ],
  "weights": [
    [
      1.0,
      0.0,
      0.0
    ],
    [
      0.5,
      0.5,
      0.0
    ],
    [
      0.3333333333333333,
      0.3333333333333333,
      0.3333333333333333
    ]
  ],
  "zero_is_not_mask": [
    0.3333333333333333,
    0.3333333333333333,
    0.3333333333333333
  ],
  "negative_legal_counterexample": [
    1.860037988010418e-44,
    0.5,
    0.5
  ],
  "post_softmax_first_row_without_renorm": [
    0.3333333333333333,
    0,
    0
  ]
}
```

源代码：[offline_calc.py](offline_calc.py)，保存输出：[offline-output.txt](offline-output.txt)。检查归一化、分母、比较方向的assert会在错误时中止；不是论文性能复现。

## 作者复核清单与局限

- 已按原页图核对公式、指数、转置、定义和上下文；页图见下方索引。文本抽取的断行公式已在调用前重建，保留原始冻结文件，事后不改模型输入或回答。
- 分母：逐题核对softmax归一化/平方根/宽度预算/交叉熵等对应分母；单位：参数个数、step、秒、nats、无量纲概率分开使用。比较方向已按各题算例核算。
- precision/recall、TP/FP/FN：本任务不是二分类检测，无混淆矩阵可报告；第75题的BLEU背景也不能替换成TP/FP/FN。未制造这些分数。
- 表头：本次关键证据为公式与正文，不从无表头行臆测数据；页图可能含其他表格，其数值未作为本题实测。
- 因果：单次调用、无对照、非盲、作者熟悉论文；不声称提示词带来提升，不用这批平台总分证明提示收益。
- 核查由Codex辅助执行逐项语义复核，不能冒称有独立人类审稿人或用户本人已人工签字。仍建议使用者在科研复用前复核关键推导。
- 所选片段可能不含完整定义；作者补充与题设教学假定明确标记。服务端seed/temperature和版本快照不可固定，重新调用可能给不同答案；离线脚本的确定性算术可直接复算。

## 复做步骤

1. 下载公开仓库；根README即完整正文，离线可读。按[SOURCES.md](SOURCES.md)获取同版论文，对照原页图和[numbered.txt](numbered.txt)。
2. `python validate.py sources.json output.json` 重放机械验收。第73题应保留失败字段；失败不是要求修写原答的理由。
3. `python offline_calc.py` 重放教学算术，对照保存输出。
4. 如需新真实调用，在已安装并登录Bohrium CLI的环境运行 `python reproduce_call.py --run-name my-new-run`；会产生新付费/额度调用，默认不会运行，已有目录禁止覆盖。结果是新的随机样本，不覆盖本次记录。
5. 参照[audit.md](audit.md)逐项复核原文支持性，区分论文值、教学推导和模型陈述。

## 材料与署名

原文证据：[sources.json](sources.json)、[numbered.txt](numbered.txt)。页图：[物理页3](page3.png)、[物理页5](page5.png)。完整输入输出、方法、审计与脚本均位于仓库根目录。

论文归Vaswani等；arXiv:1706.03762v7，NeurIPS 2017。本分享的原创解释与脚本MIT；论文片段版权归原作者，仅作为引用证据，不重新分发整篇PDF。外部指标定义如BLEU见[SOURCES.md](SOURCES.md)。
