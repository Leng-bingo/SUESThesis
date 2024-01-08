# (Transformer)Attention Is All You Need

`Transformer`论文链接：https://arxiv.org/pdf/1706.03762.pdf



- RNN 特点（缺点）：从左往右一步一步计算，对第 t 个状态 ht，由 ht-1（历史信息）和 当前词 t 计算。

  - 难以并行。100个词要算100步。

  - 过早的信息可能被丢掉。时序信息是一步一步往后传递的，当文本很长的时候，前方的信息就被忽略的。

  - 如果句子很长，计算每一步都需要存储，内存开销大。

- **自注意力机制**

  ![](https://leng-mypic.oss-cn-beijing.aliyuncs.com/20220509202157.png)

  - 输入的词都会先变成512的向量，然后融合位置信息。
  - **编码器：**使用自己本身复制维相同的三个，分别当做输入key，value，query。query是key和value每个之间的相似度关系。
  - 然后加入残差
  - 再加上MLP加残差
  - **解码器：**比编码器多一个模块，mask。只看当前词前面的信息，把从这个词后面的东西赋予一个很大的负数。再利用得出query和编码器的key和value再次送入block进行计算。