# BERT论文

>  论文链接https://arxiv.org/abs/1810.04805

---

新的语言表征模型 BERT: **B**idirectional **E**ncoder **R**epresentations from **T**ransformers，基于 ELMo Transformers 模型的双向编码表示

与 ELMo 和 GPT 不同，BERT 从无标注的文本中（jointly conditioning 联合左右的上下文信息）预训练得到 无标注文本的 deep bidirectional representations

---

GPT unidirectional，使用左边的上下文信息 预测未来
BERT bidirectional，使用左右侧的上下文信息

- **Trick：自己写论文时，要写出自己的准确率是多少，比别人高多少。一目了然，有对比性。**

NLP任务分为两类：一种是sentence-level tasks 句子情绪识别或者识别两个句子的关系，另一种是token-level tasks NER (人名、街道名) ，识别任务的实体。

---

pre-trained language representations 两类策略：

1. **基于特征的**： ELMo (构建和每一个下游任务相关的 NN 架构；训练好的特征（作为额外的特征） 和 输入 一起放进模型)
2. **基于微调参数的 GPT**：所有的权重参数根据新的数据集进行微调。

---

【CLS】判定句子的开始

【SEP】表示断开两个橘子

例如：【CLS】 [Token1] …… [Token n] 【SEP】 [Token1'] …… [Token m]

---

每一个 token 进入 BERT 得到 这个 token 的embedding 表示。
对于 BERT，输入一个序列，输出一个序列。

最后一个 transformer 块的输出，表示 这个词源 token 的 BERT 的表示。在后面再添加额外的输出层，来得到想要的结果。

For a given token, 进入 BERT 的表示 = token 本身的表示 + segment 句子的表示 + position embedding 位置表示

![](https://leng-mypic.oss-cn-beijing.aliyuncs.com/mac-img/20220516204410.png)

Token embeddings:  词源的embedding层，整成的embedding层， 每一个 token 有对应的词向量。
Segement embeddings: 这个 token 属于第一句话 A还是第二句话 B。
Position embeddings: 输入的大小 = 这个序列最长有多长？ i.e., 1024 
Position embedding 的输入是 token 词源在这个序列 sequence 中的位置信息。从0开始 1 2 3 4 --> 1024

---

生成的词源序列中的词源，它有 15% 的概率会随机替换成一个掩码。但是对于特殊的词源不做替换 

MLM （类似完形填空）带来的问题：预训练和微调看到的数据不一样。预训练的输入序列有 15% [MASK]，微调时的数据没有 [MASK].

15% 计划被 masked 的词: 80% 的概率被替换为 [MASK], 10% 换成 random token,10% 不改变原 token。但 T_i 还是被用来做预测。

---

**参数越多，效果越好。**

1. 摘要（Abstract）：与别的文章的区别是什么？效果有多好？

2. 引言（Introduction）：语言模型的简单介绍；摘要第一段的扩充；主要想法；如何解决所遇到的问题；
   **贡献点：** **双向信息**的重要性（句子从左看到右，从右看到左）、在BERT上做微调效果很好、代码开源

3. 结论（Conlusion）：无监督的预训练很重要（在计算机视觉领域，在没有标签的数据集上做训练比在有标签的数据集上做训练效果会更好）；主要贡献是将这些发现进一步推广到深度双向架构，使相同的预训练模型能够成功处理一系列的 NLP 任务。

   选了选双向性带来的不好是什么？做一个选择会得到一些，也会失去一些。
   缺点是：与GPT（Improving Language Understanding by Generative Pre-Training）比，BERT用的是编码器，GPT用的是解码器。BERT做机器翻译、文本的摘要（生成类的任务）不好做。

---

完整解决问题的思路：在一个很大的数据集上训练好一个很宽很深的模型，可以用在很多小的问题上，通过微调来全面提升小数据的性能（在计算机视觉领域用了很多年），模型越大，效果越好（很简单很暴力）。