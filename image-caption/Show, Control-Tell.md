# CVPR2019:Show, Control and Tell:A Framework for Generating Controllable and Grounded Captions

- 数据集

![image-20221024144952723](https://leng-mypic.oss-cn-beijing.aliyuncs.com/mac-img/image-20221024144952723.png)

### 1.摘要

- 提出一种可控制的图像描述生成模型，可以通过外界的控制信号操控模型生成多样化的描述。具体的，模型将图像中的一组区域作为控制信号，并生成基于这些区域的图像描述。

![image-20221024122049403](https://leng-mypic.oss-cn-beijing.aliyuncs.com/mac-img/image-20221024122049403.png)

### 2.方法

- 一个句子可以被认为是一个词的序列：进行区分视觉词（图像中存在的实体单词boy、cap、shirt等）和文本词（图像中没有对应实体的单词a、with、on等）。进一步，句子中的每个名词可以和修饰词组成一个名词块

![image-20221024122553059](https://leng-mypic.oss-cn-beijing.aliyuncs.com/mac-img/image-20221024122553059.png)

#### 2.1生成可控描述

- 给定图像I和区域有序序列集合R，生成的句子y描述R中的所有区域。
- 输入：**图像I和区域序列R**，R充当控制信号控制生个生成过程，联合预测句子词级和块级概率分布

#### 2.2Region序列

- region序列关系到整个模型的生成质量，首先要解决两个问题：1⃣️region如何得到（利用Faster- RCNN进行目标检测）。2⃣️region的序列如何控制（提出一个sorting network进行排序）。
- 排序网络：输入：目标检测特征（2048）、GloVe（Global Vectors for Word Representation把一个单词表达成实数组成的向量）提取文字特征（300维）、目标框大小和位置（4维）。最终映射到一个N维描述向量（几个区域几个N），每个区域一个向量。
  - 处理完所有的region后，会得到一个NxN的矩阵，通过Sinkhorn操作转化为一个soft排列矩阵（进行行归一化和列归一化）

![image-20221024134956187](https://leng-mypic.oss-cn-beijing.aliyuncs.com/mac-img/image-20221024134956187.png)

#### 2.3模型

- 解决了输入问题，下面是模型主体部分。模型加入了一个控制信号，因此在生成时，不仅要考虑句子的合理性p（yt｜R，I）在t时刻词语的概率，也要考虑生成的句子是符合给定region序列的

- 区域切换模块：region的选择是通过转换门gt（布尔类型）实现的：

  ![image-20221024141436606](https://leng-mypic.oss-cn-beijing.aliyuncs.com/mac-img/image-20221024141436606.png)

  - LSTM输入：隐藏状态ht-1、当前图像区域rt和当前词语wt
  - gt计算方法：首先计算一个哨兵值st，利用输出值和记忆细胞计算哨兵值。

  ![image-20221024141710638](https://leng-mypic.oss-cn-beijing.aliyuncs.com/mac-img/image-20221024141710638.png)

  - 然后计算哨兵和隐藏状态ht的相似度，以及隐藏状态和当前region每个之间的相似度

  ![image-20221024142102208](https://leng-mypic.oss-cn-beijing.aliyuncs.com/mac-img/image-20221024142102208.png)

  - 然后计算转换region的概率，

  ![image-20221024142340989](https://leng-mypic.oss-cn-beijing.aliyuncs.com/mac-img/image-20221024142340989.png)

- 具有视觉哨兵的适应性注意力

  - 区分视觉词和文本词
  - 跟切换区域类似，计算哨兵值
  - 然后利用Additive Attention计算region和哨兵值的相似度

  ![image-20221024144055409](https://leng-mypic.oss-cn-beijing.aliyuncs.com/mac-img/image-20221024144055409.png)

  - 然后可以得到一个加权向量，作为LSTM此时刻的输入



![image-20221024140312929](https://leng-mypic.oss-cn-beijing.aliyuncs.com/mac-img/image-20221024140312929.png)







