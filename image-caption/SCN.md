# 2017CVPR：Semantic Compositional Networks for Visual Captioning

> 语义组合网络SCN

### 1.摘要

- 提出一种语义组合网络SCN，从图像中检测语义概念（也就是标签），利用每个标签的概率组成LSTM中的参数。SCN将LSTM的每个权重矩阵扩展为标签相关权重矩阵集合，每个lstm参数矩阵用于生成每个标签对应图像相关的概率。
- CIDEr、BLEU-4:100.3、21.5、33.1 

### 2.方法：SCN模型

#### 2.1普通的RNN

- 图像I，标题为$X={x_1,...,x_T}$，每个词语都是one-hot编码，特征向量$V(I)$提取自CNN特征，给定图像特征$v$，描述X的概率为：![image-20221021191440332](https://leng-mypic.oss-cn-beijing.aliyuncs.com/mac-img/image-20221021191440332.png)
- RNN公式为如下，输入上一个单词，上一个输出，和特征向量
- ![image-20221021191822991](https://leng-mypic.oss-cn-beijing.aliyuncs.com/mac-img/image-20221021191822991.png)

#### 2.2语义概念检测

- i：第几张图像
- k：第几个单词
- s：语义特征

- 语义概念为标签的检测，为了从一幅图中检测出标签，首先从训练集选择标签，选择最常出现的前k(1000)个词语。$y_i$是第i个图像的标签，$y_{ik}$，k=1是第一张图像第一个单词是否出现我们把出现在描述中的单词标注为1，否则为0。$v_i$和$s_i$表示第i个图像特征向量和语义特征向量。损失函数为：![image-20221021214910889](https://leng-mypic.oss-cn-beijing.aliyuncs.com/mac-img/image-20221021214910889-20221021215031334.png)。$s_i$是k维向量。

#### 2.3 SCN-RNN

- SCN将RNN的每个权重变为与标签相关的权重矩阵集合，主观的取决于标签存在于图像中的概念。$h_t = σ(W(s)x_{t−1} +U(s)h_{t−1} + z) $。W和U是依赖于语义标签的权重。

- 对于语义向量s，我们定义两个权重矩阵：Wt和Ut

  ![image-20221022141014048](https://leng-mypic.oss-cn-beijing.aliyuncs.com/mac-img/image-20221022141014048.png)

- 然后把语义向量与输入和隐层输出结合。

- ![image-20221024145517312](https://leng-mypic.oss-cn-beijing.aliyuncs.com/mac-img/image-20221024145517312.png)























