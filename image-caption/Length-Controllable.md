# ECCV2020｜ Length-Controllable Image Captioning

### 1.摘要

- 生成长度可控的图像描述（更好比）

![image-20221024164958875](https://leng-mypic.oss-cn-beijing.aliyuncs.com/mac-img/image-20221024164958875.png)

- 基于迭代的长度可控非回归图像描述模型，非自回归模型继承BERT，对输入潜入层进行修改，加入了图像特征和文本长度信息叫做length-aware BERT LaBERT。设计了非自回归解码器
- 设计了一种非回归解码器，长度可控
- [1,9]、[10,14]、[15,19]和[20,25]

### 2.方法

#### 2.1将长度信息结合到自回归模型中

- 标题S从i=1到L，el长度信息，词信息ew，位置向量ep

- ![image-20221024172315443](https://leng-mypic.oss-cn-beijing.aliyuncs.com/mac-img/image-20221024172315443.png)
- 长度感知自回归解码器
- 直接将上述单词表示公式替换到AoA中，模型的效果就提升了。

#### 2.2非自回归模型

- 修改了BERT的潜入层，将图像信息和长度信息结合在一起，利用目标检测从图像I中检测出M个对象，![image-20221024174124787](https://leng-mypic.oss-cn-beijing.aliyuncs.com/mac-img/image-20221024174124787.png)，还有区域特征![image-20221024174216713](https://leng-mypic.oss-cn-beijing.aliyuncs.com/mac-img/image-20221024174216713.png)，分类概率Fc，位置特征Fl

![image-20221024175041897](https://leng-mypic.oss-cn-beijing.aliyuncs.com/mac-img/image-20221024175041897.png)

- 训练
  - 给定图像标题S*，首先确定他的长度级别，然后用【EOS】填充到长度的最大范围，然后用【MASK】随机替换描述中的m个单词构造输入序列S，最后模型根据输入的图片信息和序列S的信息来预测被替换掉的真实单词。
  - ![image-20221025103409340](https://leng-mypic.oss-cn-beijing.aliyuncs.com/mac-img/image-20221025103409340.png)
  - 步骤：t=1时刻，将标题s初始化为连续的最大范围MASK，然后输入图像和文本，预测S中每个位置上的词汇表的概率分布pi。t=2，之后在选择置信度最低的n=T-t/T*Lhigh（T总迭代次数，t当前迭代次数，t越大，n越小）个词语替换为mask，再次进行解码，得到一个更好的句子，更新之心度
  - 非自回归的方法计算复杂度与T相关而和生成描述的长度无关，降低了生成长描述的计算复杂度，并且还能在之后的步骤中修改早期步骤中犯的错误，这在自回归的方法中是不可行的。