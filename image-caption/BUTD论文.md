## Bottom-Up and Top-Down Attention for Image Captioning and Visual Question Answering，4000引用

- COCO数据集格式
- ![COCO_val2014_000000391895](https://leng-mypic.oss-cn-beijing.aliyuncs.com/image/COCO_val2014_000000391895.jpg)

![image-20221020142656928](https://leng-mypic.oss-cn-beijing.aliyuncs.com/image/image-20221020142656928.png)

- BUTD代码（https://github.com/ezeli/BUTD_model）

---

### 1.摘要

- 提出结合bottom-up和top-down的注意力机制
  - bottom-up机制基于Faster R-CNN提取图像区域，每个区域有一个相关的特征向量。
  - top-down机制确定特征的权重。
  - CIDEr、SPICE、BLEU-4:117.9、21.5、36.9 

---

### 2.介绍

- 在人类视觉系统中，注意力可以由**当前任务**确定**自上而下**的信号（例如图片中的某个物体）。也可以由**意外、新颖和显著刺激**来确定**自下而上**的信号。在文本中，将非视觉或特定于任务的上下文的注意力机制称为top-down，单纯的视觉前馈注意力机制被称为bottom-up。

- 传统使用的注意力机制都是top-down的，将上下文货图像表示为caption的输出，训练这些机制选择性的关注CNN的输出，但这些方法很少考虑如何确定受到关注的图像区域。如下图，注意力机制正常是在大小相同的区域上计算CNN特征，本文计算显著图像上的注意力。

  ![img](https://leng-mypic.oss-cn-beijing.aliyuncs.com/mac-img/v2-416764d0bf910466e0d7d2799ba53c72_1440w.jpg)

- 自下而上提出一组显著的图像区域，每个区域由卷积特征向量表示。自下而上使用FRCNN，使注意力表达的更加自然。从上到下的机制使用特定于任务的上下文来预测图像的注意力分布，然后将参与的特征向量计算为所有向量特征的加权平均值。

---

### 3.方法

- 给定一张图片$I$，将图像的k个特征集合$V={v_1,...,v_k},v_i \in R^D$作为输入，使每个图像都对显著区域进行编码。
  - 3.1介绍自下而上
  - 3.2介绍体系结构
  - 3.3介绍VQA模型

#### 3.1Bottom-Up Attention Model注意力模型

- RCNN：区域SS提出，特征提取，分类预测，边框回归。缺点：SS提取框大量重叠，提取特征冗余，速度慢
- Fast RCNN：一张图片生成2000候选区域SS，将整张图送入CNN，将候选区域映射到特征图上，一步就可以获得2000个区域的特征，RCNN要2000次。随机选64个，包括正负样本0.5。在进行RoI Pooling，分成7x7大小每个进行最大池化得到7x7特征矩阵，然后进行全联接得到分类结果和边界框结果。
- Faster RCNN端到端：RPN区域生成+FRCNN。输入得到特征图，使用RPN生成候选框，投影到特征图上，然后RoI，然后全连接层。每个滑动窗口中心点，生成9个anchor box，2\*9，4\*9参数,IoU设为0.7，最终剩下2k个，要256个anchor，正负样本1:1，anchor与GD的IoU超过0.7为正样本，小于0.3为负

![image-20221021144306079](https://leng-mypic.oss-cn-beijing.aliyuncs.com/mac-img/image-20221021144306079.png)

- 使用FRCNN提取图片中的感兴趣点，再利用ResNet-101提取特征，使用IoU删除重叠候选框，如果大于某个阈值，说明这两个框是一个物体，就删掉它。维度2048

<img src="https://leng-mypic.oss-cn-beijing.aliyuncs.com/mac-img/image-20221021104921977.png" alt="image-20221021104921977" style="zoom:50%;" />

- Resnet101，输入3\*224\*224，输出2048\*7\*7，faster rcnn损失函数是边框和分类四个数的loss，再额外加一个多类别损失组件去训练属性。

#### 3.2描述模型Captioning Model

![image-20221021160211023](https://leng-mypic.oss-cn-beijing.aliyuncs.com/mac-img/image-20221021160211023.png)

- 对于top-down，每个图像的特征进行加权，使用现有部分输出序列作为上下文
- 使用两个标准LSTM组成，$h_t = LSTM(x_t, h_{t−1})$，xt是LSTM输入向量，ht是LSTM输出向量

##### 3.2.1top-down attention LSTM

- 输入：上一步LSTM输出$h_{t-1}^2$，前k个图像特征平均池化![image-20221021153339085](https://leng-mypic.oss-cn-beijing.aliyuncs.com/mac-img/image-20221021153339085.png)，上时刻onehot编码的词语 将这三个cat，和ht-1输入。

##### 3.2.2Language LSTM

- 输入：第一个LSTM的图像注意力矩阵和ht输出cat作为输入，
- $y_{1:T}$表示一个单词序列$(y_i,...,y_T)$，在时间步t，可能输出单词的条件分布为：![image-20221021161156847](https://leng-mypic.oss-cn-beijing.aliyuncs.com/mac-img/image-20221021161156847.png)

##### 3.2.3目标，优化loss

- 交叉熵优化





