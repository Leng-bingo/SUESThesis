# Attention on Attention for Image Captioning 500引用

### 1.摘要

- 提出Attention on Attention
  - 首先利用注意力结果和当前上下文生成一个**信息向量**和**注意力门**，然后将他俩相乘得到另一个注意力被叫做**参与信息**。应用AoA在编码器和解码器。
  - CIDEr、BLEU-4:126.9、39.4

### 2.方法

#### 2.1 AoA

- 信息变量 I，注意力门 G，都取决于注意力结果和当前上下文q

  ![image-20221022154624175](https://leng-mypic.oss-cn-beijing.aliyuncs.com/mac-img/image-20221022154624175.png)

  - v是注意力结果，然后g和i相乘，得到文中提出的新注意力

![image-20221022155001376](https://leng-mypic.oss-cn-beijing.aliyuncs.com/mac-img/image-20221022155001376.png)

  - 整体流程：

![image-20221024154522436](https://leng-mypic.oss-cn-beijing.aliyuncs.com/mac-img/image-20221024154522436.png)

![image-20221022155244625](https://leng-mypic.oss-cn-beijing.aliyuncs.com/mac-img/image-20221022155244625.png)

#### 2.2AoANet for image captioning

- 编码器解码器都用AoA

##### 2.2.1编码器

- 利用CNN或RCNN提取特征A=「a1，a2，...，ak」，先构建一个包含AoA的网络，再送往解码器。

  ![](https://leng-mypic.oss-cn-beijing.aliyuncs.com/mac-img/image-20221022163221569.png)

  ![image-20221022163246444](https://leng-mypic.oss-cn-beijing.aliyuncs.com/mac-img/image-20221022163246444.png)

![image-20221022155841949](https://leng-mypic.oss-cn-beijing.aliyuncs.com/mac-img/image-20221022155841949.png)

##### 2.2.2解码器

![image-20221022162042982](https://leng-mypic.oss-cn-beijing.aliyuncs.com/mac-img/image-20221022162042982.png)

- LSTM输入：有三个变量 a,ct−1,WeΠt，其中 a=1k∑i=1kai ，A={a1,a2⋯ak} 表示通过Faster-RCNN中RoIPooling层之后的图像区域特征， ct−1表示AoA模块的输出，WeΠt表示单词的词向量。![image-20221022163001194](https://leng-mypic.oss-cn-beijing.aliyuncs.com/mac-img/image-20221022163001194.png)









