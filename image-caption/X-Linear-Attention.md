# 2020:X-Linear Attention Networks for Image Captioning

### 1.摘要

- CIDEr：132.8
- X线性注意力块同时利用空间和信道双线性注意分布来捕获输入单模态或多模态之间的二阶相互作用特征

![image-20221022164723296](https://leng-mypic.oss-cn-beijing.aliyuncs.com/mac-img/image-20221022164723296.png)

### 2.方法

双线性池化

- SE全称Squeeze-and-Excitation，它注重通道信息。

![image-20221022195437237](https://leng-mypic.oss-cn-beijing.aliyuncs.com/mac-img/image-20221022195437237.png)

- CBAM

![image-20221022195830308](https://leng-mypic.oss-cn-beijing.aliyuncs.com/mac-img/image-20221022195830308.png)

- 包含CAM和SAM两个模块

  - **CAM：通道数不变，压缩空间维度。**并行经过maxpool和avgpool，然后MLP 1/c和c倍，然后relu函数，在相加后sigmod得到channel attention。CAM与SEnet的不同之处是加了一个并行的最大池化层，提取到的高层特征更全面。

  ![image-20221022195914263](https://leng-mypic.oss-cn-beijing.aliyuncs.com/mac-img/image-20221022195914263.png)

  ![image-20221022200222457](https://leng-mypic.oss-cn-beijing.aliyuncs.com/mac-img/image-20221022200222457.png)

  - **SAM：空间维度不变，压缩通道数。**关注的是目标的位置信息。通过最大池化和平均池化得到两个1xHxW的特征图，然后拼接，通过7x7卷积变为通道为1的特征图，接一个sigmoid得到空间特征。

  ![image-20221022200304186](https://leng-mypic.oss-cn-beijing.aliyuncs.com/mac-img/image-20221022200304186.png)

  ![image-20221022200557053](https://leng-mypic.oss-cn-beijing.aliyuncs.com/mac-img/image-20221022200557053.png)

- 先实现Q和K的双线性池化，例如特征为7\*7\*512，49x512,加权内积，最终融合后特征矩阵512\*512，融合空间和通道信息求平均，Softmax归一化。
- ELU函数，负数还有一点值，RELU负数为0。

![image-20221022201420078](https://leng-mypic.oss-cn-beijing.aliyuncs.com/mac-img/image-20221022201420078.png)

### 3.X-LAN网络

![image-20221022205408010](https://leng-mypic.oss-cn-beijing.aliyuncs.com/mac-img/image-20221022205408010.png)

- 编码器1+3个特征提取，解码器1个解码，输入：当前输入字wt、全局图像特征v、隐藏状态ht-1、上下文量ct-1，联合特征和词语编码输入











