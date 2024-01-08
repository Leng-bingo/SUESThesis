# GAN衍生网络

- CGAN,
  - 多加了一个生成器和辨别器中的条件，可以加入生成什么图像的条件

![](https://leng-mypic.oss-cn-beijing.aliyuncs.com/mac-img/20220511175904.png)

- DCGAN：
  - 把GAN中的pooling层全都用卷积层代替，把滑动窗口改为2，就也可做到减小图片的大小，因为pooling的下采样是四个最大的选一个出来，会损失特征。
  - 生成器G和判别器D都要用BN层，减少过拟合
  - 把全连接层去掉，用券卷积层代替
  - 生成器除了输出层，激活函数统一使用ReLU。输出层用Tanh。
  - 判别器所有激活层统一都用LeakyReLU。