# U-net（代码已跑通）

- 一个u型结构网络![](https://leng-mypic.oss-cn-beijing.aliyuncs.com/mac-img/20220513134517.png)

- UNet是一个对称的网络结构，左侧为下采样，右侧为上采样。
- 按照功能可以将左侧的一系列下采样操作称为encoder，将右侧的一系列上采样操作称为decoder。
- Skip Connection中间四条灰色的平行线，Skip Connection就是在上采样的过程中，融合下采样过过程中的feature map。
- Skip Connection用到的融合的操作也很简单，就是将feature map的通道进行叠加，俗称Concat。

---

- 代码已跑通，使用简单医学数据集![](https://leng-mypic.oss-cn-beijing.aliyuncs.com/mac-img/20220513134744.png)

