# Generative Adversarial Nets

- 生成器G：用于生成假的数据，尽可能的让生成的数据靠近原始数据
- 判别器D：用于坚定数据的真假
- G和D都用了MLP，训练简单、

---

![](https://leng-mypic.oss-cn-beijing.aliyuncs.com/mac-img/20220511131711.png)

- z是噪音数据，经过辨别器D辨别G（z），如果辨别成功则为0
- x是真实数据，经过辨别器D辨别G（x），如果辨别成功则为1。
- 尽可能的让辨别器犯错
- 判断此图是生成的，D(G(z))=0,log( 1 - D(G(Z)) ) = log1 = 0
- 判断此图是真实采样，D(G(Z)) = 1,  log( 1 - D(G(Z)) ) = log0 = -∞
  

![](https://leng-mypic.oss-cn-beijing.aliyuncs.com/mac-img/20220511135326.png)

- 噪声z映射到绿色高斯分布。
- 蓝色线判别器用于去判断真假数据的分布。
- 去优化绿色线尽可能的靠近真实数据黑色线。