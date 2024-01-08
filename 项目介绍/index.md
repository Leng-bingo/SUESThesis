### 已读论文列表

> 论文阅读步骤

- 第一遍：**论文标题、摘要、结论**。顺带看看论文中**重要的图和表**。花费十几分钟时间判断论文是否和自己的研究方向相同。

- 第二遍：确定论文值得读之后，可以快速的把整个论文**过一遍**，不需要知道所有的细节，但是需要了解重要的图和表，**知道每一个部分在干什么，圈出相关文献**。觉得文章太难理解的话，可以先看看论文中引用的其它文献，了解相关背景知识。

- 第三遍：最详细的一遍，**知道每一段和每一句话是什么意思**，在复现论文时再反复研读。

### 计算机视觉~~CNN

| 已学习 | 年份 | 名字                                                         | 简介                               | 引用                                                         |
| ------ | ---- | ------------------------------------------------------------ | ---------------------------------- | ------------------------------------------------------------ |
| ✅      | 2012 | [AlexNet](https://papers.nips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf) | 深度学习热潮的奠基作               | 73342 ([link](https://www.semanticscholar.org/paper/ImageNet-classification-with-deep-convolutional-Krizhevsky-Sutskever/abd1c342495432171beb7ca8fd9551ef13cbd0ff)) |
|        | 2014 | [VGG](https://arxiv.org/pdf/1409.1556.pdf)                   | 使用 3x3 卷积构造更深的网络        | 55856 ([link](https://www.semanticscholar.org/paper/Very-Deep-Convolutional-Networks-for-Large-Scale-Simonyan-Zisserman/eb42cf88027de515750f230b23b1a057dc782108)) |
|        | 2014 | [GoogleNet](https://arxiv.org/pdf/1409.4842.pdf)             | 使用并行架构构造更深的网络         | 26878 ([link](https://www.semanticscholar.org/paper/Going-deeper-with-convolutions-Szegedy-Liu/e15cf50aa89fee8535703b9f9512fca5bfc43327)) |
|   ✅   | 2015 | [ResNet](https://arxiv.org/pdf/1512.03385.pdf)               | 构建深层网络都要有的残差连接。     | 80816 ([link](https://www.semanticscholar.org/paper/Deep-Residual-Learning-for-Image-Recognition-He-Zhang/2c03df8b48bf3fa39054345bafabfeff15bfd11d)) |
|        | 2017 | [MobileNet](https://arxiv.org/pdf/1704.04861.pdf)            | 适合终端设备的小CNN                | 8695 ([link](https://www.semanticscholar.org/paper/MobileNets%3A-Efficient-Convolutional-Neural-Networks-Howard-Zhu/3647d6d0f151dc05626449ee09cc7bce55be497e)) |
|        | 2019 | [EfficientNet](https://arxiv.org/pdf/1905.11946.pdf)         | 通过架构搜索得到的CNN              | 3426 ([link](https://www.semanticscholar.org/paper/EfficientNet%3A-Rethinking-Model-Scaling-for-Neural-Tan-Le/4f2eda8077dc7a69bb2b4e0a1a086cf054adb3f9)) |
|        | 2021 | [Non-deep networks](https://arxiv.org/pdf/2110.07641.pdf)    | 让不深的网络也能在ImageNet刷到SOTA | 0 ([link](https://www.semanticscholar.org/paper/Non-deep-Networks-Goyal-Bochkovskiy/0d7f6086772079bc3e243b7b375a9ca1a517ba8b)) |



### 计算机视觉~~Transformer

| 已学习 | 年份 | 名字                                                     | 简介                       | 引用                                                         |
| ------ | ---- | -------------------------------------------------------- | -------------------------- | ------------------------------------------------------------ |
|        | 2020 | [ViT](https://arxiv.org/pdf/2010.11929.pdf)              | Transformer杀入CV界        | 1527 ([link](https://www.semanticscholar.org/paper/An-Image-is-Worth-16x16-Words%3A-Transformers-for-at-Dosovitskiy-Beyer/7b15fa1b8d413fbe14ef7a97f651f47f5aff3903)) |
|        | 2021 | [CLIP](https://openai.com/blog/clip/)                    | 图片和文本之间的对比学习   | 399 ([link](https://www.semanticscholar.org/paper/Learning-Transferable-Visual-Models-From-Natural-Radford-Kim/6f870f7f02a8c59c3e23f407f3ef00dd1dcf8fc4)) |
|        | 2021 | [Swin Transformer](https://arxiv.org/pdf/2103.14030.pdf) | 多层次的Vision Transformer | 384 ([link](https://www.semanticscholar.org/paper/Swin-Transformer%3A-Hierarchical-Vision-Transformer-Liu-Lin/c8b25fab5608c3e033d34b4483ec47e68ba109b7)) |
|        | 2021 | [MLP-Mixer](https://arxiv.org/pdf/2105.01601.pdf)        | 使用MLP替换self-attention  | 137 ([link](https://www.semanticscholar.org/paper/MLP-Mixer%3A-An-all-MLP-Architecture-for-Vision-Tolstikhin-Houlsby/2def61f556f9a5576ace08911496b7c7e4f970a4)) |
|        | 2021 | [MAE](https://arxiv.org/pdf/2111.06377.pdf)              | BERT的CV版                 | 4 ([link](https://www.semanticscholar.org/paper/Masked-Autoencoders-Are-Scalable-Vision-Learners-He-Chen/c1962a8cf364595ed2838a097e9aa7cd159d3118)) |



### 自然语言处理~~Transformer

| 已学习 | 年份 | 名字                                                         | 简介                                              | 引用                                                         |
| ------ | ---- | ------------------------------------------------------------ | ------------------------------------------------- | ------------------------------------------------------------ |
| ✅      | 2017 | [Transformer](https://arxiv.org/abs/1706.03762)              | 继MLP、CNN、RNN后的第四大类架构                   | 26029 ([link](https://www.semanticscholar.org/paper/Attention-is-All-you-Need-Vaswani-Shazeer/204e3073870fae3d05bcbc2f6a8e263d9b72e776)) |
|        | 2018 | [GPT](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf) | 使用 Transformer 解码器来做预训练                 | 2752 ([link](https://www.semanticscholar.org/paper/Improving-Language-Understanding-by-Generative-Radford-Narasimhan/cd18800a0fe0b668a1cc19f2ec95b5003d0a5035)) |
|        | 2018 | [BERT](https://arxiv.org/abs/1810.04805)                     | Transformer一统NLP的开始                          | 25340 ([link](https://www.semanticscholar.org/paper/BERT%3A-Pre-training-of-Deep-Bidirectional-for-Devlin-Chang/df2b0e26d0599ce3e70df8a9da02e51594e0e992)) |
|        | 2019 | [GPT-2](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) | 更大的 GPT 模型，朝着zero-shot learning迈了一大步 | 4534 ([link](https://www.semanticscholar.org/paper/Language-Models-are-Unsupervised-Multitask-Learners-Radford-Wu/9405cc0d6169988371b2755e573cc28650d14dfe)) |
|        | 2020 | [GPT-3](https://arxiv.org/abs/2005.14165)                    | 100倍更大的 GPT-2，few-shot learning效果显著      | 2548 ([link](https://www.semanticscholar.org/paper/Language-Models-are-Few-Shot-Learners-Brown-Mann/6b85b63579a916f705a8e10a49bd8d849d91b1fc)) |

### 计算机视觉~~GAN

| 已学习 | 年份 | 名字                                                         | 简介               | 引用                                                         |
| ------ | ---- | ------------------------------------------------------------ | ------------------ | ------------------------------------------------------------ |
| ✅      | 2014 | [GAN](https://papers.nips.cc/paper/2014/file/5ca3e9b122f61f8f06494c97b1afccf3-Paper.pdf) | 生成模型的开创工作 | 26024 ([link](https://www.semanticscholar.org/paper/Generative-Adversarial-Nets-Goodfellow-Pouget-Abadie/54e325aee6b2d476bbbb88615ac15e251c6e8214)) |
|   ✅   | 2015 | [DCGAN](https://arxiv.org/pdf/1511.06434.pdf)                | 使用CNN的GAN       | 9022 ([link](https://www.semanticscholar.org/paper/Unsupervised-Representation-Learning-with-Deep-Radford-Metz/8388f1be26329fa45e5807e968a641ce170ea078)) |
| ✅     | 2016 | [pix2pix](https://arxiv.org/pdf/1611.07004.pdf)              |                    | 9752 ([link](https://www.semanticscholar.org/paper/Image-to-Image-Translation-with-Conditional-Isola-Zhu/8acbe90d5b852dadea7810345451a99608ee54c7)) |
|        | 2016 | [SRGAN](https://arxiv.org/pdf/1609.04802.pdf)                | 图片超分辨率       | 5524 ([link](https://www.semanticscholar.org/paper/Photo-Realistic-Single-Image-Super-Resolution-Using-Ledig-Theis/df0c54fe61f0ffb9f0e36a17c2038d9a1964cba3)) |
|        | 2017 | [WGAN](https://arxiv.org/abs/1701.07875)                     | 训练更加容易       | 2620 ([link](https://www.semanticscholar.org/paper/Wasserstein-GAN-Arjovsky-Chintala/2f85b7376769473d2bed56f855f115e23d727094)) |
|        | 2017 | [CycleGAN](https://arxiv.org/abs/1703.10593)                 |                    | 3401 ([link](https://www.semanticscholar.org/paper/Unpaired-Image-to-Image-Translation-Using-Networks-Zhu-Park/c43d954cf8133e6254499f3d68e45218067e4941)) |
|        | 2018 | [StyleGAN](https://arxiv.org/abs/1812.04948)                 |                    | 2708 ([link](https://www.semanticscholar.org/paper/A-Style-Based-Generator-Architecture-for-Generative-Karras-Laine/ceb2ebef0b41e31c1a21b28c2734123900c005e2)) |
|        | 2019 | [StyleGAN2](https://arxiv.org/pdf/1912.04958.pdf)            |                    | 1096 ([link](https://www.semanticscholar.org/paper/Analyzing-and-Improving-the-Image-Quality-of-Karras-Laine/f3e3d1f86a534a3654d0ee263142e44f4e2c61e9)) |
|        | 2021 | [StyleGAN3](https://arxiv.org/pdf/2106.12423.pdf)            |                    | 23 ([link](https://www.semanticscholar.org/paper/Alias-Free-Generative-Adversarial-Networks-Karras-Aittala/c1ff08b59f00c44f34dfdde55cd53370733a2c19)) |

### 图神经网络

| 已学习 | 年份 | 名字                                                  | 简介            | 引用                                                         |
| ------ | ---- | ----------------------------------------------------- | --------------- | ------------------------------------------------------------ |
| ✅      | 2021 | [图神经网络介绍](https://distill.pub/2021/gnn-intro/) | GNN的可视化介绍 | 7 ([link](https://www.semanticscholar.org/paper/A-Gentle-Introduction-to-Graph-Neural-Networks-Sánchez-Lengeling-Reif/2c0e0440882a42be752268d0b64243243d752a74)) |

### 计算机视觉~~对比学习

| 已学习 | 年份 | 名字                                               | 简介                                                 | 引用                                                         |
| ------ | ---- | -------------------------------------------------- | ---------------------------------------------------- | ------------------------------------------------------------ |
|        | 2018 | [InstDisc](https://arxiv.org/pdf/1805.01978.pdf)   | 提出实例判别和memory bank做对比学习                  | 885 ([link](https://www.semanticscholar.org/paper/Unsupervised-Feature-Learning-via-Non-parametric-Wu-Xiong/155b7782dbd713982a4133df3aee7adfd0b6b304)) |
|        | 2018 | [CPC](https://arxiv.org/pdf/1807.03748.pdf)        | 对比预测编码，图像语音文本强化学习全都能做           | 2187 ([link](https://www.semanticscholar.org/paper/Representation-Learning-with-Contrastive-Predictive-Oord-Li/b227f3e4c0dc96e5ac5426b85485a70f2175a205)) |
|        | 2019 | [InvaSpread](https://arxiv.org/pdf/1904.03436.pdf) | 一个编码器的端到端对比学习                           | 223 ([link](https://www.semanticscholar.org/paper/Unsupervised-Embedding-Learning-via-Invariant-and-Ye-Zhang/e4bde6fe33b6c2cf9d1647ac0b041f7d1ba29c5b)) |
|        | 2019 | [CMC](https://arxiv.org/pdf/1906.05849.pdf)        | 多视角下的对比学习                                   | 780 ([link](https://www.semanticscholar.org/paper/Contrastive-Multiview-Coding-Tian-Krishnan/97f4d09175705be4677d675fa27e55defac44800)) |
|        | 2019 | [MoCov1](https://arxiv.org/pdf/1911.05722.pdf)     | 无监督训练效果也很好                                 | 2011 ([link](https://www.semanticscholar.org/paper/Momentum-Contrast-for-Unsupervised-Visual-Learning-He-Fan/ec46830a4b275fd01d4de82bffcabe6da086128f)) |
|        | 2020 | [SimCLRv1](https://arxiv.org/pdf/2002.05709.pdf)   | 简单的对比学习 (数据增强 + MLP head + 大batch训练久) | 2958 ([link](https://www.semanticscholar.org/paper/A-Simple-Framework-for-Contrastive-Learning-of-Chen-Kornblith/34733eaf66007516347a40ad5d9bbe1cc9dacb6b)) |
|        | 2020 | [MoCov2](https://arxiv.org/pdf/2003.04297.pdf)     | MoCov1 + improvements from SimCLRv1                  | 725 ([link](https://www.semanticscholar.org/paper/Improved-Baselines-with-Momentum-Contrastive-Chen-Fan/a1b8a8df281bbaec148a897927a49ea47ea31515)) |
|        | 2020 | [SimCLRv2](https://arxiv.org/pdf/2006.10029.pdf)   | 大的自监督预训练模型很适合做半监督学习               | 526 ([link](https://www.semanticscholar.org/paper/Big-Self-Supervised-Models-are-Strong-Learners-Chen-Kornblith/3e7f5f4382ac6f9c4fef6197dd21abf74456acd1)) |
|        | 2020 | [BYOL](https://arxiv.org/pdf/2006.07733.pdf)       | 不需要负样本的对比学习                               | 932 ([link](https://www.semanticscholar.org/paper/Bootstrap-Your-Own-Latent%3A-A-New-Approach-to-Grill-Strub/38f93092ece8eee9771e61c1edaf11b1293cae1b)) |
|        | 2020 | [SWaV](https://arxiv.org/pdf/2006.09882.pdf)       | 聚类对比学习                                         | 593 ([link](https://www.semanticscholar.org/paper/Unsupervised-Learning-of-Visual-Features-by-Cluster-Caron-Misra/10161d83d29fc968c4612c9e9e2b61a2fc25842e)) |
|        | 2020 | [SimSiam](https://arxiv.org/pdf/2011.10566.pdf)    | 化繁为简的孪生表征学习                               | 403 ([link](https://www.semanticscholar.org/paper/Exploring-Simple-Siamese-Representation-Learning-Chen-He/0e23d2f14e7e56e81538f4a63e11689d8ac1eb9d)) |
|        | 2021 | [MoCov3](https://arxiv.org/pdf/2104.02057.pdf)     | 如何更稳定的自监督训练ViT                            | 96 ([link](https://www.semanticscholar.org/paper/An-Empirical-Study-of-Training-Self-Supervised-Chen-Xie/739ceacfafb1c4eaa17509351b647c773270b3ae)) |
|        | 2021 | [DINO](https://arxiv.org/pdf/2104.14294.pdf)       | transformer加自监督在视觉也很香                      | 200 ([link](https://www.semanticscholar.org/paper/Emerging-Properties-in-Self-Supervised-Vision-Caron-Touvron/ad4a0938c48e61b7827869e4ac3baffd0aefab35)) |