- CVPR 2015：Show and Tell（https://arxiv.org/pdf/1411.4555.pdf），CNN+LSTM
- ICML 2015：Show, Attend and Tell（https://arxiv.org/pdf/1502.03044v1.pdf），CNN+注意力机制+LSTM![image-20221014104322070](https://leng-mypic.oss-cn-beijing.aliyuncs.com/image/image-20221014104322070.png)
  - 图像经过卷积网络生成图像特征；
  - 根据系统前次状态，决定现在该看哪儿；
  - 用关注点对特征加权，获得当前上下文；
  - 借鉴前次系统状态，由上下文计算系统隐变量；
  - 由隐变量直接推导出当前单词。
- CVPR 2018：Bottom-Up and Top-Down Attention for Image Captioning and Visual Question Answering（https://arxiv.org/pdf/1707.07998.pdf），Code（https://github.com/peteanderson80/bottom-up-attention），Fast R-CNN+双层LSTM多模态嵌入
  - 通过Faster R-CNN提取感兴趣特征，其次计算每个特征的注意力权重
  - VQA任务：同时输入image和question，回答问题
  - top-down：利用cnn检测出所有目标，提取每个感兴趣区域的特征向量
  - 这里paper将非视觉或特定任务上下文的注意力机制称为top-down，视觉前馈注意力机制称为bottom-up。
  - VQA模型![image-20221014111729975](https://leng-mypic.oss-cn-beijing.aliyuncs.com/image/image-20221014111729975.png)
  
    - Caption模型：双层LSTM，第一层为top-down视觉注意力模型，判断每个特征区域的重要性。第二层为语言模型，将图像特征解码成文字描述。<img src="https://leng-mypic.oss-cn-beijing.aliyuncs.com/image/image-20221014112650272.png" alt="image-20221014112650272" style="zoom: 67%;" />
- CVPR 2016：Knowing when to look: Adaptive attention via a visual sentinel for image captioning（https://click.endnote.com/viewer?doi=10.48550%2Farxiv.1612.01887&token=WzM2ODUwMzUsIjEwLjQ4NTUwL2FyeGl2LjE2MTIuMDE4ODciXQ.cedK5T37mVcBgl7aD32qta4g-mc），Code（ https://github.com/jiasenlu/AdaptiveAttention），
  - 提出了**自适应注意力机制**模型 (Adaptive Attention)，该方法提出了 “视觉哨兵” (Visual Sentinel)的概念，即在LSTM 的隐藏层加入一个视觉哨兵向量，用来控制对非视觉词的生成，如介词、量词等。该方法使模型不仅依赖于图像信息，还依赖于句子的语义信息， 从而生成更加详细的描述句。![[Image Caption系列(2)] Knowing when to look(Adaptive attention)论文阅读笔记_第1张图片](https://leng-mypic.oss-cn-beijing.aliyuncs.com/image/397060f90db04c35a243969f5d9465e9.jpg)
  - 带有视觉哨兵 （Visual Sentinel）的自适应注意力模型
  - 非视觉词语，介词the，of不需要视觉信息，引进一个参数，控制attention中视觉信息和历史信息的比重
  - 改进了空间注意力模型
    - 原来的是：提取的注意力矩阵与前一时刻的输出$H_{t-1}$一起送入LSTM
    - 现在：直接使用$H_t$当前时刻的输出计算注意力
- CVPR 2017： SCA-CNN: Spatial and Channel-wise Attention in Convolutional Networks for Image Captioning。CNN+通道注意力+RNN
  - 这篇文章从CNN入手，分析CNN的特性，包括其空间性、多通道和多层级。
  - 提出了spatial attention和channel wise attention两种attention操作
    - spatial attention就是常规的方法，以当前feature map的每个像素点为单位，对每个像素点都配一个权重值，这个权重值是个二维矩阵
    - channel wise attention则是以feature map为单位，对每个channel都配一个权重值，因此这个权重值是一个向量。
- CVPR 2018：《Auto-encoding scene graphs for image captioning》（https://click.endnote.com/viewer?doi=10.48550%2Farxiv.1812.02378&token=WzM2ODUwMzUsIjEwLjQ4NTUwL2FyeGl2LjE4MTIuMDIzNzgiXQ.Yw65hAEYF_3pwhE6nJh8dWGaQbM），CNN+SGAE+RNN
  - 在编码器解码器结构潜入了一个自己提出的模型，SGAE（Scene Graph Auto-Encoder）![img](https://leng-mypic.oss-cn-beijing.aliyuncs.com/image/1334881-20220322160418699-996606573.png)
  - 提出在注意力机制中嵌入场景图结构 (SGAE)作为句子重建网络,将图像通过检测网络得到的物体、属性、关系等生成场景图,再通过图卷积网络处理作为输入，送到已经利用SGAE结构预训练好的与编码器-解码器模型共享的字典当中，对产生的词向量进行转换重建，从而利用语料库实现了更加接近人类语言的图像描述。![在这里插入图片描述](https://leng-mypic.oss-cn-beijing.aliyuncs.com/image/1470684-20210122015852439-797441497.png)
  - MGCN：首先，使用了Faster-RCNN 对图像做目标检测提取feature信息， 又利用了MOTIFS模型提取了检测目标之间的关系。利用本文自己提出了一个fc-ReLU-fc-softmax 结构提取目标的属性信息。然后将得到各类型feature和各自的label 进行了融合。
- 这个不错：CVPR 2020：《More Grounded Image Captioning by Distilling Image-Text Matching Mode》，（https://click.endnote.com/viewer?doi=10.48550%2Farxiv.2004.00390&token=WzM2ODUwMzUsIjEwLjQ4NTUwL2FyeGl2LjIwMDQuMDAzOTAiXQ.uSwB-nFURgvgUQOZernoyf7K8Ao）利用知识蒸馏，先利用带标签的图像数据训练出一个模型，再利用这个模型训练图像理解的模型。参考知乎（https://zhuanlan.zhihu.com/p/358704852）
  - 在 Up-Down的 基 础 上 融 入 图 文 匹 配 模 型 (Stacked Cross Attention Network, SCAN)ECCV 2018《Stacked Cross Attention for Image-Text Matching》（https://arxiv.org/pdf/1803.08024v2.pdf），code（https://github.com/kuanghuei/SCAN），对注意力机制 的训练过程进行弱监督,并且使用自关键序列训练算法(Self- Critical sequence training, SCST)《Self-critical sequence training for image captioning》对图像和文本的匹配程度进行强化学习，增强了注意力机制对单词和图像区域的对应能力，从而生成更加合理的描述
    - SCAN做的事情![image-20221015101427990](https://leng-mypic.oss-cn-beijing.aliyuncs.com/image/image-20221015101427990.png)
  - Grounded Image Captioning：对给定的一张图片，模型在生成该图片描述时，生成词语的attention map要集中在正确对应的图中物体上。一般来说要实现这一目标，通常的做法是引入额外的region-word标注数据作为强监督信号，从而使得模型有更强的grounding能力。而本文作者希望能够在不使用region-word标注数据的情况下，达到甚至超越使用region-word标注数据的模型效果。
  - 知识蒸馏（knowledge distillation），可看（https://blog.csdn.net/charles_zhang_/article/details/123627334）是模型压缩的一种常用的方法，不同于模型压缩中的剪枝和量化，知识蒸馏是通过构建一个轻量化的小模型，利用性能更好的大模型的监督信息，来训练这个小模型，以期达到更好的性能和精度。最早是由Hinton在2015年首次提出并应用在分类任务上面，这个大模型我们称之为teacher（教师模型），小模型我们称之为Student（学生模型）。来自Teacher模型输出的监督信息称之为knowledge(知识)，而student学习迁移来自teacher的监督信息的过程称之为Distillation(蒸馏)。
  - 模型1：Image-Text Matching Model![image-20221015101739441](https://leng-mypic.oss-cn-beijing.aliyuncs.com/image/image-20221015101739441.png)
    - 图片文本匹配模型，主要使用SCAN结构，将GD文本进行双向GRU编码，得到特征；图片通过fastercnn提取特征，通过一个线性层，映射到和句子同一个向量空间中。然后计算两者的相似度矩阵并进行归一化处理。然后利用刚刚计算得到的相似度矩阵，求相似度矩阵对图片特征的attention，得到at。再对et和at计算它们的matching score。**注意到，本文对SCAN模型的最大修改即，在计算matching score时，只计入名词对应的matching score，而非名词不予计入。**这从直观上来解释的话，是因为在将图上物体和文本单词进行匹配时，物体主要匹配的为名词或名词短语。由此可以得到模型的优化目标为：最大化groud truth图片描述对的matching score，最小化和图片最相近的非ground truth描述的matching score，最小化和描述最相近的非groud truth图片的matching score。具体公式如下：![image-20221015102520662](https://leng-mypic.oss-cn-beijing.aliyuncs.com/image/image-20221015102520662.png)
  - 模型2：标题生成Caption Generato
    - 这里采用的是一个目前较常见的Up-Down模型，它主要的结构就是一个双层的LSTM模型。第一层为带Attetion的LSTM模型，它会先把文本信息经过LSTM得到一个隐向量，求它对图片信息的attention，然后把输出的向量和隐向量作为第二层language LSTM的输入，最终的输出经过softmax即为预测的词概率分布。具体公式如下：![image-20221015102437256](https://leng-mypic.oss-cn-beijing.aliyuncs.com/image/image-20221015102437256.png)![image-20221015103855693](https://leng-mypic.oss-cn-beijing.aliyuncs.com/image/image-20221015103855693.png)
- ECCV 2010，**Every Picture Tells a Story: Generating Sentences from Images**
- - 将问题简化为构建三元组的问题，变为预测三元组是什么。
  - 利用基础图像寻找HOG特征点，计算相似度计算语义距离。没有用深度学习方法。
- CVPR 2011，**Baby Talk: Understanding and Generating Simple Image Descriptions**
  - **基本方法：**传统统计学方法：利用图像处理的一些算子提取出图像的特征，先提取目标，再体取关系，再利用CRF（构建名词、形容词和介词）（随机条件场）计算概率；经过SVM分类等等得到图像中可能存在的目标object。根据提取出的object以及它们的属性利用CRF来恢复成对图像的描述。参考（https://zhuanlan.zhihu.com/p/240716105）
  - ![img](https://imgconvert.csdnimg.cn/aHR0cHM6Ly9pLmxvbGkubmV0LzIwMTkvMDcvMTUvNWQyYzMwNjcyMTkyNzU5MDk4LnBuZw?x-oss-process=image/format,png)
- CVPR 2015 **Deep Visual-Semantic Alignments for Generating Image Descriptions**
  - RCNN+BRNN（双线RNN），将图片和描述转为h维向量，然后计算各个单词在图片中的得分
- CVPR 2015 **Long-term Recurrent Convolutional Networks forVisual Recognition and Description**
  - 提出LRCN结构结合CNN和LSTM。**其中的输入可以是单帧图片或者是视频中的序列信息，同时网络的数据也可以是单个预测值或序列预测值**，这使得该网络可以适应多种任务处理。
- ICCV 2015：**Guiding Long-Short Term Memory for Image Caption Generation**
  - **基本方法：提出了**gLSTM，在LSTM基础上引入额外的语义信息向量G；图像采用标准CCA计算特征（NLTK工具箱做CCA），句子计算TF- IDF 加权的BoW 特征，分别将CCA的结果、中间量以及原图像作为guidance（gLSTM的额外输入）。
- ICCV 2015：**Learning like a Child: Fast Novel Visual Concept Learning from Sentence Descriptions of Images**，融合文本和图像，，![image-20221015121625303](https://leng-mypic.oss-cn-beijing.aliyuncs.com/image/image-20221015121625303.png)

  - **基本方法：两阶段方法，**VGG去掉SoftMax作为视觉信息，分类labels转换为**one-hot**作为文本信息，**文本信息通过embedding编码后与视觉信息融合**，输入到m-RNN或LSTM，解码采用两步embedding：先解码到中间层，后从中间层解码到输出，后一部分embedding可与编码层共用权重以减少参数量。
- CVPR 2019：**Show, Control and Tell:A Framework for Generating Controllable and Grounded Captions**，可控制的图像语义理解，双层LSTM，中间包含两个控制模块，切换区域和视觉词还是文本词，自适应RNN输出，代码（https://github.com/aimagelab/show-control-and-tell）
  - 给定一个图片I和一个序列集合R作为输入
    
  - 提出新的带序列的数据集，包含方框的标注
    
  - ![image-20221015123544408](https://leng-mypic.oss-cn-beijing.aliyuncs.com/image/image-20221015123544408.png)
  - ![image-20221015123732253](https://leng-mypic.oss-cn-beijing.aliyuncs.com/image/image-20221015123732253.png)
- CVPR 2017：Self-critical Sequence Training for Image Captioning

  - 本文提出了一种新的序列训练方法，称之为 self-critical sequence training (SCST)，并证明 SCST 可以显着提高图像描述系统的性能。 SCST 是一种强化算法，它不是估计 reward ，而是使用了自己在测试时生成的句子作为baseline。sample 时，那些比baseline好的句子就会获得正的权重，差的句子就会被抑制。
- CVPR 2019：**Self-critical n-step Training for Image Captioning**，强化学习

  - 使用n个时间步的累计奖赏代替交叉熵损失函数来评价
- 16.CVPR 2019：**Look Back and Predict Forward in Image Captioning**，cnn+三个lstm层
  - 提出LBPF模型，整合过去的视觉信息和未来的语言信息。改进attention机制，当前时刻attention权重由图像和LSTM隐层状态共同生成，attention模块为LSTM1
  - **强化学习：**在CIDEr分数上强化学习优化。![image-20221015133140432](https://leng-mypic.oss-cn-beijing.aliyuncs.com/image/image-20221015133140432.png)
  - LB 方法将上一个时间步的注意力值输入到当前注意力模块的输入中
  - PF 方法在一个时间步内生成两个下一个单词，它利用语言连贯性并整合未来的信息，在推理阶段，将生成的两个概率组合在一起以预测当前单词。

- 17.CVPR 2019：***MSCap: Multi-Style Image Captioning with Unpaired Stylized Text\***，利用**GAN**，生成多种风格的字幕![image-20221015133924797](https://leng-mypic.oss-cn-beijing.aliyuncs.com/image/image-20221015133924797.png)
  - G：输入图像和风格，生成对应风格的文本
  - D：输入真实文本和生成文本，判断真假
- 18.CVPR 2019：**Intention Oriented Image Captions with Guiding Objects**
  - 设置了两个LSTM，这算是本文最大的创新点。选取句子中一个词作为初始点，LSTM-L预测该词左边内容，LSTM-R预测右边内容。
- 19.CVPR 2019：**Fast, Diverse and Accurate Image Captioning Guided By Part-of-Speech**
  - 提出了一种POS（Part-of-Speech）算法来取代了beam search

- 20.CVPR 2019**Dense Relational Captioning:Triple-Stream Networks for Relationship-Based Captioning**
  - 密集字幕
  - ![image-20221015135829373](https://leng-mypic.oss-cn-beijing.aliyuncs.com/image/image-20221015135829373.png)
  - 利用RPN生成图像框，对每一个框生成文本，利用POS
  - 提取出主语谓语对象三个信息进行三个LSTM，进行整合

- 21.CVPR 2019：Unsupervised Image Captioning
  - 无监督，仅仅需要一个图像集、一个句子语料库和一个视觉概念检测器。
  - 通过利用视觉概念检测器，我们为每个图像生成一个伪标题，并使用伪图像-句子对初始化图像字幕模型。
    - 图像编码器CNN：将给定的图像编码生成一个特征表示，基于还特征表示，生成器输出一个句子来描述图像；
  - 鉴别器：用于区分标题是由模型生成的还是由句子语料库生成的；
  - 生成器和鉴别器：以不同的顺序耦合来执行图像和句子重建。
  - 由判别器判断生成器生成句子的合理性作为奖励+判别器判断的结果返回给生成器生成句子作为奖励+判别器加一层FC生成图像特征作为奖励，三种奖励（损失）共同优化生成器。
- 22。CVPR 2019：**Context and Attribute Grounded Dense Captioning**
  - 密集字幕
    - 在以往只关注框的基础上，加入关注邻域，使内容更加饱满
    - 还提出了一个属性监督机制，为了使生成的caption更符合语言特性
  - 提取特征为local feature，邻域特征为目标框周围框所有特征图的加权求和。引入CFE生成全局、局部以及相邻三个方面的特征线索构成多尺度上下文线索
  - 然后送入多个LSTM![image-20221015153049330](https://leng-mypic.oss-cn-beijing.aliyuncs.com/image/image-20221015153049330.png)

- 23.CVPR 2018：**Neural Baby Talk 2018**，2011年升级版；Code（https://github.com/jiasenlu/NeuralBabyTalk）；Faster-RCNN，注意力，lstm
  - 分为两个阶段
    - 生成有空缺的模板句：使用fast-RCNN检测N个目标区域，在预测单词时，同样分成visual word和context word，context word关联一个虚拟的区域，计算这N+1个区域的概率向量，如果是context word对应的区域概率最大就是和目标无关的单词，直接预测，若是和目标相关的单词则留出空缺。
    - 使用图像中的目标填上空缺：对目标词进行精细化处理才能填入句子，单复数和更精细的种类，直接使用MLP就可以得到，最终得到句子。
  - fastrcnn。
  
- 24.CVPR 2020：**Say As You Wish: Fine-grained Control of Image Caption Generation with Abstract Scene Graphs**，细粒度控制，更加详细。Code（https://github.com/cshizhe/asg2cap）
- 自己构建了数据集，三种关系有关主动和被动![image-20221015201606942](https://leng-mypic.oss-cn-beijing.aliyuncs.com/image/image-20221015201606942.png)
  
- 提出抽象场景图（**Abstract Scene Graph, ASG** ），可以通过图结构，同时控制所希望表达的物体、属性和关系。![img](https://leng-mypic.oss-cn-beijing.aliyuncs.com/image/v2-ffd358ee82648e00f7c3e68c794aa830_1440w.webp)
  
- ASG 是一个包含三类抽象节点的有向图，这三类抽象节点分别代表用户希望描述的物体 ( object ) 、属性 ( attribute ) 和关系 ( relationship ) ，每个抽象节点在图中有具体区域的定位，但却不需要任何具体语义标签。因为ASG不需要任何语义识别，它可以方便地由用户限定或自动生成。
  
- ASG2Caption模型
  
  - 是一个编码-解码网络![img](https://leng-mypic.oss-cn-beijing.aliyuncs.com/image/v2-108877a8fefc6bbbb17aff33492532ac_1440w.webp)
    - 角色感知节点嵌入模块（Role-aware Node Embedding.）：对于给定的image和ASG, 首先经过Role-aware Graph Encoder (角色感知编码器) 进行编码, 这部分通过role-aware node embedding (角色感知节点嵌入) 和MR-GCN (多关系-图卷积神经网络) 实现. 角色感知节点嵌入用于区分每个节点的意图，文中给出了详细的计算过程, 使用MR-GCN是为了结合每个节点的的相邻节点的上下文信息, 来更好的理解节点的语义和角色.
      
    - 多关系图卷积神经网络（Multi-relational Graph Convolutional Network.），一个用来在解码过程中考虑图的语义和结构信息，另一个用来记录哪些结点被描述过。
    - 使用MR-GCN对图编码后, 我们需要同时考虑编码信息中的语义信息和图结构信息. 语义信息反映了图中的实际语义, 图结构信息反映了ASG的结构.    本文为了同时考虑这两种信息, 使用了两种不同的注意力, 分别为Graph Content Attention (图语义注意力) 和Grpah Flow Attention (图流向注意力). 并在最后进行了融合.
    
  - 解码器：两层LSTM，注意力LSTM（全局编码特征，前一个词，上一层输出为输入，输出注意力）和语言LSTM（利用注意力提取第t步文本，将其与文本一同输入生成单词）

- 25.CVPR 2020：SLL-SLE：Better Captioning with Sequence-Level Exploration，改善损失函数
  - ![img](https://leng-mypic.oss-cn-beijing.aliyuncs.com/image/20200629092835550.png)


- 26.CVPR 2020：X-Linear Attention Networks for Image Captioning，Code（https://github.com/JDAI-CV/image-captioning）修改了融合方式，Fastrcnn+lstm

  - 传统的注意力机制只挖掘了输入特征的一阶交互，作者考虑到双线性池化可以有效地处理多模态输入的二阶交互。于是使用双线性池化来改进传统的attention，引入X-线性注意力块，通过双线性池化来选择性地利用视觉信息执行多模态推理。并通过多次堆叠块和引入ELU达到提取高阶交互的作用。

  - X线性注意力块同时利用空间和信道双线性注意分布来捕获输入单模态或多模态之间的二阶相互作用特征。![img](https://leng-mypic.oss-cn-beijing.aliyuncs.com/image/20200712123159605.png)

  - SE模块，提升特征向量的相关性，增大权重大的，减少权重小的。SENet的全称是Squeeze-and-Excitation Networks，主要包含两部分：

    Squeeze：原始feature map的维度为H*W*C，其中H是高度(Height)，W是宽度(width)，C是通道数(channel)。Squeeze做的事情就是将原始特征图H*W*C压缩为1*1*C的响应图(一般采取Global Average Pooling实现)。H*W压缩成一维后，相当于这一维参数获得了之前H*W全局的视野，感受区域更广；Excitation：得到Squeeze的1*1*C的响应图之后，其加入一个全连接层FC(Fully Connected)，对每个通道的重要性进行预测，得到不同加权的通道后再激励到之前的特征图的对应通道上，再进行后续操作。![image-20221015212212137](https://leng-mypic.oss-cn-beijing.aliyuncs.com/image/image-20221015212212137.png)

- ICCV 2019 ：Attention on Attention for Image Captioning，Code（https://github.com/husthuaan/AoANet），Transformer+LSTM结合，解码特征和文本一起输入
  - AOA NET，增加了额外的一个attention来关注Q和V的相关性![在这里插入图片描述](https://leng-mypic.oss-cn-beijing.aliyuncs.com/image/b43db16fce824198b0ea63fdf441ff8b.png)
  - 编码：提取的图像特征进入AOA多头网络
  - 解码：LSTM，提取的注意力，然后再经过一次AOA网络
  
- 27 ECCV2020：Length Controllable Image Captioning，Code（https://github.com/bearcatt/LaBERT）使用bert
  
  - 使用长度控制文本，四个等级[1,9]、[10,14]、[15,19]和[20,25]
  - 非自回归解码器，正常的是文本越长越不准确或者不好计算。
  - 将长度控制，加入AoA和BERT
  - 为了实现这两种方法的长度可控图像标注，直接将长度级嵌入添加到AoANet和VLP的单词嵌入中，而不做任何其他修改。通过这种方式，它们的标注解码器可以显式地建模输入标记的长度信息。
  - 结合长度信息，非自回归模型LaBERT预测
  - ![img](https://leng-mypic.oss-cn-beijing.aliyuncs.com/image/v2-114f2df2591e6f9b1b82d86de09fd94d_1440w.webp)
  
- 30 ECCV 2020：Learning to Generate Grounded Visual Captions without Localization Supervision.Code（https://github.com/chihyaoma/cyclical-visual-captioning）无坐标监督可定位视觉描述生成模型，特点：生成的框和语义更准确


  - **介绍：**RNN等有很强的能力生成词语，并且带有很强的历史信息传播能力，所以这就弱化了一定要对应正确的视觉区域的需求。（就是词语预测对了，但是对应的区域弄错了）
  - 大多数模型是先生成视觉定位，再生成词语，这就容易导致生成的词语定位不准确。所以提出生成-定位-再生成。先对图片生成一句初始的图片内容描述，再对所生成描述中的视觉词汇进行定位，最后根据定位结果重新生成更加准确的描述。这样就可以生成既描述准确又定位准确的图片内容描述了。![img](https://leng-mypic.oss-cn-beijing.aliyuncs.com/image/v2-343cb9bd9ffdfc9da3b3ed6c91e6a750_1440w.webp)
  - 方法：解码-定位-重建

    - 解码:利用论文up-down模型作为解码器，包含两层LSTM，生成一句对图片的初始描述y
    - 定位：对生成的描述句子y进行视觉定位。使用线性层根据词语向量预测定位的区域盖伦出和定位的视觉特征
    - 重建：由于在生成句子之后重新定位的区域更加准确，因此依据定位的结果代替用注意力机制计算的权重重新生成一次图片描述。
    - 损失函数：编码阶段和重建阶段的交叉熵

- 31 CVPR 2020 Meshed-Memory Transformer for Image Captioning，提出一个带有内存的transformer，Code（https://github.com/aimagelab/meshed-memory-transformer）


  - 创新点: 1：在输入图像块中加入了编码先验知识，例如自注意力机制缺陷，如果给定编码鸡蛋和面包，先验可以判断为是早餐，灰色地方为先验知识。
  - 创新2：生成句子采用多层次结构，编码器和解码器之间建立了网格化连接
  - ![img](https://leng-mypic.oss-cn-beijing.aliyuncs.com/image/9cb874a0c6954309bff54a52521d7e46.png)编码器解码器都是多个transformer层组成![img](https://leng-mypic.oss-cn-beijing.aliyuncs.com/image/v2-451e0c35bf75e5cfd8d09d3455260534_1440w.webp)
  - **Memory-Augmented Encoder**：![img](https://leng-mypic.oss-cn-beijing.aliyuncs.com/image/59aa10658c2f4c2ca922be0a7b5a1b9c.png)输入图像区域X，代码中把K和V进行学习，然后拼接，得到矩阵，可能会得到原来没有的信息
  - 解码层：利用前面的单词和区域编码为条件，生成所有的编码。然后cross-attention加权融合求kv和q

- 32 CVPR2017：Semantic Compositional Networks for Visual Captioning，Code（https://github.com/zhegan27/Semantic_Compositional_Nets）![image-20221016161122646](https://leng-mypic.oss-cn-beijing.aliyuncs.com/image/image-20221016161122646.png)
  - 改论文提出了语义组合网络SCN，有效的利用语义标签来达到更好地描述。就是利用模型先得到标签`Detected tags: grass (1.0), field (0.985), dog (0.977), outdoor (0.94), black (0.919), yellow (0.65), green (0.591), grassy (0.388), small (0.341), standing (0.301)`，相当于为每一个词语提供了一个LSTM块的参数，所以更准
  - 利用CNN提取特征，RNN生成文字
  - 图像特征仅在开始时输入一次
  - SCN为每一个权重矩阵提供一个标签

- 33 未看完：CVPR2021 Connecting What to Say With Where to Look by Modeling Human Attention Traces
  - 两个任务：
    - 给定图像和文字预测轨迹
    - 给定图像下预测字幕和轨迹
- 34 CVPR 2021：Towards accurate text-based image captioning with content diversity exploration，Code（https://github.com/guanghuixu/AnchorCaptioner）文本图像字幕
  - 提出了Anchor-Captioner Method ，ACM，先从图像中识别出重要文字并通过attention权重来确定是否作为anchor。对于每一个选中的anchor，会将它相关的文本来构建关键词图（anchor-centred graph，ACG)；最后，基于不同的ACG来生成内容丰富的图像描述句子。
  - 

```

python demo/demo.py --config-file configs/COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml --input demo/input1.jpg --output outputs/  --opts MODEL.WEIGHTS demo/model/model_final_f10217.pkl
```



