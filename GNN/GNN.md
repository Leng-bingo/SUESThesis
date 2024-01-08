# 图神经网络

论文地址https://distill.pub/2021/gnn-intro/

> 综述科普型文章

---

Graphs are all around us 我们身边都是图，图是一个一般化的架构

十几年前，学者就已经提出 GNN（graph neural networks）

最近 GNN 重新兴起，开始被应用到

- 药物发现
- 物理模拟
- 虚假新闻的检测
- 车流量预测
- 推荐系统等

- 图是一个非常强大的东西，但是它的强大也带来了很多问题：很难在图上做出优化，图一般比较稀疏，有效的在CPU、GPU、加速器上计算是一件比较难的事情；图神经网络对超参数比较敏感。
- 很多图是交互图（既是优点（非常漂亮）又是缺点（门槛太高，有时用一个公式可以清晰的将其表达出来））。
- 什么是图？图的属性应该用向量来进行表示。对于顶点、边、全局都用向量来表示它的属性。
- GNN就是对属性做变换，但不改变图的结构。
- GNN：对图上所有属性进行的可以优化的变换（保持图的对称信息，顶点变换后结果不变）
- 信息传递神经网络，u全局信息，v边，e点做三个MLP多层感知机，输入输出一样。三个mlp组成gnn的一个层，把三个向量分别找到对应的mlp进行更新
- 属性有缺失可以做聚合操作，把边的数据拿到结点上面，补足缺失的
- GNN：每一层里面通过汇聚操作，把信息传递过来，每个顶点看邻接顶点的信息；每个顶点看邻接边的信息或全局的信息。在每一层上如果能对信息进行充分的汇聚，那么GNN可以对图的结构进行一个发掘。

---

### 图的简介

图（*graph*）是实体（entities, called *nodes*）之间关系（relations, called *edges*）的表示。

![](https://leng-mypic.oss-cn-beijing.aliyuncs.com/mac-img/20220516143105.png)

我们可以用embedding向量来分别表示nodes，edges，global

![](https://leng-mypic.oss-cn-beijing.aliyuncs.com/mac-img/20220516143203.png)



同时分为有向图（*directed*）与无向图（*undirected*）两种

![](https://leng-mypic.oss-cn-beijing.aliyuncs.com/mac-img/20220516143306.png)

### 数据如何表示为图

#### images as graphs

将每一个pixel当做一个node，用一个sparse matrix作为邻接矩阵表示edges（无向图所以是对称的）

![](https://leng-mypic.oss-cn-beijing.aliyuncs.com/mac-img/20220516143339.png)



#### texts as graphs

将每一个word当做一个node，用word books创建邻接矩阵表示edges（一般情况是有向路）

![](https://leng-mypic.oss-cn-beijing.aliyuncs.com/mac-img/20220516143403.png)



#### 现实中的图

- 分子结构

![](https://leng-mypic.oss-cn-beijing.aliyuncs.com/mac-img/20220516143430.png)



- 社交网络

![](https://leng-mypic.oss-cn-beijing.aliyuncs.com/mac-img/20220516143455.png)

---

### The challenges of using graphs in machine learning

graphs 与 ml 的兼容性：最大的challenge是connectivity-adjacent matrix

- 稀疏性
- 多样性

稀疏矩阵的密集表示：

![](https://leng-mypic.oss-cn-beijing.aliyuncs.com/mac-img/20220516143557.png)



### GNN

A GNN is an optimizable transformation on all attributes of the graph (nodes, edges, global-contex ) that preserves graph symmetries (permuation invariance).

GNN是对图上所有的属性（包括顶点、边和全局上下文）进行的可以优化的变换

- 对于embedding施加变换而不会对connectivity造成改变
- 相同类型的元素都使用相同的变换

![](https://leng-mypic.oss-cn-beijing.aliyuncs.com/mac-img/20220516143636.png)



#### Pooling

node可以有edges+global pooling得到，反之亦然

![](https://leng-mypic.oss-cn-beijing.aliyuncs.com/mac-img/20220516143658.png)



#### end2end model

![](https://leng-mypic.oss-cn-beijing.aliyuncs.com/mac-img/20220516143710.png)



#### 模型假设

- GNN：连接不变性
- CNN：空间变换不变性
- RNN：时序连续性

#### 其他

- Comparing aggregation operations： 没有一个是better的
- Edges and the Graph Dual
- Graph Attention Networks：需要注意的是weight的选择，可以用self-attention之类解决

## Comment

- 思路清晰，流程由浅到深
- 图片制作精细，交互图更能凸显重点，但同时写作门槛高
- 避免公式/Code可以降低理解门槛，但同时会不够简洁
- 宽泛的简单介绍不如一次讲明白
- graph具有强大的表示能力，但同时难以优化，工业上目前难以应用