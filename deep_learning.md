# 主干网络
## CSPNet(Cross Stage Partial Network)
设计思想：
减少计算量的同时实现更丰富的梯度组合。这个目标是通过将基础层的特征图划分为两部分，然后通过提出的跨阶段层次结构合并它们来实现的。其中最主要概念是通过拆分梯度流使梯度流通过不同的网络路径传播

主要目的：
### 1.增强CNN的学习能力
现有的CNN在轻量化后精度大大降低，所以希望增强CNN的学习能力，使其在轻量化的同时保持足够的精度。CSPNet 可以很容易地应用于 ResNet、ResNeXt 和 DenseNet。在上述网络上应用CSPNet后，计算量可以减少10%到20%，而且在准确性方面更优
### 2.降低bottlenecks的计算量
bottlenecks计算量太高会导致完成推理过程需要更多的周期，或者一些计算单元经常空闲。因此，希望能够将CNN中每一层的计算量平均分配，从而有效提升各计算单元的利用率，从而减少不必要的能耗
### 3.降低内存占用
采用跨通道池化在特征金字塔生成过程中压缩特征图

其结构图如下所示为CSPDenseNet网络：主要由部分稠密块与部分过渡层组成\
![CSPDenseNet](images/deeplearning/backbone/CSPDenseNet.png)
#### 部分稠密块
1.增加梯度路径：通过分块归并策略，可以使梯度路径的数量增加一倍。由于采用了跨阶段策略，可以减轻使用显式特征图copy进行拼接所带来的弊端\
2.每一层的平衡计算：通常，DenseNet基础层(图中x0)的通道数远大于生长率。由于在部分稠密块中，参与稠密层操作的基础层通道仅占原始数据的一半，可以有效解决近一半的计算瓶颈\
3.减少内存流量
#### 部分过渡层
部分过渡层的目的是使梯度组合的差异最大。局部过渡层是一种层次化的特征融合机制，它利用梯度流的聚合策略来防止不同的层学习重复的梯度信息，主要由两种方式：如下图所示：\
![feature fusion](images/deeplearning/backbone/csp_feature_fusion.png)
1.前融合，就是把两部分生成的特征图拼接起来，然后做transition操作。如果采用这种策略，将会重复使用大量的梯度信息。实验结果表明，显着降低计算成本，但 top-1 准确率显着下降了 1.5%\
2.后融合，密集块的输出将通过过渡层，然后与来自第1部分的特征图进行连接。梯度信息由于梯度流被截断，因此不会被重用。实验结果表明，计算成本可以显着下降，但 top-1 准确率仅下降 0.1%

cite: [paper](http://arxiv.org/abs/1911.11929)

# 特征聚合
## FPN系列
其结构图如下所示：\
![FPN](images/deeplearning/feature_integration/fpn.png)\
cite: [paper](http://arxiv.org/abs/1612.03144)
## PANet
其结构图如下所示：\
![PANet](images/deeplearning/feature_integration/PANet.png)\
红色虚线箭头表示在FPN算法中，因为要走自底向上的过程，浅层的特征传递到顶层要经过几十甚至一百多个网络层（在FPN中，对应Figure1中那4个蓝色矩形块从下到上分别是ResNet的res2、res3、res4和res5层的输出，层数大概在几十到一百多左右），显然经过这么多层的传递，浅层特征信息丢失会比较厉害。绿色虚线箭头表示作者添加一个bottom-up path augmentation，本身这个结构不到10层，这样浅层特征经过底下原来FPN的lateral connection连接到P2再从P2沿着bottom-up path augmentation传递到顶层，经过的层数就不到10层，能较好地保留浅层特征信息。关于bottom-up path augmentation的具体设计见下图：\
![bottom-up path augmentation](images/deeplearning/feature_integration/bottom_up.png)\
cite: [paper](http://arxiv.org/abs/1803.01534)
## BiFPN
其结构图如下所示：
![BiFPN](images/deeplearning/feature_integration/BiFPN.png)
特点：
1.移除单输入边的结点：因为单输入边的结点没有进行特征融合，故具有的信息比较少，对于最后的融合没有什么贡献度，相反，移除还能减少计算量。
2.权值融合：简单来说，就是针对融合的各个尺度特征增加一个权重，调节每个尺度的贡献度。
3.增加残差链接：意在通过简单的残差操作，增强特征的表示能力。\
cite: [paper](http://arxiv.org/abs/1911.09070)

## SFAM(Scale-wise Feature Aggregation Module)
首先，其整体网络结构如下所示：其中，FFMv2将前一层TUM产生的多级特征图中最大输出特征图和基本特征融合，形成的融合特征图作为下一层TUM的输入，因此可以产生多个不同深度的特征金字塔\
![MLFPN](images/deeplearning/feature_integration/mlfpn.png)
其中SFAM模块对不同级别的特征金字塔进行融合，将金字塔中同尺度的特征图concat起来并利用SE模块进行注意力加权处理，其结构图如下所示：\
![SFAM](images/deeplearning/feature_integration/SFAM.png)
cite: [paper](http://arxiv.org/abs/1811.04533)

## ASFF(Adaptively Spatial Feature Fusion)
与以前使用元素累积和或连接来集成多级特征的方法不同，此方法的关键思想是自适应地学习每个尺度的特征图融合的空间权重，它由两个步骤组成：同尺度缩放和自适应融合。如下图所示，对ASFF3模块，有3个学习参数，分别是:$\alpha,\beta,\gamma$,代表了不同尺度特征的权重，图中$X^{1\rightarrow3}$代表第1级特征图缩放到第3级特征图相同尺度后的特征图，依次类推$X^{2\rightarrow3}$代表第2级特征图缩放到第3级特征图相同尺度后的特征图。\
![ASFF](images/deeplearning/feature_integration/ASFF.png)
cite: [paper](http://arxiv.org/abs/1911.09516)

# 增强感受野的模块
## SPP(spatial pyramid pooling layer)
![SPP](images/deeplearning/enhance_receptive_field/SPP.png)
cite: [paper](http://arxiv.org/abs/1406.4729)

## ASPP(Atrous Spatial Pyramid Pooling)
![ASPP](images/deeplearning/enhance_receptive_field/ASPP.png)
cite: [paper](http://arxiv.org/abs/1606.00915)

## RFB(Receptive Field Block)
![RFB](images/deeplearning/enhance_receptive_field/RFB.png)
![COMPARE](images/deeplearning/enhance_receptive_field/COMPARE.png)
cite: [paper](http://arxiv.org/abs/1711.07767)