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
