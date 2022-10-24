# 特征聚合

# FPN系列
其结构图如下所示：
![FPN](images/deeplearning/feature_integration/fpn.png)
## BiFPN
其结构图如下所示：
![BiFPN](images/deeplearning/feature_integration/BiFPN.png)
特点：
1.移除单输入边的结点：因为单输入边的结点没有进行特征融合，故具有的信息比较少，对于最后的融合没有什么贡献度，相反，移除还能减少计算量。
2.权值融合：简单来说，就是针对融合的各个尺度特征增加一个权重，调节每个尺度的贡献度。
3.增加残差链接：意在通过简单的残差操作，增强特征的表示能力。