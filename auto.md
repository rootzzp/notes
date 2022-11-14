# Bird’s-Eye-View Panoptic Segmentation Using Monocular Frontal View Images
## 整体结构：
骨干网络采用了EfficientDet的修改变体（图中灰色），它以四种不同的尺度$\varepsilon_4$、$\varepsilon_8$、$\varepsilon_{16}$和$\varepsilon_{32}$输出特征图。然后将特征图输入到Dense transformer模块，该模块由两个不同的变换器组成，它们独立地将输入FV图像中的垂直和平坦区域转换为 BEV。然后，Dense transformer将变换后的垂直特征图和平面特征图组合起来，以产生相应的复合BEV特征。随后，转换后的特征图并行输入语义和实例头，然后是全景融合模块，生成最终的BEV全景分割输出\
![PanopticBEV](images/deeplearning/networks/PanopticBEV/panoticBEV.png)

## Dense Transformer
如下图所示：将图像中不同区域投影到BEV空间中时，是有不同的特点的。对于垂直平坦区域（路面这些），由于平坦区域是完全可观察的，除非被另一个物体遮挡，因此将平坦区域转换为BEV仅涉及使用模型校正透视失真并推断远处区域中的缺失信息。与之相反的，属于垂直区域（车辆，人等具有3d信息的区域）的列需要映射到BEV空间中正交体积区域。作为车辆和人类等 3D 体积对象的投影，垂直区域不是完全可观察的，并且通常缺少维度。例如，由于缺乏与其空间范围有关的信息，因此不能完全观察到汽车。此外，从单目相机捕捉到的它们在世界中的深度也是模棱两可的，这进一步使问题更具挑战性。因此，将垂直非平面对象转换为BEV需要预测其空间位置和其它模型使用数据驱动范式学习到的东西。由于以上特点，针对平坦和垂直区域，需要不同的模块去进行学习（平坦和垂直区域学习的难度和学习的目标有所不同）\
![BEV space](images/deeplearning/networks/PanopticBEV/bev_space.png)
如下图所示，首先，利用连续的2d卷积处理来自backbone不同尺度的特征$\varepsilon_k$分别得到垂直和平坦区域语义mask模块$S^v_k$和$S^f_k$。然后我们计算语义mask和主干特征之间的Hadamard积，以产生垂直和平面特征$V_k$和$F_k$。随后，我们使用各自的transformer将$V_k$和$F_k$独立地转换为它们的BEV表示$V^{bev}_k$和$F^{bev}_k$。然后我们在 BEV 空间中结合这些特征来生成复合BEV特征图 $\varepsilon^{bev}_k$。在训练阶段，使用groundtruth垂直平面掩码监督$S^v_k$和$S^f_k$，从而引导语义mask的学习。
![dense transform](images/deeplearning/networks/PanopticBEV/dense.png)
### Vertical Transformer
见上图中上半部分，
### Flat Transformer
见上图中下半部分，

## Semantic and Instance Heads

## Panoptic Fusion Module

## Loss

cite: [paper](http://arxiv.org/abs/2108.03227)

