# 交叉熵损失函数:
设类别数为K，$p_i=\sigma({x_i})$表示单个样本第i类的概率，通常用softmax函数得到，y为标签，用onehot的形式表示，$y_i$表示其第i个分量（0/1）
$$p_i=\sigma({x_i})=\frac {e^{-x_i}}{\sum_{m=1}^N e^{-x_m}}$$
$$l=\sum_{m=1}^N -y_ilog(\sigma(x_i))$$
# 二分类交叉熵损失函数:
类别数为2的交叉熵,y不再是Onehot的形式了，y={0,1}
$$l=\sum_{m=1}^2 -y_ilog(\sigma(x_i))=-ylog(\sigma(x))-(1-y)(log(1-\sigma(x)))=
\begin{cases}
-log(\sigma(x)), &if\ y=1\\
-(log(1-\sigma(x))), &if\ y=0
\end{cases}$$
# focal loss
设$p=\sigma(x)$，代表y=1的概率，focal loss对于分类不准确的样本，损失没有改变，对于分类准确的样本，损失会变小。 整体而言，相当于增加了分类不准确样本在损失函数中的权重。
$$l=
\begin{cases}
-\alpha(1-p)^{\gamma}\log(p), &if\ y=1\\
-\alpha(p)^{\gamma}(log(1-p)), &if\ y=0
\end{cases}$$
# Dice Loss:
用于分割，X表示预测值，Y表示真实值（由0或1表示）,交集用对应位相乘表示，||表示平方和
$$l=1-\frac{2|X\cap\\Y|}{|X|+|Y|}$$
# Center Loss:
m表示batch_size
$$l=\frac {1}{2}\sum_{i=1}^m(x_i-c_{yi})^2$$

# Kullback–Leibler divergence(KL):
$$D(p||q)=\sum p(x)log(\frac{p(x)}{q(x)})$$

# Jensen-Shannon divergence(JS):
$$D(p||q)=\frac{1}{2}KL(p||\frac{p+q}{2})+\frac{1}{2}KL(q||\frac{p+q}{2})$$

# Attention:
其实RNN存在着两种训练模式(mode):
1.free-running mode,就是大家常见的那种训练网络的方式: 上一个state的输出作为下一个state的输入
2.teacher-forcing mode,直接使用训练数据的标准答案(ground truth)的对应上一项作为下一个state的输入（训练迭代过程早期的RNN预测能力非常弱，几乎不能给出好的生成结果。如果某一个unit产生了垃圾结果，必然会影响后面一片unit的学习）
## seq2seq中的：
设encoder的隐状态为$(h_1,h_2,h_T)$ T为序列长度，序列中的每个位置处均有一个隐状态，当前decoder的隐状态为$s_{t-1}$,需要得到当前时刻的输出和下一时刻的隐状态
decoder当前位置的隐状态与第$j$个输入位置的关联性为$e_{tj}=a(s_{t-1},h_{j})$,其中a是一种相关性运算，可以用（$s_{t-1}^T h_{j}$，$s_{t-1}^TW^Th_{j}$等）表示；那么与所有输入位置的关联性可以构成一个
长度为T的向量$e=(a(s_{t-1},h_{1}),a(s_{t-1},h_{2}),,,,a(s_{t-1},h_{T}))$，利用softmax进行归一化得到
$$\hat{e_{tj}}=\frac{exp(e_{tj})}{\sum_{k=1}^Texp(e_{tk})}$$
那么加权求和得到的上下文向量为$c=\sum_{k=1}^T \hat{e_{tk}} h_{k}$,那么下一个隐藏状态$s_{t} = f(s_{t-1}, y_{t-1}, c)$,$y_{t-1}$表示上一时刻的真值（teacher-forcing），通常c和$y_{t-1}$用concat操作合并在一起
## transformer中的：
$$Attention(Q,K,V)=softmax(\frac{QK^T}{\sqrt{d_k}})V$$

# 感受野:
$$l_k=l_{k-1}+[(f_k-1)\prod_{i=0}^n(s_i)]$$

# 卷积输出尺寸:
$$(W+2p-w)/s+1$$

# 空洞卷积等效卷积核:
$$k_e=k+(k-1)(r-1)$$

# BN:
卷积的bn计算是沿着通道方向，将h*w平面计算均值，当作一个神经元，然后按batch求得均值，所以$\alpha和\beta的维度均为c(通道数)$

# MobileNet v1,v2,v3:
v1: dw卷积 + relu6\
v2: inverted residual with linear bottleneck(pw卷积升维(bn+relu) + dw卷积(bn+relu) + pw卷积降维(bn))\
v3: h-swish se模块

# ShuffleNet:
V1:1*1 group conv + channel shuffle + 3*3 dw conv + 1*1 group conv\
V2:(a)当输入、输出channels数目相同时，conv计算所需的MAC(memory access cost)最为节省(b)过多的Group convolution会加大MAC开销(c)网络结构整体的碎片化会减少其可并行优化的程序(d)Element-wise操作会消耗较多的时间

# GhostNet:
将传统的卷积分成两步进行，首先利用较少的计算量通过传统的卷积生成channel较小的特征图，然后在此特征图的基础上，通过cheap operation(depthwise conv)再进一步利用较少的计算量，生成新的特征图，最后将两组特征图拼接到一起

# PPLCNET:
v1:  (a)H-Swish(b)合适的位置添加 SE 模块(c)合适的位置添加更大的卷积核(d)GAP 后使用更大的 1x1 卷积层
v2:  (a)重参数化(b)两层 PW 卷积设置为：第一个在通道维度对特征图压缩，第二个再通过放大还原特征图通道


# Resnet
四个stage,每个stage由多个block组成，block可以使用BasicBlock(2个3*3卷积)和Bottleneck(通道利用1*1降维，再3*3卷积，再1*1卷积升维),ResNet50起，就采用Bottleneck，每个stage通过步长为2的卷积层执行下采样，而却这个下采样只会在每一个stage的第一个卷积完成，有且仅有一次\
改进一：downsample部分，减少信息流失(使用和Bottleneck时，下采样使用1*1卷积，并且stride为2，会造成信息损失)，所以将stride=2移到3*3卷积层上去\
改进二: 预激活，残差结构相加后需要进入ReLU做一个非线性激活，这里一个改进就是砍掉了这个非线性激活，不难理解，如果将ReLU放在原先的位置，那么残差块输出永远是非负的，这制约了模型的表达能力，因此我们需要做一些调整，我们将这个ReLU移入了残差块内部\
ResNeXt：组卷积+Inception的计算模式增加网络宽度\
ResNeSt：ResNeXt基础上加入SKNet（网络中使用多尺寸的卷积核，结果相加，经过类似senet的结构得到squeeze的向量再和不同尺寸的卷积核得到的结果相乘再相加）,额外介绍一下spacital attention 通道方向计算均值 和 最大值 concat后利用sigmoid激活后和原始特征图相乘\

# Faster Rcnn:
训练：\
RPN: 对每个生成的anchor预测位置偏移量和目标分类（二分类），根据位置偏移量生成roi，再根据分数值排序筛选分数值靠前的多少个roi，最后进行NMS，输出roi；根据anchor和gt的关系为每个anchor赋予标签（正样本，负样本，类别），roi与anchor之间计算loss\
Rcnn: 对RPN输出的roi根据预设的正负样本比率和与box的iou值进行筛选，挑选出128个roi,这些roi到box的偏移为位置回归的真值；根据这128个roi利用roipooling提取特征，对这些特征利用卷积输出位置回归的预测值，计算loss\
Anchor :
$$scale = \sqrt{h*w},ratio = \frac{h}{w}$$
$$a_h = s*\sqrt{ratio},a_w = s*\sqrt{\frac{1}{ratio}}$$
位置回归的真值：其中$x_a$值anchor的中心点x坐标，依次类推
$$t^*_x=(x^*-x_a)/w_a$$
$$t^*_y=(y^*-y_a)/h_a$$
$$t^*_w=log(w^*/w_a)$$
$$t^*_h=log(h^*/h_a)$$
利用位置回归的预测值进行位置解码：
$$x = t_x*w_a+x_a$$
$$y=t_y*h_a+y_a$$
$$w=e^{t_w}*w_a$$
$$h=e^{t_h}*h_a$$

# yolo:
## v1:
## v2:
## v3:
## v4:
## v5:
## v6:
## v7:

# fcos:

# GAN:
$$\min_{G}\max_{D}E_{x~pdata(x)}[log(D(x))] + E_{z~p_z(z)}[1-log(D(z))]$$

# GAT图注意力网络：

# 常见分割网络：
## UNET:
下采样使用max_pooling,上采样用反卷积，相同大小的特征图用concat融合。之所以用于医学领域：1.语义简单所有信息都重要，concat能保存所有信息2.大网络往往需要大量数据，对医学小数据容易过拟合3.结构简单，可以添加自己想要的分支
## FCN:
上采样+跳连结构
## FPN:
FPN在金字塔的所有层都进行prediction
## PANet:
自顶向下和自底向上双向融合
https://zhuanlan.zhihu.com/p/373907181
## Deeplab:
v1:基于深度卷积网络和全连接CRF的语义图像分割
v2:利用空洞卷积，可实现在不增加参数量的情况下有效扩大感受域，合并
更多的上下文信息；DCNNs与CRF相结合，进一步优化网络效果；提出了ASPP模块
v3:为了解决多尺度下的分割问题，本文设计了级联或并行的空洞卷积模
块；扩充了ASPP模块
v3+:另行设计了解码器结构，取消CRF做后处理
## Segformer:
ViT-large 参数和计算量非常大
ViT的结构不太适合做语义分割，因为ViT是柱状结构，全程只能输出固定分辨率的feature map, 比如1/16, 这么低的分辨率对于语义分割不太友好，尤其是对轮廓等细节要求比较精细的场景
位置编码. ViT 用的是固定分辨率的positional embedding, 但是语义分割在测试的时候往往图片的分辨率不是固定的，这时要么对positional embedding做双线性插值，这会损害性能, 要么做固定分辨率的滑动窗口测试，这样效率很低而且很不灵活
(1) 之前ViT和PVT做patch embedding时，每个patch是独立的，我们这里对patch设计成有overlap的，这样可以保证局部连续性。(2) 我们彻底去掉了Positional Embedding, 取而代之的是Mix FFN, 即在feed forward network中引入3x3 deepwise conv传递位置信息。（3）仅由几个FC构成的decoder

# 常见VIT：
## 

# 常见传统图像特征提取：
## SIFT(高斯卷积核是实现尺度变换的唯一变换核):
### 1尺度空间极值检测：搜索所有尺度上的图像位置。通过高斯微分函数来识别潜在的对于尺度和旋转不变的兴趣点
尺度空间(尺度空间中各尺度图像的模糊程度逐渐变大，能够模拟人在距离目标由近到远时目标在视网膜上的形成过程)使用高斯金字塔表示。Tony Lindeberg指出尺度规范化的LoG(Laplacion of Gaussian)算子具有真正的尺度不变性，Lowe使用高斯差分金字塔近似LoG算子，在尺度空间检测稳定的关键点。一个图像的尺度空间，定义为一个变化尺度的高斯函数与原图像的卷积。为了让尺度体现其连续性，高斯金字塔在简单降采样的基础上加上了高斯滤波。将图像金字塔每层的一张图像使用不同参数做高斯模糊，使得金字塔的每层含有多张高斯模糊图像，将金字塔每层多张图像合称为一组(Octave)，组数和金字塔层数相等。在实际计算时，使用高斯金字塔每组中相邻上下两层图像相减，得到高斯差分图像，进行极值检测
### 2关键点定位：在每个候选的位置上，通过一个拟合精细的模型来确定位置和尺度。关键点的选择依据于它们的稳定程度
关键点粗定位：关键点是由DOG空间的局部极值点组成的，关键点的初步探查是通过同一组内各DoG相邻两层图像之间比较完成的。为了寻找DoG函数的极值点，每一个像素点要和它所有的相邻点比较，看其是否比它的图像域和尺度域的相邻点大或者小。每个检测点要和它同尺度的8个相邻点和上下相邻尺度对应的9×2个点共26个点比较，以确保在尺度空间和二维图像空间都检测到极值点
精定位：离散空间的极值点并不是真正的极值点，利用已知的离散空间点插值得到的连续空间极值点的方法叫做子像素插值。为了提高关键点的稳定性，需要对尺度空间DoG函数进行曲线拟合。利用DoG函数在尺度空间的Taylor展开式进行插值
DOG算子会产生较强的边缘响应，需要剔除不稳定的边缘响应点。获取特征点处的Hessian矩阵，主曲率通过一个2x2 的Hessian矩阵H求出，H的特征值α和β代表x和y方向的梯度
### 3方向确定：基于图像局部的梯度方向，分配给每个关键点位置一个或多个方向。所有后面的对图像数据的操作都相对于关键点的方向、尺度和位置进行变换，从而提供对于这些变换的不变性
为了使描述符具有旋转不变性，需要利用图像的局部特征为给每一个关键点分配一个基准方向。使用图像梯度的方法求取局部结构的稳定方向。对于在DOG金字塔中检测出的关键点点，采集其所在高斯金字塔图像3σ邻域窗口内像素的梯度和方向分布特征
###  4关键点描述：在每个关键点周围的邻域内，在选定的尺度上测量图像局部的梯度。这些梯度被变换成一种表示，这种表示允许比较大的局部形状的变形和光照变化
通过以上步骤，对于每一个关键点，拥有三个信息：位置、尺度以及方向。接下来就是为每个关键点建立一个描述符，用一组向量将这个关键点描述出来，使其不随各种变化而改变，比如光照变化、视角变化等等。SIFT描述子是关键点邻域高斯图像梯度统计结果的一种表示。通过对关键点周围图像区域分块，计算块内梯度直方图，生成具有独特性的向量
### 优缺点：
SIFT在图像的不变特征提取方面拥有无与伦比的优势，但并不完美，仍然存在：实时性不高、有时特征点较少、对边缘光滑的目标无法准确提取特征点等问题
## SURF
其实surf构造的金字塔图像与sift有很大不同，就是因为这些不同才加快了其检测的速度。Sift采用的是DOG图像，而surf采用的是Hessian矩阵行列式近似值图像。
借助积分图，图像与高斯二阶微分模板的滤波转化为对积分图像的加减运算，从而在特征点的检测时大大缩短了搜索时间
尺度空间：SIFT使用DoG金字塔与图像进行卷积操作，而且对图像有做降采样处理；SURF是用近似DoH金字塔(不同尺度的盒式滤波)与图像做卷积，借助积分图，实际操作只涉及到数次简单的加减运算，而且不改变图像大小。
特征点检测：SIFT是先进行非极大值抑制，去除对比度低的点，再通过Hessian矩阵剔除边缘点。而SURF是计算Hessian矩阵的行列式值，再进行非极大值抑制。
特征点主方向：SIFT在方形邻域窗口内统计梯度方向直方图，并对梯度幅值加权，取最大峰对应的方向；SURF是在圆形区域内，计算各个扇形范围内x、y方向的Haar小波响应值，确定响应累加和值最大的扇形方向。
特征描述子：SIFT将关键点附近的邻域划分为4×4的区域，统计每个子区域的梯度方向直方图，连接成一个4×4×8=128维的特征向量；SURF将20s×20s的邻域划分为4×4个子块，计算每个子块的Haar小波响应，并统计4个特征量，得到4×4×4=64维的特征向量。
## ORB
### 利用FAST定位角点（若某像素与其周围邻域内足够多的像素点相差较大，则该像素可能是角点）：
1我们仍旧选择一个点，不妨仍设这个点为像素点P。我们首先把它的亮度值设为Ip。
2虑以该像素点为中心的一个半径等于3像素的离散化的Bresenham圆，这个圆的边界上有16个像素，分别为p1 ～ p16，亮度值分别为Ip1 ～ Ip16。
3我们选定一个阈值t，t做什么的我们接下来看。
4首先计算p1、p9、p5、p13(分别在四个方向的四个点)与中心p的像素差，若它们的绝对值有至少3个超过阈值t，则当做候选角点，再进行下一步考察；否则，不可能是角点；
5在p是候选点的情况下，计算p1到p16这16个点与中心p的像素差，若它们有至少连续9个超过阈值（也可以测试其他大小，实验表明9的效果更好），则是角点；否则，不是角点。
6对完整图进行1-5 的角点判断。
7对完整图像进行非极大值抑制，目的去除小区域内多个重复的特征点：
7.1. 计算特征点出的FAST得分值（或者说响应值），即16个点与中心差值的绝对值总和。
7.2. 判断以特征点p为中心的一个邻域（可以设为3x3或着5x5）内，若有多个特征点，则判断每个特征点的响应值,如果p是其中最大的，则保留，否则，删除。如果只有一个特征点，就保留。
### 计算描述子：
ORB实现旋转不变性，也是通过确定一个特征点的方向来实现的，ORB的论文中提出了一种利用灰度质心法来解决这个问题，通过计算一个矩来计算特征点以r为半径范围内的质心，特征点坐标到质心形成一个向量作为该特征点的方向。
ORB选择了BRIEF作为特征描述方法，它是在每一个特征点的邻域内，选择n对像素点pi、qi（i=1,2,…,n）。然后比较每个点对的灰度值的大小。如果I(pi)> I(qi)，则生成二进制串中的1，否则为所有的点对都进行比较，则生成长度为n的二进制串。一般n取128、256或512，通常取256。

# 文本定位优化
spatical attention + channel attention + 重参数化

# 文本定位优化
扭曲文本识别：
tps + stn + 1d attention
2d attention(Hidden_state作为一个Query，Feature Map作为一个Key和Value存在；encoder层得到的最后一个隐向量作为编码特征，该编码特征concat标签嵌入得到decoder的输入，decoder后的向量作为query,图像特征作为keu和value，attention得到的值经过全连接层得到输出特征（n,t,c）)
transformer
辅助loss:
ctc + attention

# transformer
用在文本识别中时，encoder的输入是图像卷积特征，decoder的输入是标签序列的embeeding

# mAP
## Precision：
根据上述方法，对于某个类别A，我们先计算每张图片中A类别TP和FP的数量并进行累加，即可得到类别A在整个数据集中TP和FP的数量，计算TP/(TP+FP)即可得到类别A的Precision (计算Precision的时候只需要用到TP和FP)，但是会发现Precision的数值是受模型预测出的边界框的数量(上述计算式的分母部分)影响的，如果我们控制模型输出预测框的数量，就可以得到不同的Precision，所以我们可以设置不同的score阈值，最终得到不同数量的TP和FP。

## Recall：
对于某个类别A，按上述方法进行累加TP的数量，计算TP/(n_gt)即可得到Recall，其中n_gt表示类别A在所有图片中gt的数量之和。同理，如果控制模型输出的预测框的数量，就会改变TP的数量，也就会改变Recall的值。

综上，想要得到PR曲线，可以通过改变score的阈值来控制模型输出的预测框数量，从而得到不同的TP、FP、FN。不过在实际操作中，并不需要手动来设置score的阈值，因为每个预测框都有一个score，我们只需要将其按从小到大进行排序，然后每次选择最后一个score作为阈值即可，这样如果类别A在所有图片中的预测框数量之和有100个，就可以计算100组类别A的Precision-Recall值。

随着score的阈值的增大，预测框总数会一直减小，而TP却会有时保持不变，有时变小，所以会出现同一Recall对应多组Precision的情况。最终在计算AP时，我们只需要取每个Recall值对应的最大Precision进行计算即可。因为A类别一共有6个gt，所以Recall的值应该是从1/6~6/6共6个，也就是要取6组PR值计算平均Precision，因为这个例子中没有出现Recall=6/6的情况，所以R=6/6时的Precision算作0，即类别A的AP=(1/1 + 2/2 + 3/3 + 4/4+ 5/8 + 0) / 6 = 0.7708

mAP：平均精度超过IOU阈值的平均精度，范围从.5到.95，增量为.05

mAP（large）：大对象的平均精度（96 ^ 2像素<区域<10000 ^ 2像素）

mAP (medium)：中等大小对象的平均精度（32 ^ 2像素<区域<96 ^ 2像素）

mAP (small)：小对象的平均精度（区域<32 ^ 2像素）(此处为区域为面积)

mAP@.50IOU：平均精度为50％IOU

mAP@.75IOU：平均精度为75％IOU

# 优化算法
设$g_t$表示第t次更新的梯度，则$g_t=\frac{1}{K}\sum{\frac{\partial{L(\theta,x)}}{\partial \theta}}$\
SGD：
$$\theta{_t}=\theta{_{t-1}}-\alpha*g_t$$
Adam：
$$M_t=\beta_1M_{t-1}+(1-\beta_1)g_t$$
$$G_t=\beta_2G_{t-1}+(1-\beta_2)g_t*g_t$$
$$\hat{M_t}=\frac{M_t}{1-\beta_1^t}$$
$$\hat{G_t}=\frac{G_t}{1-\beta_2^t}$$
$$\theta{_t}=\theta{_{t-1}}-\frac{\alpha}{\sqrt{\hat{G_t}+\epsilon}}\hat{M_t}$$



# kalman filter
## 一维
### 预测
当前位置为$x_{k-1}$,预测的下时刻的位置为$\overline{x}_{k}$,那么
$$\overline{x}_{k} = x_{k-1} + v_k*\Delta{t}=x_{k-1} + f_x$$
由于当前位置（状态）本身就带有噪声，有一定的不确定度，可以用$x_{k-1}$~$N_1(\mu_1, \sigma_1^2)$表示，而速度$v_k$（或其他控制量）也是具有不确定度的，可以用$f_x$~$N_2(\mu_2, \sigma_2^2)$表示。

那么这两个不确定叠加，得到预测的下时刻的位置为$\overline{x}_{k}$~$(N_1+N_2)=N_3$，即预测的位置为$\overline{\mu} = \mu_1+\mu_2$，不确定度为$\overline{\sigma}^2 = \sigma_1^2+\sigma_2^2$，不确定度会变大
### 更新
根据测量对状态和不确定度更新，假设测量量为$z_k$~$N_4(\mu_z, \sigma_z^2)$,利用两个高斯分布相乘对$\overline{x}_{k}$进行更新得到$x_{k}$，更新方式为$N_3*N_4=N(\frac{\sigma_z^2\overline{\mu} + \overline{\sigma}^2\mu_z}{\sigma_z^2+\overline{\sigma}^2},\frac{\sigma_z^2\overline{\sigma}^2}{\sigma_z^2+\overline{\sigma}^2})$，可以这么理解，更新完后，k处的位置$x_{k}$为$\frac{\sigma_z^2\overline{\mu} + \overline{\sigma}^2\mu_z}{\sigma_z^2+\overline{\sigma}^2}$，新的方差为$\frac{\sigma_z^2\overline{\sigma}^2}{\sigma_z^2+\overline{\sigma}^2}$,可以与$x_{k-1}$对应理解，后面再利用$x_{k}$预测$\overline{x}_{k+1}$，利用测量$z_{k+1}$把$\overline{x}_{k+1}$更新为$x_{k+1}$($\bold{注意：带上划线的是预测量，还需要根据测量量对预测量进行更新，得到真正的输出结果}$)

卡尔曼增益：对公式$\frac{\sigma_z^2\overline{\mu} + \overline{\sigma}^2\mu_z}{\sigma_z^2+\overline{\sigma}^2}$和公式$\frac{\sigma_z^2\overline{\sigma}^2}{\sigma_z^2+\overline{\sigma}^2}$进行变形：
$$
\begin{aligned}
\frac{\sigma_z^2\overline{\mu} + \overline{\sigma}^2\mu_z}{\sigma_z^2+\overline{\sigma}^2}&=(\frac{\overline{\sigma}^2}{\sigma_z^2+\overline{\sigma}^2})\mu_{z}+(\frac{\sigma_z^2}{\sigma_z^2+\overline{\sigma}^2})\overline{\mu}\\&=K\mu_{z}+(1-K)\overline{\mu}\\&=\overline{\mu}+K(\mu_{z}-\overline{\mu})
\end{aligned}
$$

$$\begin{aligned}
\frac{\sigma_z^2\overline{\sigma}^2}{\sigma_z^2+\overline{\sigma}^2}=(1-K)\overline{\sigma}^2
\end{aligned}
$$
利用卡尔曼增益对公式进行抽象化，设x的方差对应P，$f_x$的方差对应Q，测量方差对应R
则

预测：
$$\begin{aligned}
\overline{x}&=x+dx\\
\overline{P}&=P+Q
\end{aligned}
$$

更新：
$$\begin{aligned}
y &= z - \overline{x}\\
K&=\frac{\overline{P}}{\overline{P} + R}\\
x&=\overline{x}+Ky\\
P&=(1-K)\overline{P}
\end{aligned}
$$

## 扩展到多维：
预测：
$$\begin{aligned}
\overline{x}&=Fx+Bu\\
\overline{P}&=FPF^T+Q
\end{aligned}
$$

更新：
$$\begin{aligned}
y &= z - H\overline{x}\\
K&=\overline{P}H^T(H\overline{P}H^T + R)^{-1}\\
x&=\overline{x}+Ky\\
P&=(I-KH)\overline{P}
\end{aligned}
$$

![BiFPN](images/deeplearning/feature%20integration/BiFPN.png)