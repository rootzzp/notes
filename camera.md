# 针孔相机模型
几个重要的坐标系：
- 世界坐标系（点表示为$P_w$）
- 相机坐标系（点表示为$P_c$）
- 图像坐标系（点表示为$P_i$）
## 相机内参：
作用：将相机坐标系的点转换为图像坐标系，根据针孔模型
，假设相机坐标系中的点$P_c=(X_c,Y_c,Z_c)$,根据相似三角形原理，成像点$P^{\prime}_c=(X^{\prime}_c,Y^{\prime}_c,Z^{\prime}_c)$(由于物理成像平面和光心的距离为$f$，所以此处的$Z^{\prime}_c=f$)与其的关系为：

$$\frac{Z_c}{f}=\frac{X_c}{X^{\prime}_c}=\frac{Y_c}{Y^{\prime}_c}$$
那么：
$X^{\prime}_c=f\frac{X_c}{Z_c}$，$Y^{\prime}_c=f\frac{Y_c}{Z_c}$（均代表实际物理距离，后面把这个距离转换到像素需要除每个像素的物理长度）

图像坐标系定义：原点o位于图像的左上角，u轴向右与x轴平行，v轴向下与y轴平行。设像素坐标在u轴上缩放$\alpha$倍（$\frac{1}{每个像素在x方向的物理长度}$），在v轴上缩放了$\beta$倍（$\frac{1}{每个像素在y方向的物理长度}$）。同时，原点平移了$(c_x,c_y)$，那么
$$u=\alpha X^{\prime}_c=\alpha f\frac{X_c}{Z_c}+c_x$$
$$v=\beta Y^{\prime}_c=\beta f\frac{Y_c}{Z_c}+c_y$$
用$f_x,f_y分别表示\alpha f，\beta f$（单位是像素，意义是焦距的长度相当于多少个像素的尺寸），将上面的运算写成矩阵形式即：
$$\begin{pmatrix}
u\\
v\\
1\\
\end{pmatrix}=\frac{1}{Z_c}
\begin{pmatrix}
f_x&0&c_x\\
0&f_y&c_y\\
0&0&1\\
\end{pmatrix}
\begin{pmatrix}
X_c\\
Y_c\\
Z_c\\
\end{pmatrix}
$$
简写为：$Z_c P_i=KP_c$，其中K为相机内参

## 相机外参：
作用：将世界坐标系的点转换到相机坐标系
$$P_c=
\begin{pmatrix}
X_c\\
Y_c\\
Z_c\\
1\\
\end{pmatrix}=
TP_w=
\begin{pmatrix}
R&T\\
0&1\\
\end{pmatrix}
\begin{pmatrix}
X_w\\
Y_w\\
Z_w\\
1\\
\end{pmatrix}=
RP_w+T
$$
## 归一化平面
指距离相机坐标系原点为1米的平面（$Z_c=1$）,某个像素点在归一化平面上的坐标为（还是相机坐标系下的）：
$(x_{norm},y_{norm})=(X_c,Y_c)=(\frac{u-c_x}{f_x},\frac{u-c_y}{f_y})$

## 相机畸变模型
对于归一化平面上的点$(x, y, 1)$(相机坐标系下的点)，受畸变的影响变为$(x_{dist}, y_{dist}, 1)$\
径向畸变(通常只用$k_1,k_2$),其中$r=(x^2+y^2)$：
$$x_{dist} = x(1+k_1r^2+k_2r^4+k_3r^6)$$
$$y_{dist} = y(1+k_1r^2+k_2r^4+k_3r^6)$$
切向畸变($p_1,p_2$)：
$$x_{dist} = x + 2p_1xy+p_2(r^2+2x^2)$$
$$y_{dist} = y + 2p_2xy+p_1(r^2+2y^2)$$

组合起来就是：
$$x_{dist} = x(1+k_1r^2+k_2r^4+k_3r^6) + 2p_1xy+p_2(r^2+2x^2)$$
$$y_{dist} = y(1+k_1r^2+k_2r^4+k_3r^6) + 2p_2xy+p_1(r^2+2y^2)$$

### 计算流程：
- 计算目标图像（去畸变）上的点$(u,v)$在归一化平面上的坐标$(x,y)$：
$$(x,y)=(\frac{u-c_x}{f_x},\frac{u-c_y}{f_y})$$
- 利用畸变模型，计算$(x_{dist},y_{dist})$
- 将畸变后的归一化平面的点$(x_{dist},y_{dist})$，计算对应的图像坐标点:
$$u_{dist}=f_x x_{dist} + c_x$$
$$v_{dist}=f_y y_{dist} + c_y$$
- 然后利用插值算法，根据$(u_{dist},v_{dist})$找原图(去畸变前)对应的像素值填充目标图像(去畸变后)$(u,v)$中


# 鱼眼相机
利用多次折射提高视野范围
## 畸变模型
### 等距投影模型
[cite：paper](https://oulu3dvision.github.io/calibgeneric/Kannala_Brandt_calibration.pdf)\
![fisheye](/images/auto_drive/fisheye.png#pic_center)\
原文中的公式：
$$r(\theta)=k_1 \theta + k_2 \theta^3 + k_3 \theta^5 + k_4 \theta^7 + k_5 \theta^9$$
$$
\begin{pmatrix}
x\\
y\\
\end{pmatrix}=
r(\theta)
\begin{pmatrix}
cos \varphi \\
sin \varphi \\
\end{pmatrix}
$$
实际计算流程：
- 计算相机坐标系中的坐标$(x,y,z)$在归一化平面上的坐标：
$a = \frac{x}{z}$，$b = \frac{y}{z}$
- 计算$r^2$
$$r^2=a^2+b^2$$
- 计算$\theta$(与论文中的一致,1代表归一化平面到坐标系的z轴距离为1，其实也就是根据归一化平面的定义来的)
$$tan(\theta) = \frac{r}{1} \Rightarrow \theta = arctan(r)$$
- 计算$r(\theta)=k_1 \theta + k_2 \theta^3 + k_3 \theta^5 + k_4 \theta^7 + k_5 \theta^9$，等价于在opencv中的$\theta_d$
$$\theta_d=\theta(k_1 + k_2 \theta^2 + k_3 \theta^4 + k_4 \theta^6 + k_5 \theta^8)$$
- 根据畸变模型，计算畸变后的坐标，注意(a为归一化平面上的x,b为y)
$$
\begin{pmatrix}
x_{dist} \\
y_{dist}\\
\end{pmatrix}=
r(\theta)
\begin{pmatrix}
cos \varphi \\
sin \varphi \\
\end{pmatrix}=
\theta_d
\begin{pmatrix}
a/r \\
b/r \\
\end{pmatrix}=
$$
- 将畸变后的归一化平面的点$(x_{dist},y_{dist})$，计算对应的图像坐标点:
$$u_{dist}=f_x x_{dist} + c_x$$
$$v_{dist}=f_y y_{dist} + c_y$$
- 如果使用Unified camera model(opencv)，考虑多一个光心到投影点之间的平移$\alpha$
$$u_{dist}=f_x(x_{dist} + \alpha y_{dist}) + c_x$$
$$v_{dist}=f_y y_{dist} + c_y$$

# 鱼眼相机投到bev，得到虚拟相机图像
- 设置虚拟相机内参，外参数
- 遍历虚拟相机的每个像素$(u,v)$
- 根据虚拟相机内参矩阵，计算每个像素点对应的地面点在相机坐标系下的坐标(地面点到相机的距离z已知，所以可以直接带入内参矩阵计算出来),得到$(X_c,Y_c,Z_c)$，其中$Z_c$是已知量
$$Z_c
\begin{pmatrix}
u\\
v\\
1\\
\end{pmatrix}=
\begin{pmatrix}
f_x&0&c_x\\
0&f_y&c_y\\
0&0&1\\
\end{pmatrix}
\begin{pmatrix}
X_c\\
Y_c\\
Z_c\\
\end{pmatrix}
$$
- 利用虚拟相机的外参和真实鱼眼相机的外参，将虚拟相机坐标系下的坐标点$(X_c,Y_c,Z_c)$转换到鱼眼相机坐标系下的点$(x,y,z)$；其中$T_{camera2car}$是虚拟相机相机坐标到自车坐标的转换矩阵，$T_{car2camera}$是鱼眼相机自车坐标到相机坐标的转换矩阵
$$
\begin{pmatrix}
x\\
y\\
z\\
\end{pmatrix}=
T_{car2camera}*
T_{camera2car}*
\begin{pmatrix}
X_c\\
Y_c\\
Z_c\\
\end{pmatrix}
$$
- 利用上面的鱼眼畸变校准算法，计算鱼眼相机坐标系下的点$(x,y,z)$对应的去畸变前的图像坐标点，这个图像坐标点和$(u,v)$之间构成的一一映射就可以把鱼眼相机的图像转换到虚拟相机中的图像了

# 旋转矩阵：
- 绕x轴逆时针旋转
$$
\begin{pmatrix}
1&0&0\\
0&cos \theta&-sin \theta\\
0&sin \theta&cos \theta\\
\end{pmatrix}
$$
- 绕y轴逆时针旋转
$$
\begin{pmatrix}
cos \theta&0&sin \theta\\
0&1&0\\
-sin \theta&0&cos \theta\\
\end{pmatrix}
$$
- 绕z轴逆时针旋转
$$
\begin{pmatrix}
cos \theta&-sin \theta&0\\
sin \theta&cos \theta&0\\
0&0&1\\
\end{pmatrix}
$$