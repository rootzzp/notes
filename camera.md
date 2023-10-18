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
对于归一化平面上的点$(x, y, 1)$，受畸变的影响变为$(x_{dist}, y_{dist}, 1)$\
径向畸变(通常只用$k_1,k_2$)：
$$x_{dist} = x(1+k_1r^2+k_2r^4+k_3r^6)$$
$$y_{dist} = y(1+k_1r^2+k_2r^4+k_3r^6)$$
切向畸变($p_1,p_2$)：
$$x_{dist} = x + 2p_1xy+p_2(r^2+2x^2)$$
$$y_{dist} = y + 2p_2xy+p_1(r^2+2y^2)$$

组合起来就是：
$$x_{dist} = x(1+k_1r^2+k_2r^4+k_3r^6) + 2p_1xy+p_2(r^2+2x^2)$$
$$y_{dist} = y(1+k_1r^2+k_2r^4+k_3r^6) + 2p_2xy+p_1(r^2+2y^2)$$

### 计算流程：
- 计算目标图像（去畸变）上的点$(u,v)$在归一化平面上的坐标$(x,y)$