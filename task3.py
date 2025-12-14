import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.svm import SVC
from matplotlib import cm

# 加载Iris数据集
iris = datasets.load_iris()
X = iris.data
y = iris.target

# 选择Setosa(0)和Virginica(2)的样本
mask = (y == 0) | (y == 2)
# 修复索引错误 - 分两步选择
X_selected = X[mask]  # 先选择行
X_selected = X_selected[:, [1, 2, 3]]  # 再选择列: Sepal Width, Petal Length, Petal Width
y_selected = y[mask]

# 训练SVM模型
svm = SVC(kernel='linear', C=1.0, probability=True)
svm.fit(X_selected, y_selected)

# 创建网格点用于概率预测
x_min, x_max = X_selected[:, 0].min() - 0.5, X_selected[:, 0].max() + 0.5
y_min, y_max = X_selected[:, 1].min() - 0.5, X_selected[:, 1].max() + 0.5
z_min, z_max = X_selected[:, 2].min() - 0.5, X_selected[:, 2].max() + 0.5

# 创建xoy平面的网格 (z = z_min + 0.2)
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
fixed_z = z_min + 0.2
grid_points_xoy = np.c_[xx.ravel(), yy.ravel(), np.full(xx.ravel().shape, fixed_z)]
probabilities_xoy = svm.predict_proba(grid_points_xoy)[:, 1]
prob_grid_xoy = probabilities_xoy.reshape(xx.shape)

# 创建yoz平面的网格 (x = x_min + 0.2)
yy_yoz, zz_yoz = np.meshgrid(np.linspace(y_min, y_max, 100), np.linspace(z_min, z_max, 100))
fixed_x = x_min + 0.2
grid_points_yoz = np.c_[np.full(yy_yoz.ravel().shape, fixed_x), yy_yoz.ravel(), zz_yoz.ravel()]
probabilities_yoz = svm.predict_proba(grid_points_yoz)[:, 1]
prob_grid_yoz = probabilities_yoz.reshape(yy_yoz.shape)

# 创建xoz平面的网格 (y = y_min + 0.2)
xx_xoz, zz_xoz = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(z_min, z_max, 100))
fixed_y = y_min + 0.2
grid_points_xoz = np.c_[xx_xoz.ravel(), np.full(xx_xoz.ravel().shape, fixed_y), zz_xoz.ravel()]
probabilities_xoz = svm.predict_proba(grid_points_xoz)[:, 1]
prob_grid_xoz = probabilities_xoz.reshape(xx_xoz.shape)

# 创建3D图表
fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(projection='3d')

# 绘制数据点
ax.scatter(
    X_selected[y_selected == 0, 0], X_selected[y_selected == 0, 1], X_selected[y_selected == 0, 2],
    c='blue', label='Setosa', s=40, alpha=0.9, edgecolor='black'
)
ax.scatter(
    X_selected[y_selected == 2, 0], X_selected[y_selected == 2, 1], X_selected[y_selected == 2, 2],
    c='red', label='Virginica', s=40, alpha=0.9, edgecolor='black'
)

# 在xoy平面上绘制等高线投影 (z = z_min)
cset = ax.contourf(xx, yy, prob_grid_xoy, zdir='z', offset=z_min, cmap=cm.coolwarm, alpha=0.7)

# 在yoz平面上绘制等高线投影 (x = x_min)
cset = ax.contourf(prob_grid_yoz, yy_yoz, zz_yoz, zdir='x', offset=x_min, cmap=cm.coolwarm, alpha=0.7)

# 在xoz平面上绘制等高线投影 (y = y_min)
cset = ax.contourf(xx_xoz, prob_grid_xoz, zz_xoz, zdir='y', offset=y_min, cmap=cm.coolwarm, alpha=0.7)

# 添加颜色条
mappable = cm.ScalarMappable(cmap=cm.coolwarm)
mappable.set_array(np.concatenate([probabilities_xoy, probabilities_yoz, probabilities_xoz]))
cbar = plt.colorbar(mappable, ax=ax, shrink=0.5, pad=0.1)
cbar.set_label('Probability of Virginica', fontsize=12)

# 设置坐标轴标签和范围
ax.set_xlabel('Sepal Width')
ax.set_xlim(x_min, x_max)
ax.set_ylabel('Petal Length')
ax.set_ylim(y_min, y_max)
ax.set_zlabel('Petal Width')
ax.set_zlim(z_min, z_max)

# 设置图形标题
ax.set_title('SVM Probability Projections on Three Coordinate Planes', fontsize=16, y=1)

# 设置视角
ax.view_init(elev=25, azim=45)

plt.tight_layout()
plt.show()