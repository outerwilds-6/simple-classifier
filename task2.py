import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.svm import SVC

# 加载Iris数据集
iris = datasets.load_iris()
X = iris.data
y = iris.target

# 选择Setosa(0)和Virginica(2)的样本
mask = (y == 0) | (y == 2)
X_selected = X[mask]  # 先筛选行
X_selected = X_selected[:, [1, 2, 3]]  # [Sepal Width, Petal Length, Petal Width]
y_selected = y[mask]

# 训练线性SVM
svm = SVC(kernel='linear', C=1.0)
svm.fit(X_selected, y_selected)

# 获取决策边界系数
w = svm.coef_[0]
b = svm.intercept_[0]

# 创建3D图
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# 绘制数据点（增大点尺寸+黑色轮廓）
ax.scatter(
    X_selected[y_selected == 0, 0],  # Sepal Width
    X_selected[y_selected == 0, 1],  # Petal Length
    X_selected[y_selected == 0, 2],  # Petal Width
    c='blue', label='Setosa', s=50, alpha=0.8, edgecolor='black', linewidth=0.5
)
ax.scatter(
    X_selected[y_selected == 2, 0],  # Sepal Width
    X_selected[y_selected == 2, 1],  # Petal Length
    X_selected[y_selected == 2, 2],  # Petal Width
    c='red', label='Virginica', s=50, alpha=0.8, edgecolor='black', linewidth=0.5
)

# 用原始数据范围计算决策边界（不扩展范围）
x_min, x_max = X_selected[:, 0].min(), X_selected[:, 0].max()
y_min, y_max = X_selected[:, 1].min(), X_selected[:, 1].max()
xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 50),
    np.linspace(y_min, y_max, 50)
)
zz = (-b - w[0]*xx - w[1]*yy) / w[2]

# 绘制决策边界平面（保持原始大小）
ax.plot_surface(xx, yy, zz, alpha=0.3, color='green', edgecolor='k')

# 添加标签和标题
ax.set_xlabel('Sepal Width')
ax.set_ylabel('Petal Length')
ax.set_zlabel('Petal Width')
ax.set_title('Iris: Setosa vs Virginica (3D Boundary)', fontsize=16, y=1.05)
ax.legend()

# 保持1:1:1立方体比例（关键！）
ax.set_box_aspect([1, 1, 1])

plt.tight_layout()
plt.show()