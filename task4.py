import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from matplotlib.lines import Line2D

# 加载Iris数据集
iris = datasets.load_iris()
X = iris.data
y = iris.target

# 选择所有三类样本：Setosa(0)、Versicolor(1)和Virginica(2)
mask = (y == 0) | (y == 1) | (y == 2)
X_selected = X[mask]  # 先选择行
X_selected = X_selected[:, [1, 2, 3]]  # 再选择列: Sepal Width, Petal Length, Petal Width
y_selected = y[mask]

# 训练随机森林模型
rf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=5)
rf.fit(X_selected, y_selected)

# 创建网格点用于概率预测
x_min, x_max = X_selected[:, 0].min() - 0.5, X_selected[:, 0].max() + 0.5
y_min, y_max = X_selected[:, 1].min() - 0.5, X_selected[:, 1].max() + 0.5
z_min, z_max = X_selected[:, 2].min() - 0.5, X_selected[:, 2].max() + 0.5

# 创建xoy平面的网格 (z = z_min + 0.2)
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
fixed_z = z_min + 0.2
grid_points_xoy = np.c_[xx.ravel(), yy.ravel(), np.full(xx.ravel().shape, fixed_z)]

# 创建yoz平面的网格 (x = x_min + 0.2)
yy_yoz, zz_yoz = np.meshgrid(np.linspace(y_min, y_max, 100), np.linspace(z_min, z_max, 100))
fixed_x = x_min + 0.2
grid_points_yoz = np.c_[np.full(yy_yoz.ravel().shape, fixed_x), yy_yoz.ravel(), zz_yoz.ravel()]

# 创建xoz平面的网格 (y = y_min + 0.2)
xx_xoz, zz_xoz = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(z_min, z_max, 100))
fixed_y = y_min + 0.2
grid_points_xoz = np.c_[xx_xoz.ravel(), np.full(xx_xoz.ravel().shape, fixed_y), zz_xoz.ravel()]

# 创建3D图表
fig = plt.figure(figsize=(14, 10))
ax = fig.add_subplot(projection='3d')

# 绘制所有三类数据点
ax.scatter(
    X_selected[y_selected == 0, 0], X_selected[y_selected == 0, 1], X_selected[y_selected == 0, 2],
    c='blue', label='Setosa (Class 0)', s=50, alpha=0.9, edgecolor='black'
)
ax.scatter(
    X_selected[y_selected == 1, 0], X_selected[y_selected == 1, 1], X_selected[y_selected == 1, 2],
    c='green', label='Versicolor (Class 1)', s=50, alpha=0.9, edgecolor='black'
)
ax.scatter(
    X_selected[y_selected == 2, 0], X_selected[y_selected == 2, 1], X_selected[y_selected == 2, 2],
    c='red', label='Virginica (Class 2)', s=50, alpha=0.9, edgecolor='black'
)

# 为xoy平面创建概率感知的填充
all_probs_xoy = rf.predict_proba(grid_points_xoy)
max_class_xoy = np.argmax(all_probs_xoy, axis=1)
max_probs_xoy = np.max(all_probs_xoy, axis=1)  # 最大概率值
max_class_grid_xoy = max_class_xoy.reshape(xx.shape)
max_probs_grid_xoy = max_probs_xoy.reshape(xx.shape)

# 为每个类别创建半透明填充，透明度与概率成正比
for i, (color_name, color_rgb) in enumerate(zip(['blue', 'green', 'red'], 
                                               [(0, 0, 1), (0, 1, 0), (1, 0, 0)])):
    # 创建当前类别的概率掩码
    class_mask = (max_class_grid_xoy == i)
    # 创建一个基础概率网格，应用一个函数增强视觉效果
    prob_grid = np.zeros_like(max_probs_grid_xoy)
    prob_grid[class_mask] = max_probs_grid_xoy[class_mask]
    
    # 应用非线性变换增强视觉效果（概率平方，使低概率区域更透明）
    alpha_grid = prob_grid ** 1.5
    
    # 绘制轮廓线
    ax.contour(xx, yy, prob_grid, zdir='z', offset=z_min, 
               colors=color_name, alpha=0.7, levels=[0.5], linewidths=2)
    
    # 使用contourf创建多级概率填充
    levels = [0.3, 0.5, 0.7, 0.9]
    alphas = [0.1, 0.2, 0.3, 0.4]  # 透明度随概率增加
    
    for level, alpha in zip(levels, alphas):
        mask = (prob_grid >= level)
        if np.any(mask):
            # 创建一个临时网格，只包含当前概率级别以上的区域
            temp_grid = np.zeros_like(prob_grid)
            temp_grid[mask] = 1
            
            # 创建自定义颜色映射，控制透明度
            rgba_color = list(color_rgb) + [alpha]
            ax.contourf(xx, yy, temp_grid, zdir='z', offset=z_min,
                       colors=[rgba_color], levels=[0.5, 1.5], alpha=alpha)

# 为yoz平面创建概率感知的填充
all_probs_yoz = rf.predict_proba(grid_points_yoz)
max_class_yoz = np.argmax(all_probs_yoz, axis=1)
max_probs_yoz = np.max(all_probs_yoz, axis=1)
max_class_grid_yoz = max_class_yoz.reshape(yy_yoz.shape)
max_probs_grid_yoz = max_probs_yoz.reshape(yy_yoz.shape)

for i, (color_name, color_rgb) in enumerate(zip(['blue', 'green', 'red'], 
                                               [(0, 0, 1), (0, 1, 0), (1, 0, 0)])):
    class_mask = (max_class_grid_yoz == i)
    prob_grid = np.zeros_like(max_probs_grid_yoz)
    prob_grid[class_mask] = max_probs_grid_yoz[class_mask]
    
    alpha_grid = prob_grid ** 1.5
    
    # 修正参数顺序
    ax.contour(prob_grid, yy_yoz, zz_yoz, zdir='x', offset=x_min, 
               colors=color_name, alpha=0.7, levels=[0.5], linewidths=2)
    
    # 概率感知填充
    levels = [0.3, 0.5, 0.7, 0.9]
    alphas = [0.1, 0.2, 0.3, 0.4]
    
    for level, alpha in zip(levels, alphas):
        mask = (prob_grid >= level)
        if np.any(mask):
            temp_grid = np.zeros_like(prob_grid)
            temp_grid[mask] = 1
            
            rgba_color = list(color_rgb) + [alpha]
            # 注意正确的参数顺序
            ax.contourf(temp_grid, yy_yoz, zz_yoz, zdir='x', offset=x_min,
                       colors=[rgba_color], levels=[0.5, 1.5], alpha=alpha)

# 为xoz平面创建概率感知的填充
all_probs_xoz = rf.predict_proba(grid_points_xoz)
max_class_xoz = np.argmax(all_probs_xoz, axis=1)
max_probs_xoz = np.max(all_probs_xoz, axis=1)
max_class_grid_xoz = max_class_xoz.reshape(xx_xoz.shape)
max_probs_grid_xoz = max_probs_xoz.reshape(xx_xoz.shape)

for i, (color_name, color_rgb) in enumerate(zip(['blue', 'green', 'red'], 
                                               [(0, 0, 1), (0, 1, 0), (1, 0, 0)])):
    class_mask = (max_class_grid_xoz == i)
    prob_grid = np.zeros_like(max_probs_grid_xoz)
    prob_grid[class_mask] = max_probs_grid_xoz[class_mask]
    
    alpha_grid = prob_grid ** 1.5
    
    # 修正参数顺序
    ax.contour(xx_xoz, prob_grid, zz_xoz, zdir='y', offset=y_min, 
               colors=color_name, alpha=0.7, levels=[0.5], linewidths=2)
    
    # 概率感知填充
    levels = [0.3, 0.5, 0.7, 0.9]
    alphas = [0.1, 0.2, 0.3, 0.4]
    
    for level, alpha in zip(levels, alphas):
        mask = (prob_grid >= level)
        if np.any(mask):
            temp_grid = np.zeros_like(prob_grid)
            temp_grid[mask] = 1
            
            rgba_color = list(color_rgb) + [alpha]
            # 注意正确的参数顺序
            ax.contourf(xx_xoz, temp_grid, zz_xoz, zdir='y', offset=y_min,
                       colors=[rgba_color], levels=[0.5, 1.5], alpha=alpha)

# 创建自定义图例用于决策边界
legend_elements = [
    Line2D([0], [0], color='blue', lw=2, label='Setosa Boundary'),
    Line2D([0], [0], color='green', lw=2, label='Versicolor Boundary'),
    Line2D([0], [0], color='red', lw=2, label='Virginica Boundary')
]

# 创建概率深度图例
prob_legend_elements = [
    plt.Rectangle((0, 0), 1, 1, fc=(0, 0, 1, 0.4), edgecolor='none', label='High Probability (>=0.9)'),
    plt.Rectangle((0, 0), 1, 1, fc=(0, 0, 1, 0.2), edgecolor='none', label='Medium Probability (>=0.7)'),
    plt.Rectangle((0, 0), 1, 1, fc=(0, 0, 1, 0.1), edgecolor='none', label='Low Probability (>=0.5)')
]

# 添加数据点图例
point_legend = ax.legend(loc='upper left', bbox_to_anchor=(-0.33, 0.7))
ax.add_artist(point_legend)  # 保存第一个图例

# 添加决策边界图例
boundary_legend = ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(-0.33, 0.45))
ax.add_artist(boundary_legend)

# 添加概率深度图例
ax.legend(handles=prob_legend_elements, loc='upper left', bbox_to_anchor=(-0.33, 0.25), title='Probability Depth')

# 设置坐标轴标签和范围
ax.set_xlabel('Sepal Width', fontsize=12)
ax.set_xlim(x_min, x_max)
ax.set_ylabel('Petal Length', fontsize=12)
ax.set_ylim(y_min, y_max)
ax.set_zlabel('Petal Width', fontsize=12)
ax.set_zlim(z_min, z_max)

# 设置图形标题
ax.set_title('Random Forest Decision Regions with Probability-Aware Depth', fontsize=16, y=1)

# 设置视角
ax.view_init(elev=25, azim=45)

# 确保3D图形中各坐标轴的比例一致
ax.set_box_aspect([1, 1, 1])

plt.tight_layout()
plt.show()