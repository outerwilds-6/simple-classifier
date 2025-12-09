from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import PolynomialFeatures, FunctionTransformer, KBinsDiscretizer
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.pipeline import make_pipeline
from scipy.ndimage import gaussian_filter

# 加载Iris数据集
iris = load_iris()
X = iris.data[:, 2:]  # 选择petal length和petal width
y = iris.target

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 定义四种模型
models = {
    'Logistic Regression (RBF)': make_pipeline(
        FunctionTransformer(lambda X: np.sqrt(np.abs(X))),
        LogisticRegression(max_iter=500)
    ),
    'Logistic Regression (Binned)': make_pipeline(
        KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='uniform'),
        LogisticRegression(max_iter=500)
    ),
    'Logistic Regression (Spline)': make_pipeline(
        PolynomialFeatures(degree=3),
        LogisticRegression(max_iter=500)
    ),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
}

# ===== 关键修正：统一坐标轴范围 =====
x_min, x_max = X[:, 0].min(), X[:, 0].max()
y_min, y_max = X[:, 1].min(), X[:, 1].max()

x_range = x_max - x_min
y_range = y_max - y_min
max_range = max(x_range, y_range)

x_center = (x_min + x_max) / 2
y_center = (y_min + y_max) / 2

x_min_adj = x_center - max_range / 2
x_max_adj = x_center + max_range / 2
y_min_adj = y_center - max_range / 2
y_max_adj = y_center + max_range / 2

xx, yy = np.meshgrid(
    np.arange(x_min_adj - 0.5, x_max_adj + 0.5, 0.05),
    np.arange(y_min_adj - 0.5, y_max_adj + 0.5, 0.05)
)
# ===== 修正结束 =====

# 创建图形（4行4列，8x6英寸）
fig, axs = plt.subplots(4, 4, figsize=(8, 6), dpi=100)

# ===== 关键修改：调整 colorbar 位置和大小 =====
# 原 cbar_ax: [0.05, 0.1, 0.02, 0.8] -> 缩小高度并上移
cbar_ax = fig.add_axes([0.1, 0.35, 0.02, 0.6])  # y=0.2 (上移), height=0.6 (缩小)

# ===== 新增：创建三个对应三类颜色的 colorbar =====
# 类0 (绿色)
cbar_ax0 = fig.add_axes([0.03, 0.08, 0.02, 0.2])  # y=0.1, height=0.2
# 类1 (橙色)
cbar_ax1 = fig.add_axes([0.11, 0.08, 0.02, 0.2])  # y=0.133, height=0.2
# 类2 (蓝色)
cbar_ax2 = fig.add_axes([0.19, 0.08, 0.02, 0.2])  # y=0.166, height=0.2

# 创建三类颜色的 cmap
green_cmap = mcolors.LinearSegmentedColormap.from_list('green_cmap', ['white', 'green'], N=256)
orange_cmap = mcolors.LinearSegmentedColormap.from_list('orange_cmap', ['white', 'orange'], N=256)
blue_cmap = mcolors.LinearSegmentedColormap.from_list('blue_cmap', ['white', 'blue'], N=256)

# ===== 训练并绘制图表 =====
for model_idx, (model_name, model) in enumerate(models.items()):
    # 训练模型
    model.fit(X_train, y_train)
    
    # 获取概率
    probs = model.predict_proba(np.c_[xx.ravel(), yy.ravel()])
    probs = probs.reshape(xx.shape[0], xx.shape[1], 3)
    
    # **1. 每一类的概率图（统一使用蓝色渐变，RGB混合方式）**
    for i in range(3):
        ax = axs[model_idx, i]
        
        # 创建蓝色渐变颜色数组（白色→深蓝）
        blue_array = np.zeros((xx.shape[0], xx.shape[1], 3))
        blue_array[:, :, 0] = 1 - probs[:, :, i]  # 红色通道
        blue_array[:, :, 1] = 1 - probs[:, :, i]  # 绿色通道
        blue_array[:, :, 2] = 1  # 蓝色通道
        
        # 绘制RGB混合效果
        ax.imshow(
            blue_array, 
            extent=(x_min_adj - 0.5, x_max_adj + 0.5, y_min_adj - 0.5, y_max_adj + 0.5), 
            origin='lower',
            alpha=0.8
        )
        
        # 真实类别点
        ax.scatter(
            X[y == i, 0], X[y == i, 1], 
            c='white',
            edgecolors='k', 
            marker='o', 
            s=10, 
            alpha=1
        )
        
        # 确保正方形 + 保留边框
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(x_min_adj - 0.5, x_max_adj + 0.5)
        ax.set_ylim(y_min_adj - 0.5, y_max_adj + 0.5)
        ax.set_aspect('equal')
        
        # 保留并设置边框
        for spine in ax.spines.values():
            spine.set_edgecolor('k')
            spine.set_linewidth(1)
        
        # 标题
        ax.set_title(f'Class {i}', fontsize=8)

    # **2. Max Class 概率叠加图**
    ax = axs[model_idx, 3]
    
    # 创建叠加颜色
    color_array = np.zeros((xx.shape[0], xx.shape[1], 3))
    for i in range(3):
        rgb = mcolors.to_rgb(['green', 'orange', 'blue'][i])
        color_array += probs[:, :, i, np.newaxis] * np.array(rgb)
    
    # 绘制叠加图
    ax.imshow(
        color_array, 
        extent=(x_min_adj - 0.5, x_max_adj + 0.5, y_min_adj - 0.5, y_max_adj + 0.5), 
        origin='lower',
        alpha=0.8
    )
    
    # 真实类别点
    for i in range(3):
        ax.scatter(
            X[y == i, 0], X[y == i, 1], 
            c=['green', 'orange', 'blue'][i],
            edgecolors='k', 
            marker='o', 
            s=10, 
            alpha=1
        )
    
    # 确保正方形 + 保留边框
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(x_min_adj - 0.5, x_max_adj + 0.5)
    ax.set_ylim(y_min_adj - 0.5, y_max_adj + 0.5)
    ax.set_aspect('equal')
    
    # 保留并设置边框
    for spine in ax.spines.values():
        spine.set_edgecolor('k')
        spine.set_linewidth(1)
    
    # 标题
    ax.set_title('Max Class', fontsize=8)

# **关键修改：模型名称竖直显示（旋转90度）**
for model_idx, model_name in enumerate(models.keys()):
    fig.text(
        0.32,
        1 - (model_idx + 0.8) * 0.222,
        model_name,
        fontsize=6.5,
        rotation=90,
        verticalalignment='center',
        horizontalalignment='center',
        color='black'
    )

# ===== 创建 colorbar =====
# 原概率 colorbar (缩小并上移)
cbar = fig.colorbar(
    plt.cm.ScalarMappable(cmap=blue_cmap), 
    cax=cbar_ax, 
    orientation='vertical', 
    label='Probability'
)
cbar.ax.yaxis.set_label_position('left')
cbar.ax.tick_params(labelsize=7)

# 三个类别 colorbar (下方)
cbar0 = fig.colorbar(
    plt.cm.ScalarMappable(cmap=green_cmap), 
    cax=cbar_ax0, 
    orientation='vertical', 
    label='Class 0'
)
cbar0.ax.yaxis.set_label_position('left')
cbar0.ax.tick_params(labelsize=5)

cbar1 = fig.colorbar(
    plt.cm.ScalarMappable(cmap=orange_cmap), 
    cax=cbar_ax1, 
    orientation='vertical', 
    label='Class 1'
)
cbar1.ax.yaxis.set_label_position('left')
cbar1.ax.tick_params(labelsize=5)

cbar2 = fig.colorbar(
    plt.cm.ScalarMappable(cmap=blue_cmap), 
    cax=cbar_ax2, 
    orientation='vertical', 
    label='Class 2'
)
cbar2.ax.yaxis.set_label_position('left')
cbar2.ax.tick_params(labelsize=5)

# 调整布局
plt.subplots_adjust(
    left=0.35,
    right=0.95,
    bottom=0.05,
    top=0.92,
    hspace=0.1,
    wspace=0.1
)

plt.suptitle('Iris Classification: Probability Distributions', fontsize=14, y=0.98)
plt.show()