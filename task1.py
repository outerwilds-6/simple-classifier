from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import PolynomialFeatures, FunctionTransformer, KBinsDiscretizer
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.pipeline import make_pipeline
from scipy.ndimage import gaussian_filter  # 引入高斯滤波器

# 加载Iris数据集
iris = load_iris()
X = iris.data[:, 2:4]  # 选择petal length和petal width进行可视化
y = iris.target

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 定义四种模型：Logistic Regression (RBF, Binned, Spline), Gradient Boosting
models = {
    'Logistic Regression (RBF)': make_pipeline(
        FunctionTransformer(lambda X: np.sqrt(np.abs(X))),  # 处理负值，确保为非负
        LogisticRegression(max_iter=200)
    ),
    'Logistic Regression (Binned)': make_pipeline(
        KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='uniform'),  # 分箱特征
        LogisticRegression(max_iter=200)
    ),
    'Logistic Regression (Spline)': make_pipeline(
        PolynomialFeatures(degree=3),  # 样条特征
        LogisticRegression(max_iter=200)
    ),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
}

# 创建网格，用于决策边界可视化
xx, yy = np.meshgrid(np.arange(X[:, 0].min() - 1, X[:, 0].max() + 1, 0.1),
                     np.arange(X[:, 1].min() - 1, X[:, 1].max() + 1, 0.1))

# 设置基色：将三个类别都改成蓝色的调色板
class_colors = ['#1f77b4', '#3b5998', '#4682b4']  # 自定义蓝色：从浅到深

# 创建图形，画概率图
fig, axs = plt.subplots(4, 4, figsize=(10, 10))  # 四种模型，每个模型四张图（class0, class1, class2, max class）

# 训练并绘制图表
for model_idx, (model_name, model) in enumerate(models.items()):
    # 训练模型
    model.fit(X_train, y_train)
    
    # 使用 predict_proba 获取每个点的概率
    probs = model.predict_proba(np.c_[xx.ravel(), yy.ravel()])
    probs = probs.reshape(xx.shape[0], xx.shape[1], 3)  # reshape to (height, width, classes)

    # **1. 每一类的概率图**
    for i, class_prob in enumerate(probs.transpose(2, 0, 1)):  # i corresponds to each class
        ax = axs[model_idx, i]  # Use axs[model_idx, 0], axs[model_idx, 1], axs[model_idx, 2]
        
        # 使用高斯滤波器平滑数据
        smoothed_class_prob = gaussian_filter(class_prob, sigma=1.0)  # 你可以调整sigma来控制平滑程度
        
        # 绘制平滑后的概率图
        levels = np.linspace(0, 1, 20)  # 增加level数量来让渐变更平滑
        contour = ax.contourf(xx, yy, smoothed_class_prob, levels=levels, alpha=0.7, cmap=mcolors.LinearSegmentedColormap.from_list(
            f'class_{i}_colormap', ['white', class_colors[i]], N=256))
        
        # 只显示当前类别的点
        ax.scatter(X[y == i, 0], X[y == i, 1], c=[class_colors[i]], edgecolors='k', marker='o', s=50, alpha=1)
        
        # 设置标题和标签
        ax.set_title(f'{model_name} - Class {i} Probability')
        ax.set_xlabel('Petal Length')
        ax.set_ylabel('Petal Width')

    # **2. 最大类别图（Max class）**
    max_class = np.argmax(probs, axis=2)  # 获取每个点的最大概率类别
    axs[model_idx, 3].imshow(max_class, extent=(xx.min(), xx.max(), yy.min(), yy.max()), origin='lower',
                             cmap=plt.cm.colors.ListedColormap(class_colors), alpha=0.6)
    
    # 显示所有类别的点，在最大类别图中
    for i in range(3):
        axs[model_idx, 3].scatter(X[y == i, 0], X[y == i, 1], c=[class_colors[i]], edgecolors='k', marker='o', s=50, alpha=1)

    axs[model_idx, 3].set_title(f'{model_name} - Max Class Prediction')
    axs[model_idx, 3].set_xlabel('Petal Length')
    axs[model_idx, 3].set_ylabel('Petal Width')

    # **3. 特征重要性图**
    if hasattr(model, 'coef_'):  # Logistic Regression
        feature_importance = np.abs(model.named_steps['logisticregression'].coef_[0])
        ax = axs[model_idx, 2]
        ax.bar(range(len(feature_importance)), feature_importance, color='blue')
        ax.set_title(f'{model_name} - Feature Importance')
        ax.set_xlabel('Feature')
        ax.set_ylabel('Importance')
        ax.set_xticks(range(len(feature_importance)))
        ax.set_xticklabels(['Petal Length', 'Petal Width'])

    elif hasattr(model, 'feature_importances_'):  # Gradient Boosting
        feature_importance = model.feature_importances_
        ax = axs[model_idx, 2]
        ax.bar(range(len(feature_importance)), feature_importance, color='green')
        ax.set_title(f'{model_name} - Feature Importance')
        ax.set_xlabel('Feature')
        ax.set_ylabel('Importance')
        ax.set_xticks(range(len(feature_importance)))
        ax.set_xticklabels(['Petal Length', 'Petal Width'])

# 添加统一的colorbar在最左侧
fig.colorbar(contour, ax=axs[:, 0], shrink=0.8, orientation='vertical', label='Probability')

# 调整布局
plt.tight_layout()
plt.show()
