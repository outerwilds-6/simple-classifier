# simple-classifier

## ✨ Iris 分类模型可视化项目

本项目通过随机森林模型对Iris数据集进行三维概率分布可视化分析，包含**4个核心任务**（task1-4），分别实现不同维度的决策边界展示与概率深度分析。

## 📚 项目结构

├── task1.py    # 基础三分类可视化

├── task2.py    # 二分类三维决策边界可视化

├── task3.py    # 二分类三位概率分布可视化

└── task4.py    # 三分类三维决策边界与概率分布可视化

## 💻 环境配置

为确保项目运行顺利，建议先使用 **`conda`** 安装虚拟环境，再通过 **`pip`** 安装项目所需的依赖包。

这些方法已在 **Windows** 平台上测试过。如在其他操作系统上遇到问题，请开 **`issue`** 来帮助我完善该项目。

### 🛠️ 安装 Python 环境

建议使用 **`conda`** 安装 **`Python 3.11`** 虚拟环境。如果已安装 **`conda`**，可以跳过此步骤。

在命令行中运行以下命令创建一个新的虚拟环境：

```shell
conda create --prefix .\\env python=3.11
```

若出现 `conda` 不存在之类的提示，建议切换到 `conda` 安装目录后运行下面指令完成 **shell association**，然后切回仓库目录重试虚拟环境部署指令。

```shell
conda init
```

### 🔧安装依赖包

如果你使用 `conda` 管理虚拟环境，执行这条命令进入 `Python` 环境：

```shell
conda activate .\\env
```

然后安装项目所需的包：

```shell
pip install -r requirements.txt
```

## 🕹️运行可视化任务

运行任意 `task.py` 打开可视化视窗。

例如：

```shell
python task1.py
```