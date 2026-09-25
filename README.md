# easy-deeplearning

一套面向深度学习的入门学习资料，包含**教程文档（Markdown）**、**手写代码练习题（Jupyter Notebook）**与**配套答案**，以及练习所使用的数据集。

内容以 PyTorch 为主线，按「张量 → 神经网络基础 → 神经网络的学习 → 学习的技巧 → 卷积神经网络」的顺序递进，每个主题都配有与文档章节一一对应的动手练习。

---

## 目录结构

```
easy-deeplearning/
├── data/                       # 练习所用的数据集
├── docs/                       # 教程文档（Markdown，含配图）
│   ├── 01-pytorch/
│   ├── 02-nerual-network-basic/
│   ├── 03-neural-learn/
│   ├── 04-learn-skill/
│   └── 05-cnn/
├── pytorch_exercise/
│   ├── exercise/               # 练习题（题目 + 代码占位，用于手写练习）
│   ├── answer/                 # 参考答案（完整可运行代码）
│   └── exercise_hand/          # 个人手写练习区（已被 .gitignore 忽略）
├── img_search/                 # 图片搜索项目（预留目录）
├── main.py                     # 项目入口示例脚本
├── pyproject.toml              # 依赖与项目配置
└── uv.lock                     # 依赖锁定文件
```

---

## 环境准备

项目使用 [uv](https://docs.astral.sh/uv/) 管理依赖，要求 **Python >= 3.13**。

```bash
# 1. 安装 uv（如已安装可跳过）
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. 在项目根目录安装依赖（会自动创建 .venv）
uv sync
```

主要依赖：

| 包 | 用途 |
| --- | --- |
| `torch` / `torchvision` | 深度学习框架与视觉工具 |
| `torchsummary` | 查看模型各层输出形状 |
| `pandas` / `scikit-learn` | 数据处理与特征工程 |
| `matplotlib` | 绘图 |
| `ipykernel` | Jupyter 内核 |

验证环境：

```bash
uv run python -c "import torch, pandas, sklearn; print(torch.__version__)"
```

---

## 使用方法

### 1. 运行练习题

在 **VS Code / Cursor** 中打开 `pytorch_exercise/exercise/` 下的 notebook，右上角选择解释器为项目根目录下的 `.venv`，然后按 cell 顺序执行即可。

也可以使用命令行：

```bash
uv run jupyter lab
```

> 若提示找不到 `jupyter lab`，先安装：`uv add --dev jupyterlab`

**练题流程建议**：先在 `pytorch_exercise/exercise/` 中对着题目手写代码，再与 `pytorch_exercise/answer/` 中同名（带 `_答案` 后缀）的参考答案对照。也可以在 `pytorch_exercise/exercise_hand/` 中新建文件练习（该目录不会进入版本库）。

### 2. 重要：工作目录

所有 notebook 内部使用**相对路径**读取数据（形如 `../../data/house_prices.csv`）。因此运行 notebook 时，工作目录必须是**该 notebook 所在的文件夹**：

```
pytorch_exercise/exercise/  →  ../../data/  →  <项目根>/data/
pytorch_exercise/answer/    →  ../../data/  →  <项目根>/data/
```

在 VS Code 中直接打开 notebook 并运行即满足该条件；若用 Jupyter 启动，请在 `pytorch_exercise/exercise/`（或 `answer/`）目录下启动。

### 3. 运行示例脚本

```bash
uv run python main.py
# 输出：Hello from easy-deeplearning!
```

---

## 教程文档（docs/）

每个目录下包含一份 Markdown 文档与配套图片（`images/`），对应教材的一个章节。

| 目录 | 章节 | 主要内容 |
| --- | --- | --- |
| [01-pytorch](docs/01-pytorch/pytorch.md) | 第 2 章 PyTorch 简介 | 张量创建、转换、数值计算、运算函数、索引、形状操作、拼接 |
| [02-nerual-network-basic](docs/02-nerual-network-basic/02-neural-network-basic.md) | 第 3 章 神经网络基础 | 神经网络构成、全连接层、激活函数、搭建网络、加载模型、手写数字识别 |
| [03-neural-learn](docs/03-neural-learn/03-neural-learn.md) | 第 4 章 神经网络的学习 | 损失函数、随机梯度下降、数据集创建与分批、反向传播、自动微分、应用案例 |
| [04-learn-skill](docs/04-learn-skill/04-learn-skill.md) | 第 5 章 学习的技巧 | 深度网络的问题、参数更新优化、参数初始化、正则化、房价预测案例 |
| [05-cnn](docs/05-cnn/05-cnn.md) | 第 6 章 卷积神经网络 | CNN 概述、卷积层、池化层、深度卷积网络、服装分类案例 |

---

## 代码练习（pytorch_exercise/）

每个 notebook 均为「一道题一个 Markdown 说明 + 一个代码 cell」的形式。答案版与练习版**题目完全相同**，仅多出参考实现。

| 练习 | 章 | 题量 | 涉及知识点 |
| --- | --- | --- | --- |
| [01PyTorch张量练习](pytorch_exercise/exercise/01PyTorch张量练习.ipynb) | 2.3 ~ 2.9 | 31 | 张量创建/转换/计算/索引/形状变换/拼接 |
| [02神经网络基础练习](pytorch_exercise/exercise/02神经网络基础练习.ipynb) | 3 | 17 | 全连接层、激活函数、`nn.Sequential`、模型保存与加载 |
| [03神经网络的学习练习](pytorch_exercise/exercise/03神经网络的学习练习.ipynb) | 4 | 22 | 各类损失函数、SGD、`Dataset`/`DataLoader`、反向传播、自动微分 |
| [04学习的技巧练习](pytorch_exercise/exercise/04学习的技巧.ipynb) | 5 | 22 | 梯度消失/爆炸、优化器、学习率衰减、初始化、正则化、房价预测 |
| [04-02-学习的技巧练习](pytorch_exercise/exercise/04-02-学习的技巧.ipynb) | 5 | 20 | 第 5 章第二套练习，采用教材原始示例参数，侧重重写与数值验证 |
| [05卷积神经网络练习](pytorch_exercise/exercise/05卷积神经网络.ipynb) | 6 | 15 | 手写卷积/池化、`nn.Conv2d`、CNN 搭建与训练、残差连接 |

参考答案位于 `pytorch_exercise/answer/` 下，文件名与练习题相同并追加 `_答案`，例如：

```
pytorch_exercise/exercise/04-02-学习的技巧.ipynb
pytorch_exercise/answer/04-02-学习的技巧_答案.ipynb
```

**第 5 章的两套练习的区别**：

- `04学习的技巧` 使用改造过的参数（目标函数 `0.1x₁² + 2x₂²`、起点 `(-6.0, 1.5)` 等），综合性更强。
- `04-02-学习的技巧` 直接沿用教材原始示例（目标函数 `0.05x₁² + x₂²`、起点 `(-7, 2)`、`test_size=0.2` 等），并强调「手写实现 vs PyTorch 官方实现」的数值比对（如手写五种优化器与 `torch.optim` 的最大误差需小于 `1e-6`）。

---

## 数据集（data/）

| 文件 | 说明 |
| --- | --- |
| `house_prices.csv` | Kaggle *House Prices* 数据集，1460 行 × 81 列，目标列为 `SalePrice`，含数值型与类别型特征。用于第 4、5 章的房价预测案例 |
| `house_prices_cols.txt` | 上述数据集的字段中文说明（80 行） |
| `train.csv` | Digit Recognizer 手写数字数据集，42000 行 × 785 列（第 1 列为标签，其余 784 列为 28×28 像素亮度）。用于第 3、4、6 章的图像任务 |
| `nn_example.pt` | 预先训练好的手写数字识别模型参数（保存于 GPU，无 GPU 时加载需加 `map_location="cpu"`） |
| `duck.jpg` | 示例图片，用于第 6 章卷积/池化实验 |
| `poems.txt` | 古诗文本语料（预留资源） |

---

## 其他说明

- `img_search/` 为「深度学习之图片搜索」项目预留目录，当前仅有说明文件。
- `bak/`、`pytorch_exercise/exercise_hand/` 与 `.venv/` 均已在 `.gitignore` 中忽略，不会提交到版本库。
