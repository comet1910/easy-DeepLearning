# easy-deeplearning

一套以 **PyTorch** 为主线的深度学习学习与实践项目，由两大部分构成：

1. **核心深度学习模块**：系统教程（Markdown）+ 配套手写练习（Jupyter Notebook）+ 参考答案，覆盖「张量 → 神经网络基础 → 神经网络的学习 → 学习的技巧 → 卷积神经网络」。
2. **图片搜索实践模块**：把前面学到的 CNN、自编码器与向量检索知识，落地为一个可运行的「智能商品识别系统」——包含图像去噪、商品分类、相似图检索三个深度学习任务，以及一个 Flask Web 演示界面。

---

## 一、项目整体架构

### 1.1 两大模块的职责划分

| 模块 | 目录 | 职责 | 产出物 |
| --- | --- | --- | --- |
| **核心深度学习模块** | `docs/`、`pytorch_exercise/` | 讲解原理、提供动手练习与标准答案，回答「CNN 是什么、怎么训练」 | 5 章教程文档、6 套练习 + 6 套答案 |
| **图片搜索实践模块** | `img_search/` | 把原理工程化：数据管道、模型定义、训练/验证/测试引擎、权重管理、Web 服务 | 3 个可训练模型 + 4 个 `.pt` 权重 + 1 个 Web 应用 |

### 1.2 两个模块的关联关系

```
docs/  ──讲解──▶  pytorch_exercise/  ──工程化──▶  img_search/
（原理+配图）      （手写验证原理）              （模块化落地为产品）
```

- **知识承接**：第 6 章「卷积神经网络」中的卷积、池化、转置卷积、残差连接，正是 `img_search` 中三个模型的全部构件。
- **代码承接**：练习中的 `nn.Sequential` 堆叠、`Dataset`/`DataLoader`、`random_split`、MSE/交叉熵损失、Adam 优化思路，在 `img_search` 中被拆分为 `config / data / model / engine / train / test` 六层结构。
- **数据承接**：`data/train.csv`（手写数字 28×28）用于教学，`img_search/common/dataset/`（商品图 64×64）用于实践，两者形态一致（「第 1 列标签 + 像素」与「图片 + 标签 CSV」）。

### 1.3 图片搜索实践模块的内部分层

```
img_search/
├── common/                      # 公共层：数据集、标签、通用工具
│   ├── dataset/                 # 24853 张商品图（64×64 RGB，已 gitignore）
│   ├── fashion-labels.csv       # id,target 两列，与图片文件名一一对应
│   └── utils.py                 # seed_everything()、sorted_alphanum()
│
├── image_denoising/             # 模型 1：图像去噪（自编码器）
├── image_classification/        # 模型 2：商品分类（CNN 分类器）
├── image_similarity/            # 模型 3：图像嵌入 + 向量检索
│
├── web/                         # 应用层：Flask + 前端页面，串起三个模型
└── test/                        # 学习区：notebook 实验与 ChromaDB 用法验证
```

三个算法模块统一采用相同的六文件结构，便于对照阅读：

| 文件 | 作用 |
| --- | --- |
| `xxx_config.py` | 路径、尺寸、超参数、模型文件名、类别名映射 |
| `xxx_data.py` | 自定义 `Dataset` + `create_datasets()`（划分 train/val/test） |
| `xxx_model.py` | 网络结构定义 |
| `xxx_engine.py` | `train_one_epoch` / `validate` / `test_step` / `evaluate` |
| `xxx_train.py` | 训练入口：建数据 → 建模型 → 训练 → 按验证集最优保存权重 |
| `xxx_test.py` | 测试入口：加载权重 → 推理 → 可视化 / 计算指标 |

数据流：

```
原始图片 ──Resize(64,64)+ToTensor──▶ 张量 (3,64,64)
      ├─ 去噪任务：加噪 ──▶ (noisy, clean) ──▶ ConvDenoiser ──▶ 去噪图
      ├─ 分类任务：读标签 ──▶ (image, label) ──▶ Classifier ──▶ 5 类概率
      └─ 检索任务：恒等映射 ──▶ (image, image) ──▶ ConvEncoder ──▶ 512 维向量
```

---

## 二、图片搜索功能的技术实现原理

### 2.1 整体流程

```
用户上传图片
   │
   ▼
① 图像去噪  ConvDenoiser(Conv-AE)     ── 清洗输入，提升后续任务鲁棒性
   │
   ▼
② 商品分类  Classifier(CNN)           ── 输出 5 类：上衣/鞋/包/下装/手表
   │
   ▼
③ 相似检索  ConvEncoder + ChromaDB    ── 512 维嵌入 → 向量库近邻搜索 → Top-5
```

前端页面（`web/templates/index.html`）提供四个按钮：**上传图片 / 图像去噪 / 商品分类 / 相似商品**。分类与检索都要求先去噪，因此页面会校验「请先去噪！」。

### 2.2 模型一：图像去噪（卷积自编码器）

| 项目 | 说明 |
| --- | --- |
| 网络 | `ConvDenoiser`，编码器 3 层卷积 + 3 次下采样，解码器 3 层转置卷积 |
| 通道变化 | `3 → 32 → 16 → 8 → 8 → 16 → 32 → 3` |
| 空间变化 | `64 → 32 → 16 → 8 → 16 → 32 → 64` |
| 激活 | 隐藏层 ReLU，输出层 Sigmoid（把像素压回 `[0,1]`） |
| 噪声注入 | `noisy = clamp(image + 0.5 × randn, 0, 1)`（高斯噪声，训练与推理一致） |
| 损失函数 | `MSELoss`（逐像素回归） |
| 优化器 | Adam，`lr = 1e-3` |
| 权重文件 | `image_denoising/denoiser.pt` |

**原理**：自编码器把图像压缩到 8 通道 × 8 × 8 的瓶颈表示，再重建回原图。由于训练目标是「干净图」，网络被迫学习「忽略随机噪声、保留结构信息」，从而具备去噪能力。这属于**自监督学习**——标签就是输入自身。

> ⚠️ 关键约束：网络有 3 次 `MaxPool2d(2,2)`，输入边长必须是 **2³ = 8 的整数倍**，否则 `17 → 8` 这类向下取整会丢失像素，输出尺寸回不到原尺寸（`64 ≠ 68`）。

### 2.3 模型二：商品分类（CNN 分类器）

| 项目 | 说明 |
| --- | --- |
| 网络 | `Classifier`，2 个「卷积 + ReLU + 最大池化」块 + 全连接层 |
| 结构 | `Conv2d(3→8) → Pool → Conv2d(8→16) → Pool → Flatten → Linear(4096→5)` |
| 类别 | `0:上衣 1:鞋 2:包 3:下装 4:手表` |
| 损失函数 | `CrossEntropyLoss`（内含 Softmax） |
| 优化器 | AdamW，`lr = 1e-3`，20 个 epoch |
| 权重文件 | `image_classification/classifier.pt` |

**标签对齐方式**：`ImageLabelDataset` 用 `sorted_alphanum()` 对文件名排序（保证 `2.jpg < 10.jpg` 的自然顺序），再按 `fashion-labels.csv` 的行序取 `target`，两者一一对应。

### 2.4 模型三：相似图检索（嵌入 + 向量数据库）

| 项目 | 说明 |
| --- | --- |
| 网络 | `ConvEncoder` + `ConvDecoder`，构成一个对称卷积自编码器 |
| 嵌入维度 | `ConvEncoder`：6 个卷积块，通道 `3→16→32→64→128→256→512`，6 次池化后 `64×64 → 1×1`，`squeeze` 得到 **(N, 512)** 向量 |
| 训练目标 | 重建自身（`MSELoss`），迫使 512 维向量保留图像的**语义与外观特征** |
| 损失函数 | `MSELoss` |
| 优化器 | AdamW（编码器 + 解码器参数联合优化） |
| 权重文件 | `image_similarity/encoder.pt`、`image_similarity/decoder.pt` |

**检索算法**：训练完成后**丢弃解码器**，只保留编码器作为「特征提取器」，把图库中所有图片编码为 512 维向量并写入 **ChromaDB** 向量库；查询时把上传图编码为同维度向量，在库中做近邻搜索（默认余弦/欧氏距离，由 ChromaDB collection 配置决定），返回 Top-K 图片索引。

```
图库图片 ──ConvEncoder──▶ 512 维向量 ──▶ ChromaDB Collection（持久化）
查询图片 ──ConvEncoder──▶ 512 维向量 ──▶ collection.query(n_results=5) ──▶ 相似图 id
```

`similarity_config.py` 中的检索相关配置：

```python
CHROMA_BACKEND_PATH = 'chroma_backend'      # 向量库持久化目录
IMAGE_COLLECTION_NAME = "image_collection"  # collection 名称
CHROMA_INSERT_BATCH_SIZE = 5000             # 批量写入大小（2.4 万张图分 5 批）
```

### 2.5 Web 服务接口

`web/web_app.py` 基于 Flask，启动时一次性加载三个模型与向量集合，提供 4 个路由：

| 路由 | 方法 | 入参 | 返回 |
| --- | --- | --- | --- |
| `/` | GET | — | 前端页面 `index.html` |
| `/denoising` | POST | `image` 文件 | JSON：`noisy_img`、`denoised_image`（Base64 PNG） |
| `/classification` | POST | `image` 文件 | 纯文本：「您搜索的商品类型是：xxx」 |
| `/simimages` | POST | `image` 文件 | JSON：`indices_list`（相似图索引，前 5） |

另外通过 `static_folder='../common/dataset'` 与 `/logo/<filename>`、`/pictures/<filename>` 三个通道提供静态资源。

---

## 三、环境依赖与本地部署

### 3.1 环境要求

| 项目 | 要求 | 本项目实测版本 |
| --- | --- | --- |
| Python | `>= 3.13` | 3.13.3 |
| 包管理 | [uv](https://docs.astral.sh/uv/)（推荐） | 最新版 |
| 操作系统 | macOS / Linux / Windows | macOS |
| 硬件 | CPU 可跑；GPU（CUDA）自动启用 | Apple Silicon CPU |

### 3.2 依赖清单

**已在 `pyproject.toml` 中声明**（执行 `uv sync` 自动安装）：

| 包 | 版本要求 | 实测版本 | 用途 |
| --- | --- | --- | --- |
| `torch` | `>=2.14.0` | 2.14.0 | 深度学习框架 |
| `torchvision` | `>=0.29.0` | 0.29.0 | 图像变换 `Resize`/`ToTensor` |
| `numpy` | 随 torch | 2.5.3 | 数值计算 |
| `pandas` | `>=3.0.5` | 3.0.5 | 读取标签 CSV |
| `scikit-learn` | `>=1.9.1` | 1.9.1 | 数据处理（教程章节使用） |
| `matplotlib` | `>=3.11.2` | 3.11.2 | 训练结果可视化 |
| `tqdm` | `>=4.70.1` | 4.70.1 | 进度条 |
| `torchsummary` | `>=1.5.1` | 1.5.1 | 查看模型各层形状 |
| `ipykernel` | `>=7.3.0` | 7.3.0 | Jupyter 内核 |

**Web 与向量库相关（需手动安装）**：

```bash
uv add flask chromadb
# 或使用 pip
# .venv/bin/pip install flask chromadb
```

> 当前 `pyproject.toml` 未声明 `flask`、`chromadb`，运行 Web 服务前必须补装。

### 3.3 本地部署步骤

```bash
# 1. 克隆仓库
git clone <your-repo-url> easy-deeplearning
cd easy-deeplearning

# 2. 安装 uv（已安装可跳过）
curl -LsSf https://astral.sh/uv/install.sh | sh

# 3. 创建虚拟环境并安装依赖
uv sync

# 4. 安装 Web 依赖
uv add flask chromadb

# 5. 验证环境
.venv/bin/python -c "import torch, torchvision, pandas; print(torch.__version__, torchvision.__version__)"
```

**准备数据**（仓库中数据集已被 `.gitignore` 忽略，需本地放置）：

```
img_search/common/dataset/       # 24853 张商品图（0.jpg ~ 24853.jpg）
img_search/common/fashion-labels.csv
data/train.csv                   # 手写数字，教程练习用
data/house_prices.csv
data/fashion-mnist_train.csv     # 第 6 章服装分类用（可选，需自行下载）
data/fashion-mnist_test.csv
```

---

## 四、项目目录结构

```
easy-deeplearning/
├── data/                              # 教学数据集
│   ├── house_prices.csv               # Kaggle 房价数据（1460×81）
│   ├── house_prices_cols.txt          # 字段中文说明
│   ├── train.csv                      # 手写数字（42000×785）
│   ├── fashion-mnist_train.csv        # 服装数据（已 gitignore）
│   ├── fashion-mnist_test.csv         # 服装数据（已 gitignore）
│   ├── nn_example.pt                  # 预训练手写数字模型
│   ├── duck.jpg                       # 卷积/池化示例图
│   └── poems.txt                      # 古诗语料（预留）
│
├── docs/                              # 教程文档（Markdown + images/）
│   ├── 01-pytorch/pytorch.md
│   ├── 02-nerual-network-basic/02-neural-network-basic.md
│   ├── 03-neural-learn/03-neural-learn.md
│   ├── 04-learn-skill/04-learn-skill.md
│   └── 05-cnn/05-cnn.md
│
├── pytorch_exercise/                  # 动手练习
│   ├── exercise/                      # 6 个题目 notebook
│   ├── answer/                        # 6 个答案 notebook（*_答案.ipynb）
│   ├── docs/                          # 预留
│   └── exercise_hand/                 # 个人手写区（gitignore）
│
├── img_search/                        # 图片搜索实践
│   ├── common/
│   │   ├── dataset/                   # 商品图库（gitignore，331MB）
│   │   ├── fashion-labels.csv         # id,target 标签表
│   │   ├── utils.py                   # seed_everything / sorted_alphanum
│   │   └── __init__.py
│   │
│   ├── image_denoising/               # 子模块 1：去噪
│   │   ├── denoising_config.py
│   │   ├── denoising_data.py
│   │   ├── denoising_model.py
│   │   ├── denoising_engine.py
│   │   ├── denoising_train.py
│   │   ├── denoising_test.py
│   │   └── denoiser.pt
│   │
│   ├── image_classification/          # 子模块 2：分类
│   │   ├── classification_config.py
│   │   ├── classification_data.py
│   │   ├── classification_model.py
│   │   ├── classification_engine.py
│   │   ├── classification_train.py
│   │   ├── classification_test.py
│   │   └── classifier.pt
│   │
│   ├── image_similarity/              # 子模块 3：嵌入与检索
│   │   ├── similarity_config.py
│   │   ├── similarity_data.py
│   │   ├── similarity_model.py
│   │   ├── similarity_engine.py
│   │   ├── similarity_train.py
│   │   ├── similarity_test.py         # 预留（当前为空）
│   │   ├── encoder.pt
│   │   └── decoder.pt
│   │
│   ├── web/                           # Web 应用
│   │   ├── web_app.py                 # Flask 入口（端口 9000）
│   │   ├── templates/index.html
│   │   ├── logo/logo.jpg
│   │   └── pictures/*.png             # 前端图标
│   │
│   ├── test/                          # 学习/验证区
│   │   ├── 01-denoising_autoencoder.ipynb
│   │   ├── 02-denoising-test.ipynb
│   │   ├── 03-classification_test.ipynb
│   │   └── chroma/                    # ChromaDB 六个用法 notebook
│   │
│   └── readme.md
│
├── main.py                            # 最小入口示例
├── pyproject.toml                     # 依赖声明（uv 管理）
├── uv.lock                            # 依赖锁定
└── README.md
```

---

## 五、功能使用示例

### 5.0 通用约定

三个算法模块内部都使用**同目录导入**（如 `from denoising_config import *`）与**相对路径**（`../common/dataset/`）。因此运行脚本前**必须切到模块目录**：

```bash
cd img_search/image_denoising     # 或 image_classification / image_similarity
```

### 5.1 训练图像去噪模型

```bash
cd img_search/image_denoising
../../.venv/bin/python denoising_train.py
```

预期输出：

```
设置训练去噪模型的随机数种子, seed = 42
------------ 正在创建数据集 ------------
------------ 数据集创建完成 ------------
------------ 创建数据加载器 ------------
------------ 数据加载器创建完成 ------------
------------ 开始训练 ------------
  0%|          | 0/30 ...
----------> Epochs = 1, Training Loss : 0.012345 <----------
...验证集损失减小，保存模型...
```

权重保存为当前目录的 `denoiser.pt`（验证集损失创新低时才覆盖）。

### 5.2 训练商品分类模型

```bash
cd img_search/image_classification
../../.venv/bin/python classification_train.py
```

20 个 epoch，`CrossEntropyLoss`。权重保存为 `classifier.pt`。

### 5.3 训练相似图嵌入模型

```bash
cd img_search/image_similarity
../../.venv/bin/python similarity_train.py
```

30 个 epoch，编码器 + 解码器联合训练，权重分别保存为 `encoder.pt`、`decoder.pt`。

### 5.4 测试与可视化

```bash
cd img_search/image_denoising && ../../.venv/bin/python denoising_test.py
cd img_search/image_classification && ../../.venv/bin/python classification_test.py
```

- `denoising_test.py`：画 3 行 × 10 列对比图（原图 / 加噪图 / 去噪图），并输出测试集 MSE。
- `classification_test.py`：画 10 张测试图，打印真实标签与预测类别，并输出测试集准确率。

### 5.5 启动 Web 演示

```bash
cd img_search/web
../../.venv/bin/python web_app.py
```

浏览器访问 <http://127.0.0.1:9000>，操作顺序：

1. **上传图片** → 左侧显示原始图预览；
2. **图像去噪** → 中间两栏显示「噪声图片」「去噪结果」，前端自动把去噪结果作为后续输入；
3. **商品分类** → 下方文本框显示「您搜索的商品类型是：上衣/鞋/包/下装/手表」；
4. **相似商品** → 底部网格显示 Top-5 相似商品图。

> 启动 Web 前需保证 `image_similarity/similarity_embeddings.py`（向量库封装模块）存在，且向量库已建库。详见「常见问题 Q7」。

### 5.6 运行教程与练习

在 VS Code 中打开 `pytorch_exercise/exercise/` 下的 notebook，解释器选择 `.venv`，按 cell 顺序执行；工作目录必须是 notebook 所在文件夹（notebook 内使用 `../../data/...`）。

```bash
uv run jupyter lab        # 未安装：uv add --dev jupyterlab
```

### 5.7 自定义推理（脚本化调用）

```python
import torch, torchvision.transforms as T
from PIL import Image

from image_denoising.denoising_model import ConvDenoiser
from image_classification.classification_model import Classifier
from image_similarity.similarity_model import ConvEncoder

device = "cuda" if torch.cuda.is_available() else "cpu"
transform = T.Compose([T.Resize((64, 64)), T.ToTensor()])

img = transform(Image.open("test.jpg").convert("RGB")).unsqueeze(0).to(device)

# 去噪
denoiser = ConvDenoiser().to(device)
denoiser.load_state_dict(torch.load("image_denoising/denoiser.pt", map_location=device))
denoiser.eval()
clean = denoiser(img)

# 分类
classifier = Classifier().to(device)
classifier.load_state_dict(torch.load("image_classification/classifier.pt", map_location=device))
classifier.eval()
print(classifier(clean).argmax(dim=-1))   # 预测类别索引

# 嵌入
encoder = ConvEncoder().to(device)
encoder.load_state_dict(torch.load("image_similarity/encoder.pt", map_location=device))
encoder.eval()
print(encoder(clean).shape)               # torch.Size([1, 512])
```

---

## 六、常见问题排查

### Q1. `ModuleNotFoundError: No module named 'common'`

**原因**：直接运行脚本时，`sys.path[0]` 是脚本所在目录（如 `img_search/image_denoising`），而 `common` 包在上一级。

**解决**：模块内已用下列代码把 `img_search` 加入搜索路径，确认脚本顶部有：

```python
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from common.utils import *
```

若仍报错，检查运行的是否是**旧版本文件**，或在命令前加 `PYTHONPATH=..`。

### Q2. `ModuleNotFoundError: No module named 'denoising_config'`

**原因**：没有切到模块目录，`from denoising_config import *` 找不到同级文件。

**解决**：

```bash
cd img_search/image_denoising && ../../.venv/bin/python denoising_train.py
```

不要用 `python -m img_search.image_denoising.denoising_train`，否则同目录导入会失败。

### Q3. `FileNotFoundError: '../common/dataset/'`

**原因**：相对路径依赖**当前工作目录**。在仓库根目录运行时，`../common/dataset/` 指向仓库外层，自然不存在。

**解决**：切到模块目录运行；或把配置改为基于文件位置定位（`denoising_config.py` 已改为这种写法）：

```python
IMG_PATH = Path(__file__).resolve().parents[1] / "common" / "dataset"
```

### Q4. `RuntimeError: The size of tensor a (64) must match the size of tensor b (68)`

**原因**：网络含 3 次 `MaxPool2d(2,2)`，输入边长必须是 **8 的整数倍**。68 经 `68→34→17→8` 时被向下取整丢像素，解码后只能回到 64，与目标 68 不匹配。

**解决**：把 `Resize` 改成 8 的倍数（推荐 64，与 `IMG_H/IMG_W` 一致）：

```python
transforms = T.Compose([T.Resize((IMG_H, IMG_W)), T.ToTensor()])
```

若必须用任意尺寸，可在 `forward` 末尾插值回原尺寸：

```python
x = torch.nn.functional.interpolate(x, size=(h, w), mode="bilinear", align_corners=False)
```

### Q5. `UserWarning: Using a target size (...) that is different to the input size (...)`

**原因**：输入与目标尺寸不一致时 `MSELoss` 会静默广播，导致「所有样本都在和同一张图比较」，训练结果完全错误。常见于循环变量写错（如 `noisy_img` / `noisy_imgs` 混用），或把 `Dataset` 迭代结果（单张 `(3,64,64)`）当成了 `DataLoader` 的一个 batch（`(32,3,64,64)`）。

**解决**：确认 `model(...)` 与 `criterion(...)` 的两边都是 `(B, 3, 64, 64)`，并统一循环变量命名。

### Q6. `TypeError: cannot unpack non-iterable NoneType object`

**原因**：自定义 `Dataset.__getitem__` 忘记 `return`，隐式返回 `None`，循环解包时报错。

**解决**：

```python
return noisy_imgs, tensor_image
```

### Q7. `ModuleNotFoundError: No module named 'image_similarity.similarity_embeddings'`

**原因**：`web_app.py` 依赖 `image_similarity/similarity_embeddings.py`（ChromaDB 封装：`create_embeddings`、`get_embedding_collection`、`search_similar_image_ids`），该文件当前**尚未提交到仓库**。

**解决**：补齐该模块后再启动 Web。其实现可直接参考 `img_search/test/chroma/` 下 6 个 notebook（客户端 → collection → CRUD → 自定义 EmbeddingFunction 的完整用法）：

- `1_get_start.ipynb`、`2_client_test.ipynb`：客户端与持久化
- `4_collection_test.ipynb`：collection 创建
- `5_crud_test.ipynb`：增删改查
- `6_embedding_test.ipynb`：自定义 `EmbeddingFunction`（把 `ConvEncoder` 包成 chroma 的嵌入函数）
- `3_similarity_test.ipynb`：相似度查询

临时跳过 Web 时，可先只跑训练与测试脚本（5.1 ~ 5.4）。

### Q8. `ModuleNotFoundError: No module named 'flask'` / `'chromadb'`

**原因**：这两个包未写入 `pyproject.toml`，`uv sync` 不会安装。

**解决**：

```bash
uv add flask chromadb
```

### Q9. `NameError: name '__file__' is not defined`（notebook 中）

**原因**：`__file__` 只在 `.py` 模块中存在；Notebook 的 cell 以字符串方式执行，没有该变量。

**解决**：改用基于工作目录的定位：

```python
from pathlib import Path
import sys
sys.path.insert(0, str(Path.cwd().parent))
from common.utils import sorted_alphanum
```

### Q10. 训练很慢 / 显存不足

- 代码会自动选择设备：`torch.device("cuda" if torch.cuda.is_available() else "cpu")`；
- 显存不足时调小 `TRAIN_BATCH_SIZE`、`VAL_BATCH_SIZE`（`*_config.py`）；
- 数据集共 24853 张图、每轮全量训练，CPU 上耗时较长属正常现象；
- 先用小 `EPOCHS`（如 1）验证流程是否跑通，再放大。

### Q11. 模型权重加载失败（`size mismatch` / CUDA 相关报错）

- `size mismatch`：权重与当前网络结构不一致，确认 `*_model.py` 未被修改过；
- 在无 GPU 机器上加载 GPU 保存的权重（如 `data/nn_example.pt`），需加：

```python
torch.load("xxx.pt", map_location="cpu")
```

### Q12. 为什么 `git push` 提示文件超过 100MB？

`data/fashion-mnist_train.csv`（约 127MB）超过 GitHub 单文件限制。项目中已把它们加入 `.gitignore`，但**已进入提交历史的文件仍会被推送**。需从历史中移除（如 `git filter-repo` 或重置该提交），详见 Git 官方文档「Removing sensitive data」。

---

## 七、其他说明

- `img_search/test/` 是学习与调试区，包含去噪自编码器实验（`01`、`02`）与分类实验（`03`）notebook，以及 ChromaDB 用法验证；其中 `conv_autoencoder.pth`、`conv_denoiser.pth` 为实验产物。
- `img_search/image_similarity/similarity_test.py` 当前为空文件，检索评估逻辑仍在 Web 层与 notebook 中验证。
- `bak/`、`.venv/`、`pytorch_exercise/exercise_hand/`、`data/fashion-mnist_*.csv`、`img_search/common/dataset/` 均已在 `.gitignore` 中忽略。
- `pyproject.toml` 中的 `some-package>=0.1` 是占位依赖，可安全移除。
