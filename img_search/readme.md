# 深度学习之图片搜索（img_search）

把「卷积神经网络」课程知识落地为一条可运行的图片搜索流水线：**图像去噪 → 商品分类 → 相似商品检索**，并附一个 Flask 演示界面。

> 整体项目说明见 [../README.md](../README.md)；Web 模块细节见 [web/README.md](web/README.md)。

---

## 一、模块导航

| 目录 | 职责 | 核心产物 |
| --- | --- | --- |
| [common/](common/) | 公共层：图库、标签表、随机种子工具 | `dataset/`（24853 张）、`fashion-labels.csv`、`utils.py` |
| [image_denoising/](image_denoising/) | 卷积自编码器去噪 | `denoiser.pt` |
| [image_classification/](image_classification/) | CNN 五分类 | `classifier.pt` |
| [image_similarity/](image_similarity/) | 卷积编码器 + 向量检索 | `encoder.pt`、`decoder.pt` |
| [web/](web/) | Flask 应用，串起三个模型 | `web_app.py` + `index.html` |
| [test/](test/) | 学习与验证区 | 实验 notebook、ChromaDB 用法 |

三个算法模块统一采用六文件结构，便于横向对照：

```
xxx_config.py    # 路径、尺寸、超参、权重文件名、类别名
xxx_data.py      # Dataset 定义 + create_datasets() 划分
xxx_model.py     # 网络结构
xxx_engine.py    # train_one_epoch / validate / test_step / evaluate
xxx_train.py     # 训练入口（按验证集最优保存权重）
xxx_test.py      # 测试入口（加载权重、推理、可视化）
```

---

## 二、整体架构

```
                     ┌─────────────────────────────┐
                     │      common/dataset/        │
                     │  24853 张 64×64 RGB 商品图   │
                     └──────────────┬──────────────┘
                                    │ ImageDataset
        ┌───────────────────────────┼───────────────────────────┐
        ▼                           ▼                           ▼
┌───────────────┐          ┌───────────────┐          ┌─────────────────┐
│ image_denoising│          │image_classification│     │ image_similarity│
│  加噪 → 去噪   │          │  图 → 5 类概率  │          │  图 → 512 维向量 │
│   Conv-AE     │          │     CNN       │          │ ConvEncoder + DB│
└───────┬───────┘          └───────┬───────┘          └────────┬────────┘
        │ denoiser.pt              │ classifier.pt             │ encoder.pt
        └──────────────┬───────────┴────────────┬──────────────┘
                       ▼                        ▼
                  ┌──────────────────────────────────┐
                  │            web/                  │
                  │  上传 → 去噪 → 分类 → 相似商品     │
                  └──────────────────────────────────┘
```

**关联关系**：Web 层先把用户图片送去噪（清洗输入），再把去噪结果分别送进分类器与编码器；编码器输出的向量在 ChromaDB 中做近邻查询，返回 Top-5 相似图。去噪是后两个任务的**前置步骤**，不是可选项。

---

## 三、数据约定

| 项 | 说明 |
| --- | --- |
| 图片 | `common/dataset/{id}.jpg`，文件名中的数字即图片 id |
| 尺寸 | 统一 `Resize((64, 64))` + `ToTensor()`，即 `(3, 64, 64)` |
| 标签 | `common/fashion-labels.csv`，两列 `id,target`，按行序与排序后的文件名一一对应 |
| 类别 | `0 上衣 / 1 鞋 / 2 包 / 3 下装 / 4 手表` |
| 噪声 | `noisy = clamp(image + NOISE_FACTOR × randn, 0, 1)` |

> ⚠️ 64 是 8 的整数倍。三个网络都含 3 次 `MaxPool2d(2,2)`，输入边长必须满足 `2³ = 8 | 边长`，否则 `68 → 34 → 17 → 8` 的向下取整会让输出尺寸回不到原图（`64 ≠ 68`）。

---

## 四、运行约定

模块内部使用**同目录导入**（`from denoising_config import *`）与**相对路径**（`../common/dataset/`），因此运行前必须切到模块目录：

```bash
# 训练
cd img_search/image_denoising     && ../../.venv/bin/python denoising_train.py
cd img_search/image_classification && ../../.venv/bin/python classification_train.py
cd img_search/image_similarity    && ../../.venv/bin/python similarity_train.py

# 测试
cd img_search/image_denoising     && ../../.venv/bin/python denoising_test.py
cd img_search/image_classification && ../../.venv/bin/python classification_test.py

# Web（端口 9000）
cd img_search/web && ../../.venv/bin/python web_app.py
```

从仓库根目录直接运行会报 `ModuleNotFoundError: No module named 'xxx_config'` 或 `FileNotFoundError: '../common/dataset/'`。

### 训练超参速查

| 模块 | 损失 | 优化器 | epoch | 关键配置 |
| --- | --- | --- | --- | --- |
| 去噪 | `MSELoss` | Adam, `1e-3` | 30 | `NOISE_FACTOR = 0.5` |
| 分类 | `CrossEntropyLoss` | AdamW, `1e-3` | 20 | `class_names` 五类 |
| 检索 | `MSELoss` | AdamW, `1e-3` | 30 | 嵌入 512 维、`CHROMA_INSERT_BATCH_SIZE = 5000` |

---

## 五、模块速览

### image_denoising —— 卷积自编码器去噪

通道 `3→32→16→8→8→16→32→3`，空间 `64→32→16→8→16→32→64`，输出层 Sigmoid。瓶颈为 `8×8×8`。训练目标是原图，因此属于**自监督学习**：标签就是输入自身，模型被迫学会忽略随机噪声、保留结构。

### image_classification —— CNN 分类

`Conv2d(3→8) → Pool → Conv2d(8→16) → Pool → Flatten → Linear(4096→5)`，两处 `MaxPool2d(2,2)` 使 `64→16`，故全连接输入为 `16 × 16 × 16 = 4096`。标签对齐依赖 `sorted_alphanum()`：保证 `2.jpg` 排在 `10.jpg` 之前。

### image_similarity —— 嵌入与检索

`ConvEncoder`：6 个卷积块，通道 `3→16→32→64→128→256→512`，6 次池化后 `64×64 → 1×1`，`squeeze` 得 `(N, 512)`。训练时与 `ConvDecoder` 构成对称自编码器联合优化；**检索时只保留编码器**，把图库编码入 ChromaDB；查询时同法编码，取 Top-K。

```python
CHROMA_BACKEND_PATH = 'chroma_backend'
IMAGE_COLLECTION_NAME = "image_collection"
CHROMA_INSERT_BATCH_SIZE = 5000
```

### web —— Flask 演示

四个路由：`/`、`/denoising`、`/classification`、`/simimages`，另有 `/logo/<file>`、`/pictures/<file>` 与 `static_folder='../common/dataset'` 三个静态通道。前端强制「先上传 → 再去噪 → 才能分类/检索」。

---

## 六、已知缺口

| 位置 | 状态 | 影响 |
| --- | --- | --- |
| `image_similarity/similarity_embeddings.py` | **未提交** | Web 启动即 `ModuleNotFoundError`，需实现 `create_embeddings` / `get_embedding_collection` / `search_similar_image_ids` |
| `image_similarity/similarity_test.py` | 空文件 | 检索模块缺少独立测试入口，评估仍在 Web 与 notebook 中进行 |
| `pyproject.toml` | 未声明 `flask`、`chromadb` | `uv sync` 后仍需手动 `uv add flask chromadb` |
| `common/dataset/` | 已 gitignore | 新克隆仓库需自行放置图库（331MB） |

`similarity_embeddings` 的实现可参考 [test/chroma/](test/chroma/) 下的 6 个 notebook：客户端与持久化、collection 创建、CRUD、自定义 `EmbeddingFunction`（把 `ConvEncoder` 包成嵌入函数）、相似查询。

---

## 七、常见报错

| 报错 | 原因 | 处理 |
| --- | --- | --- |
| `No module named 'common'` | `sys.path` 缺 `img_search` | 脚本内已 `sys.path.append(parents[1])`，确认运行的是最新文件 |
| `No module named 'xxx_config'` | 未切到模块目录 | `cd img_search/<模块>` 后运行 |
| `FileNotFoundError: '../common/dataset/'` | 相对路径依赖工作目录 | 切到模块目录，或用 `Path(__file__).resolve().parents[1]` 定位 |
| `size of tensor a (64) must match ... (68)` | 输入边长不是 8 的倍数 | `Resize` 改为 64 / 72 / 96 |
| `cannot unpack non-iterable NoneType` | `__getitem__` 忘记 `return` | 返回 `(noisy, clean)` 元组 |
| `No module named 'flask'` | 依赖未安装 | `uv add flask chromadb` |

更多排查见 [../README.md](../README.md) 第六章。
