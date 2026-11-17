# 图片搜索 Web 应用（img_search/web）

基于 **Flask** 的「智能商品识别系统」演示界面，把 `image_denoising`、`image_classification`、`image_similarity` 三个深度学习模块串成一条完整的图片搜索流水线：

```
上传图片 → 图像去噪 → 商品分类 → 相似商品检索
```

---

## 一、目录结构

```
web/
├── web_app.py              # Flask 入口：加载模型、向量库，提供 4 个业务接口
├── templates/
│   └── index.html          # 单页前端（原生 JS，无构建步骤）
├── logo/
│   └── logo.jpg            # 品牌 Logo
├── pictures/               # 前端图标资源（按钮、标题图等，含 智图寻宝.png）
└── __init__.py             # 使 web 成为包（内容为空）
```

---

## 二、启动方式

### 2.1 前置条件

| 依赖 | 说明 |
| --- | --- |
| Python 环境 | 项目根目录的 `.venv`（Python 3.13） |
| `flask` / `chromadb` | 已写入 `pyproject.toml`，`uv sync` 即可安装 |
| 三个模型权重 | `../image_denoising/denoiser.pt`、`../image_classification/classifier.pt`、`../image_similarity/encoder.pt` |
| 向量库模块 | `../image_similarity/similarity_embeddings.py`（提供 `create_embeddings()`、`get_embedding_collection()`、`search_similar_image_ids()`） |
| 图库数据 | `../common/dataset/`（24853 张图，文件名 `0.jpg` ~ `24853.jpg`） |

同步依赖（网络较慢时加国内镜像）：

```bash
uv sync
# 或
uv sync -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
```

### 2.2 启动

```bash
cd img_search/web
PYTHONPATH=.. ../../.venv/bin/python web_app.py
```

> **两个前提**：
> 1. 必须在 `web/` 目录下执行。脚本内所有路径都相对于当前工作目录，例如 `os.path.join('..', PACKAGE_NAME, 'denoiser.pt')` 和 `static_folder='../common/dataset'`。
> 2. 必须设置 `PYTHONPATH=..`（或把 `img_search` 设为 IDE 的 Sources Root）。`python web_app.py` 只会把 `web/` 加入 `sys.path`，无法导入 `image_denoising`、`image_classification`、`image_similarity` 等兄弟包。

启动日志（正常情况）：

```
启动应用
正在加载去噪模型
去噪模型加载完毕
正在加载分类模型
分类模型加载完毕
正在加载嵌入模型
嵌入模型加载完毕
正在加载向量集合
向量集合加载完毕
 * Running on http://127.0.0.1:9000
```

浏览器访问 <http://127.0.0.1:9000>。

### 2.3 首次运行：先建向量库

`web_app.py` 启动时调用 `get_embedding_collection(encoder)` 读取已有集合，而建库语句 `create_embeddings(encoder)` 被注释掉了（第 79 行）。**首次部署**时，需要先执行一次建库：

```python
# web_app.py 第 79-80 行
similarity_embeddings.create_embeddings(encoder)      # 首次运行：取消注释
collection = similarity_embeddings.get_embedding_collection(encoder)
```

2.4 万张图按 `CHROMA_INSERT_BATCH_SIZE = 5000` 分 5 批写入，建库完成后把该行重新注释，避免每次启动都重复计算嵌入。

---

## 三、接口说明

所有接口均为 `POST`，使用 `multipart/form-data`，文件字段名统一为 **`image`**。

| 路由 | 入参 | 返回 | 说明 |
| --- | --- | --- | --- |
| `GET /` | — | HTML | 渲染 `index.html` |
| `POST /denoising` | `image` | JSON | 返回加噪图与去噪图（Base64 PNG） |
| `POST /classification` | `image` | 纯文本 | `您搜索的商品类型是：xxx` |
| `POST /simimages` | `image` | JSON | `{"indices_list": [id1, ..., id5]}` |
| `GET /logo/<filename>` | — | 图片 | 从 `./logo` 读取 |
| `GET /pictures/<filename>` | — | 图片 | 从 `./pictures` 读取 |
| `GET /<filename>` | — | 图片 | 静态图库（`../common/dataset`） |

### 3.1 `/denoising`

```bash
curl -X POST http://127.0.0.1:9000/denoising -F "image=@your.jpg"
```

```json
{
  "noisy_img": "iVBORw0KGgoAAAANSUhEUg...",
  "denoised_image": "iVBORw0KGgoAAAANSUhEUg..."
}
```

处理流程：`Image.open → convert("RGB") → Resize(64,64) → ToTensor` → 叠加高斯噪声（`NOISE_FACTOR`）→ `ConvDenoiser` 推理 → `CHW→HWC`、`×255`、转 `uint8` → PNG → Base64。

### 3.2 `/classification`

```bash
curl -X POST http://127.0.0.1:9000/classification -F "image=@your.jpg"
```

```
您搜索的商品类型是：上衣
```

类别名取自 `classification_config.classification_names`，共 5 类：上衣 / 鞋 / 包 / 下装 / 手表。

### 3.3 `/simimages`

```bash
curl -X POST http://127.0.0.1:9000/simimages -F "image=@your.jpg"
```

```json
{"indices_list": [1024, 3391, 88, 12045, 7]}
```

前端用 `dataset/{index}.jpg` 拼接出图片地址，因此**向量库里保存的 id 必须与图库文件名（数字）一致**。

---

## 四、前端交互

`templates/index.html` 是单页应用，四个按钮与状态耦合：

```
[上传图片]  ──▶ 原始图片预览
     │
[图像去噪]  ── 校验原始图已上传 ──▶ 噪声图片 / 去噪结果 两栏
     │
[商品分类]  ── 校验「请先去噪！」──▶ 用去噪结果请求 /classification ──▶ 文本框
     │
[相似商品]  ── 校验「请先去噪！」──▶ 用去噪结果请求 /simimages ──▶ 5 张相似图
```

要点：

- 上传与去噪走**原始图**；分类与检索走**去噪后的图**，所以顺序必须是「上传 → 去噪 → 分类 / 相似」；
- 去噪、分类、相似三个按钮都会 `alert` 提示前置条件未满足；
- 相似结果会先 `innerHTML = ''` 清空再渲染，不会累积上一次结果。

### 操作流程

1. 点击「选择文件」，选一张商品图 → 点「上传图片」，左侧出现原图预览；
2. 点「图像去噪」，中间显示「噪声图片」，右侧显示「去噪结果」；
3. 点「商品分类」，下方文本框显示预测类别；
4. 点「相似商品」，底部展示 Top-5 相似商品缩略图。

---

## 五、代码结构要点

`web_app.py` 全程只有「启动期加载」和「请求期推理」两个阶段：

```python
# 启动期（模块级，导入即执行）
app = Flask(__name__, static_folder='../common/dataset')
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
denoiser   = ConvDenoiser().load_state_dict(...)
classifier = Classifier().load_state_dict(...)
encoder    = ConvEncoder().load_state_dict(...)
collection = similarity_embeddings.get_embedding_collection(encoder)

# 请求期
with torch.no_grad():
    output = model(input_tensor.to(device))
```

- **模型只加载一次**，常驻内存，单次请求无需重复 `torch.load`；
- 推理统一用 `torch.no_grad()`，不建计算图；
- 设备自动选择：有 CUDA 用 CUDA，否则 CPU；
- 分类输出用 `np.argmax` 取索引，再查 `classification_names` 映射为中文；
- JSON 响应使用 `json.dumps(...)` 而非 `jsonify`，并显式返回 `(body, 200, headers)` 三元组——注意 `ContentType` 少了连字符（正确写法是 `Content-Type`），浏览器能正常工作，但如果用严格的客户端调用，建议改为 `jsonify`。

---

## 六、常见问题排查

### Q1. `ModuleNotFoundError: No module named 'flask'`

```bash
uv sync
# 或指定清华镜像
uv sync -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
```

### Q2. `ModuleNotFoundError: No module named 'image_similarity'` / `'image_denoising'`

不是文件缺失，而是**模块搜索路径**问题：`python web_app.py` 只把 `web/` 放进 `sys.path`。

```bash
cd img_search/web
PYTHONPATH=.. ../../.venv/bin/python web_app.py
```

### Q2b. 向量集合为空 / `向量集合为空，请先调用 create_embeddings() 建库`

Web 只读取已有集合，不会自动建库。先建库：

```bash
cd img_search/image_similarity
../../.venv/bin/python similarity_test.py --build 500     # 先小规模验证
```

全量建库则在 `web_app.py` 中临时取消第 79 行 `similarity_embeddings.create_embeddings(encoder)` 的注释，启动一次后再注释回去。索引目录为 `image_similarity/chroma_backend`（以模块目录为基准，与 cwd 无关），建库与服务读取的是同一份。

### Q3. `FileNotFoundError: '../image_denoising/denoiser.pt'`

两个原因：

1. 没有在 `web/` 目录下启动（路径是相对 cwd 的）→ `cd img_search/web` 后重跑；
2. 对应模型尚未训练 → 分别执行 `image_denoising/denoising_train.py`、`image_classification/classification_train.py`、`image_similarity/similarity_train.py`。

### Q4. `RuntimeError: Error(s) in loading state_dict ... size mismatch`

权重与当前网络结构不匹配：确认三个 `*_model.py` 未被改动，或重新训练生成权重。

### Q5. 相似商品显示为**破图**

`<img src="dataset/{index}.jpg">` 依赖两个条件：

- `static_folder='../common/dataset'` 能被正确解析（必须从 `web/` 启动）；
- collection 中的 id 就是文件名中的数字（建库时应使用 `int(filename.split('.')[0])` 作为 id）。

可在浏览器直接访问 `http://127.0.0.1:9000/1024.jpg` 验证静态资源是否可读。

### Q6. 点击分类/相似时提示「请先去噪！」

这是前端**预期行为**：两个功能强制以去噪结果为输入。先点「图像去噪」，等两栏图片出现后再操作。

### Q7. 端口 9000 被占用

```bash
lsof -i :9000            # 查看占用进程
kill -9 <PID>            # 或修改 web_app.py 最后一行的 port
```

### Q8. 上传大图后分类结果离谱

前端/后端都会 `Resize(64, 64)`，极小文字或细长商品会严重变形。建议上传主体居中、接近正方形的商品图。

---

## 七、与其余模块的关系

| 上游模块 | 本层使用的产物 |
| --- | --- |
| `image_denoising` | `ConvDenoiser` + `denoiser.pt`、`NOISE_FACTOR` |
| `image_classification` | `Classifier` + `classifier.pt`、`classification_names` |
| `image_similarity` | `ConvEncoder` + `encoder.pt`、`similarity_embeddings`、ChromaDB collection |
| `common` | `dataset/` 图库（静态资源与检索数据源） |

模型训练与评估请见各子模块的 `*_train.py` / `*_test.py`，整体项目说明见仓库根目录 `README.md`。
