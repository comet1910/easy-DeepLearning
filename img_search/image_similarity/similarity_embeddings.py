# 图片嵌入与向量库封装：建库、取集合、相似检索
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from tqdm import tqdm

# 兼容两种运行方式：
# 1) web_app 以包方式导入（image_similarity.similarity_embeddings）
# 2) 在模块目录下直接运行脚本
_MODULE_DIR = Path(__file__).resolve().parent
_IMG_SEARCH_DIR = _MODULE_DIR.parent
for _path in (str(_MODULE_DIR), str(_IMG_SEARCH_DIR)):
    if _path not in sys.path:
        sys.path.append(_path)

import chromadb

from similarity_config import (
    CHROMA_BACKEND_PATH,
    CHROMA_INSERT_BATCH_SIZE,
    IMAGE_COLLECTION_NAME,
    IMG_H,
    IMG_W,
    IMG_PATH,
)
from common.utils import sorted_alphanum

# 检索时复用的编码器与设备（由 get_embedding_collection 注入）
_ENCODER = None
_DEVICE = torch.device("cpu")

# 图库中允许的图片后缀
_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def _backend_path():
    """向量库目录：相对路径固定以模块所在目录为基准，避免受运行目录影响。

    web_app 在 web/ 下启动、建库脚本在 image_similarity/ 下运行，
    若直接用相对路径会指向两个不同目录，导致 Web 读不到已建好的索引。
    """
    path = Path(CHROMA_BACKEND_PATH)
    return path if path.is_absolute() else _MODULE_DIR / path


def _get_client():
    """创建持久化客户端，向量保存在 CHROMA_BACKEND_PATH 目录。"""
    return chromadb.PersistentClient(path=str(_backend_path()))


def _list_image_files(image_dir=IMG_PATH):
    """按自然顺序列出图库中的图片路径。"""
    names = sorted_alphanum(os.listdir(image_dir))
    return [
        os.path.join(image_dir, name)
        for name in names
        if name.lower().endswith(_IMAGE_EXTENSIONS)
    ]


def _normalize_embedding(vectors):
    """统一转成 float32 的 ndarray，避免 chroma 对数据类型挑剔。"""
    return np.asarray(vectors, dtype=np.float32)


def create_embeddings(encoder, image_dir=IMG_PATH, batch_size=CHROMA_INSERT_BATCH_SIZE,
                      device=None, limit=None):
    """遍历图库，把每张图编码为向量写入 ChromaDB。

    参数：
        encoder: ConvEncoder 实例
        image_dir: 图库目录
        batch_size: 单批编码/写入数量
        device: 推理设备，缺省自动选择
        limit: 只取前 limit 张图（用于快速冒烟测试）

    注意：若集合已存在会先删除，保证 id 与向量一一对应、不产生重复。
    """
    global _ENCODER, _DEVICE

    _DEVICE = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _ENCODER = encoder.to(_DEVICE).eval()

    client = _get_client()
    existing = [c.name if hasattr(c, "name") else c for c in client.list_collections()]
    if IMAGE_COLLECTION_NAME in existing:
        client.delete_collection(IMAGE_COLLECTION_NAME)

    collection = client.create_collection(
        name=IMAGE_COLLECTION_NAME,
        configuration={"hnsw": {"space": "cosine"}},
    )

    paths = _list_image_files(image_dir)
    if limit is not None:
        paths = paths[:limit]
    if not paths:
        raise ValueError(f"图库目录为空或没有图片：{image_dir}")

    transform = T.Compose([T.Resize((IMG_H, IMG_W)), T.ToTensor()])

    for start in tqdm(range(0, len(paths), batch_size), desc="写入向量库"):
        batch_paths = paths[start:start + batch_size]
        images = torch.stack([
            transform(Image.open(path).convert("RGB")) for path in batch_paths
        ]).to(_DEVICE)

        with torch.no_grad():
            vectors = _normalize_embedding(_ENCODER(images).cpu().numpy())

        # id 使用文件名中的数字，前端据此拼出 dataset/{id}.jpg
        ids = [str(int(Path(path).stem)) for path in batch_paths]
        collection.upsert(ids=ids, embeddings=vectors)

    print(f"向量库构建完成：{collection.count()} 条，目录 {CHROMA_BACKEND_PATH}")
    return collection


def get_embedding_collection(encoder=None, device=None):
    """获取（必要时创建）向量集合；传入编码器时缓存下来供检索使用。"""
    global _ENCODER, _DEVICE

    if encoder is not None:
        if device is not None:
            _DEVICE = device
        elif list(encoder.parameters()):
            _DEVICE = next(encoder.parameters()).device
        _ENCODER = encoder.to(_DEVICE).eval()

    client = _get_client()
    return client.get_or_create_collection(
        name=IMAGE_COLLECTION_NAME,
        configuration={"hnsw": {"space": "cosine"}},
    )


def search_similar_image_ids(collection, image_tensor, cnt=5, exclude_ids=None):
    """用查询图的嵌入向量做近邻检索，返回最相似的 cnt 个图片 id（int 列表）。

    参数：
        collection: ChromaDB 集合
        image_tensor: (3, H, W) 或 (1, 3, H, W) 的查询图张量
        cnt: 返回数量
        exclude_ids: 需要排除的 id（例如查询图自身）
    """
    if _ENCODER is None:
        raise RuntimeError(
            "编码器尚未初始化：请先调用 create_embeddings() "
            "或 get_embedding_collection(encoder) 注入编码器"
        )

    total = collection.count()
    if total == 0:
        raise ValueError("向量集合为空，请先调用 create_embeddings() 建库")

    if image_tensor.dim() == 3:
        image_tensor = image_tensor.unsqueeze(0)
    image_tensor = image_tensor.to(_DEVICE, dtype=torch.float32)

    with torch.no_grad():
        query_vector = _normalize_embedding(_ENCODER(image_tensor).cpu().numpy())

    exclude_ids = {str(i) for i in (exclude_ids or [])}
    # 多取一些，过滤掉排除项后再截断
    n_results = min(cnt + len(exclude_ids), total)

    result = collection.query(
        query_embeddings=query_vector.tolist(),
        n_results=n_results,
        include=["distances"],
    )

    ids = [i for i in result["ids"][0] if i not in exclude_ids]
    return [int(i) for i in ids[:cnt]]
