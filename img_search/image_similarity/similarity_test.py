# 相似图检索测试：加载编码器 → 检索 Top-K → 计算同类别命中率并可视化
import argparse
import os
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torchvision.transforms as T
from PIL import Image

# 使 common 包可导入（脚本位于 img_search/image_similarity/）
sys.path.append(str(Path(__file__).resolve().parents[1]))

from similarity_config import *
from similarity_model import ConvEncoder
from similarity_embeddings import (
    create_embeddings,
    get_embedding_collection,
    search_similar_image_ids,
)
from common.utils import seed_everything, sorted_alphanum

# 标签文件与图库的公共位置
COMMON_DIR = Path(__file__).resolve().parents[1] / "common"
LABEL_PATH = COMMON_DIR / "fashion-labels.csv"


def parse_args():
    parser = argparse.ArgumentParser(description="相似图检索测试")
    parser.add_argument("--topk", type=int, default=5, help="每次检索返回的图片数量")
    parser.add_argument("--queries", type=int, default=20, help="评估时抽样的查询图数量")
    parser.add_argument("--build", type=int, default=0,
                        help="建库使用的图片数量；0 表示复用已有集合")
    parser.add_argument("--save", type=str, default="retrieval_result.png",
                        help="检索结果可视化保存路径")
    return parser.parse_args()


def load_encoder(device):
    """加载训练好的编码器权重。"""
    encoder = ConvEncoder()
    model_path = Path(__file__).resolve().parent / ENCODER_MODEL_NAME
    encoder.load_state_dict(torch.load(model_path, map_location=device))
    return encoder.to(device).eval()


def load_labels():
    """读取 id -> 类别 的映射，用于评估同类别命中率。"""
    labels = pd.read_csv(LABEL_PATH)
    return dict(zip(labels["id"], labels["target"]))


def evaluate_topk(collection, paths, label_map, transform, device, topk, num_queries):
    """随机抽取若干图库图片作为查询，统计 Top-K 中是否出现同类别图片。"""
    rng = random.Random(SEED)
    samples = rng.sample(paths, min(num_queries, len(paths)))

    hits = 0
    records = []
    for path in samples:
        query_id = int(Path(path).stem)
        image = transform(Image.open(path).convert("RGB")).to(device)

        similar_ids = search_similar_image_ids(
            collection, image, cnt=topk, exclude_ids=[query_id]
        )

        target = label_map.get(query_id)
        similar_targets = [label_map.get(i) for i in similar_ids]
        is_hit = target in similar_targets
        hits += is_hit
        records.append((path, similar_ids, is_hit))

    return hits / len(records), records


def visualize(records, transform, save_path, max_queries=5):
    """把「查询图 + Top-K 结果」画成网格图并保存。"""
    records = records[:max_queries]
    cols = len(records[0][1]) + 1
    rows = len(records)

    fig, axes = plt.subplots(rows, cols, figsize=(2.2 * cols, 2.4 * rows))
    axes = axes.reshape(rows, cols)

    for row, (path, similar_ids, is_hit) in enumerate(records):
        query = Image.open(path).convert("RGB")
        axes[row][0].imshow(query)
        axes[row][0].set_title("query", fontsize=9)
        axes[row][0].axis("off")

        for col, image_id in enumerate(similar_ids, start=1):
            image_path = os.path.join(IMG_PATH, f"{image_id}.jpg")
            if os.path.exists(image_path):
                axes[row][col].imshow(Image.open(image_path).convert("RGB"))
            axes[row][col].set_title(f"id={image_id}", fontsize=8)
            axes[row][col].axis("off")

        axes[row][0].set_ylabel("hit" if is_hit else "miss", fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close(fig)


def main():
    args = parse_args()
    seed_everything(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备：{device}")

    # 1. 加载编码器（检索阶段只需要编码器，解码器已不再使用）
    encoder = load_encoder(device)
    print(f"编码器加载完成：{ENCODER_MODEL_NAME}")

    # 2. 准备向量集合
    collection = get_embedding_collection(encoder, device=device)
    if args.build > 0:
        print(f"正在用前 {args.build} 张图建库……")
        collection = create_embeddings(encoder, limit=args.build, device=device)

    total = collection.count()
    if total == 0:
        print("向量集合为空。请先运行 similarity_train.py 训练，再用 --build N 建立索引，"
              "例如：python similarity_test.py --build 500")
        return

    print(f"向量集合规模：{total}")

    # 3. 检索评估
    paths = [
        os.path.join(IMG_PATH, name)
        for name in sorted_alphanum(os.listdir(IMG_PATH))
        if name.lower().endswith((".jpg", ".jpeg", ".png"))
    ]
    label_map = load_labels()
    transform = T.Compose([T.Resize((IMG_H, IMG_W)), T.ToTensor()])

    hit_rate, records = evaluate_topk(
        collection, paths, label_map, transform, device, args.topk, args.queries
    )
    print(f"同类别命中率@{args.topk}：{hit_rate:.4f}"
          f"（{sum(r[2] for r in records)}/{len(records)}）")

    # 4. 可视化前若干个查询结果
    visualize(records, transform, args.save)
    print(f"检索结果已保存：{Path(args.save).resolve()}")

    # 5. 单图演示：打印某个查询图的 Top-K id
    demo_path = paths[0]
    demo_id = int(Path(demo_path).stem)
    demo_tensor = transform(Image.open(demo_path).convert("RGB")).to(device)
    demo_ids = search_similar_image_ids(
        collection, demo_tensor, cnt=args.topk, exclude_ids=[demo_id]
    )
    print(f"示例查询 id={demo_id} → Top-{args.topk}: {demo_ids}")


if __name__ == "__main__":
    main()
