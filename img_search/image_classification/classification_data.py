import os

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, random_split
import torchvision.transforms as T

from common.utils import sorted_alphanum
from classification_config import *


class ImageLabelDataset(Dataset):
    # 初始化
    def __init__(self, image_dir, label_path, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.image_names = sorted_alphanum(os.listdir(image_dir))
        label_data = pd.read_csv(label_path)
        self.labels = label_data['target'].tolist()

    # 获取数据集长度
    def __len__(self):
        return len(self.image_names)

    # 根据索引号获取元素：(input, target) = (image, label)
    def __getitem__(self, idx):
        # 1. 构建图片的完整访问路径
        image_path = os.path.join(self.image_dir, self.image_names[idx])
        # 2. 打开图片
        image = Image.open(image_path).convert('RGB')
        # 3. 应用转换操作，得到Tensor
        if self.transform is not None:
            img_tensor = self.transform(image)
        else:
            raise ValueError("transform 参数不能为 None！")
        # 4. 找到图片对应的分类标签
        img_label = self.labels[idx]

        return img_tensor, img_label

# 创建数据集并划分
def create_datasets():
    # 定义图像转换操作（调整大小，转为Tensor）
    transform = T.Compose([
        T.Resize((IMG_H, IMG_W)),
        T.ToTensor()
    ])
    # 创建数据集
    dataset = ImageLabelDataset( IMG_PATH, LABELS_PATH,transform=transform )
    # 划分数据集
    train_dataset, val_dataset, test_dataset = random_split(dataset, [TRAIN_RATIO, VAL_RATIO, TEST_RATIO])

    return train_dataset, val_dataset, test_dataset

if __name__ == '__main__':
    # dataset = NoiseImageDataset(IMG_PATH)
    # print(dataset.image_names)
    train_dataset, val_dataset, test_dataset = create_datasets()
    print(f'train_dataset: {len(train_dataset)}, val_dataset: {len(val_dataset)}, test_dataset: {len(test_dataset)}')