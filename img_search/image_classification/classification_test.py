import torch
from torch.utils.data import DataLoader

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from common.utils import *
from classification_config import *
from classification_data import create_datasets
from classification_model import Classifier
from classification_engine import evaluate

import matplotlib.pyplot as plt

def predict(model, test_loader, device):
    # 1. 取一个批次的测试图像
    data_iter = iter(test_loader)
    images, labels = next(data_iter)

    # 2. 推理预测
    with torch.no_grad():
        inputs = images.to(device)
        # 前向传播
        outputs = model(inputs)

    # 3. 得到预测分类标签
    pred_labels = outputs.argmax(dim=-1).cpu().numpy()
    print(pred_labels.shape)

    # 4. 转换输入图片，为画图做准备
    images = images.permute(0, 2, 3, 1).numpy()
    print(images.shape)

    # 对比显示预测结果
    fig, axes = plt.subplots(1, 10, figsize=(25, 4), sharex=True, sharey=True)
    for i in range(10):
        axes[i].imshow(images[i])
        axes[i].axis('off')
        # 真实标签
        print(f"{i + 1}-label: {labels[i]}")
        # 预测标签
        pred_class = classification_names[pred_labels[i]]
        print(f"{i + 1}-pred: {pred_labels[i]}, 分类：{pred_class}")
        print()
    plt.show()

if __name__ == '__main__':
    seed_everything(SEED)

    # 1. 定义设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2. 创建数据集
    _, _, test_dataset = create_datasets()
    print("===========数据集创建完成===========")

    # 3. 创建加载器
    test_loader = DataLoader(test_dataset, batch_size=TEST_BATCH_SIZE)
    print("===========数据加载器创建完成===========")

    # 4. 加载模型
    model = Classifier().to(device)
    state_dict = torch.load(CLASSIFIER_MODEL_NAME)
    model.load_state_dict( state_dict )
    print("===========模型加载完成===========")

    # 5. 测试
    predict(model, test_loader, device)
    acc = evaluate(model, test_loader, device=device)

    print("测试集准确率：", acc)