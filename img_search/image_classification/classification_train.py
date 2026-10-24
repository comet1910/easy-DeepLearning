import torch
from torch import nn, optim
from torch.utils.data import DataLoader

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from common.utils import *
from classification_config import *
from classification_data import create_datasets
from classification_model import Classifier
from classification_engine import train_one_epoch, validate

from tqdm import tqdm

if __name__ == '__main__':
    seed_everything(SEED)

    # 1. 定义设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2. 创建数据集
    train_dataset, val_dataset, _ = create_datasets()
    print("===========数据集创建完成===========")

    # 3. 创建加载器
    train_loader = DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=VAL_BATCH_SIZE)
    print("===========数据加载器创建完成===========")

    # 4. 创建模型
    model = Classifier().to(device)

    # 5. 损失函数
    loss_fn = nn.CrossEntropyLoss()

    # 6. 优化器
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    # 7. 训练核心流程
    print("===========训练开始===========")
    min_val_loss = float('inf')
    for epoch in tqdm(range(EPOCHS)):
        # 训练
        train_loss = train_one_epoch(model, train_loader, loss_fn=loss_fn, optimizer=optimizer, device=device)
        # 验证
        val_loss = validate(model, val_loader, loss_fn=loss_fn, device=device)

        print(f"Epoch: {epoch+1}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

        # 判断如果验证损失减小，就保存模型
        if val_loss < min_val_loss:
            print("验证集损失减小，保存模型...")
            min_val_loss = val_loss
            torch.save(model.state_dict(), CLASSIFIER_MODEL_NAME)
        else:
            print("验证集损失没有减小，不保存模型。")

    print("===========训练结束===========")

