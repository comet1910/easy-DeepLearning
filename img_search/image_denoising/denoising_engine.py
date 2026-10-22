import torch
from tqdm import tqdm


# 训练一个轮次
def train_one_epoch(model, train_loader, loss_fn, optimizer, device):
    # 训练
    model.train()
    total_loss = 0
    for noise_img, image in train_loader:
        input = noise_img.to(device)
        target = image.to(device)
        # 前向传播
        output = model(input)
        # 计算损失
        loss = loss_fn(output, target)
        # 反向传播
        loss.backward()
        # 更新参数
        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss.item()
    # 本轮训练结束，返回平均损失
    return total_loss / len(train_loader)

# 验证
def val_step(model, val_loader, loss_fn, device):
    # 验证
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for noise_img, image in val_loader:
            input = noise_img.to(device)
            target = image.to(device)
            # 前向传播
            output = model(input)
            # 计算损失
            loss = loss_fn(output, target)

            total_loss += loss.item()
    # 本轮训练结束，返回平均损失
    return total_loss / len(val_loader)

# 测试
def test_step(model, test_loader, loss_fn, device):
    # 测试
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for noise_img, image in tqdm(test_loader, desc='测试'):
            input = noise_img.to(device)
            target = image.to(device)
            # 前向传播
            output = model(input)
            # 计算损失
            loss = loss_fn(output, target)

            total_loss += loss.item()
    # 本轮训练结束，返回平均损失
    return total_loss / len(test_loader)
