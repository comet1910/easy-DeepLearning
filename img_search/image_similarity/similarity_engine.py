import torch
from tqdm import tqdm


# 训练一个轮次
def train_one_epoch(encoder, decoder, train_loader, loss_fn, optimizer, device):
    # 训练
    encoder.train()
    decoder.train()
    total_loss = 0
    total_num = 0
    for input, target in train_loader:
        input = input.to(device)
        target = target.to(device)
        # 前向传播
        embedding = encoder(input)
        output = decoder(embedding)
        # 计算损失
        loss = loss_fn(output, target)
        # 反向传播
        loss.backward()
        # 更新参数
        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss.item() * input.shape[0]
        total_num += input.shape[0]
    # 本轮训练结束，返回平均损失
    return total_loss / total_num

# 验证
def validate(encoder, decoder, val_loader, loss_fn, device):
    # 验证
    encoder.eval()
    decoder.eval()
    total_loss = 0
    total_num = 0
    with torch.no_grad():
        for input, target in val_loader:
            input = input.to(device)
            target = target.to(device)
            # 前向传播
            embedding = encoder(input)
            output = decoder(embedding)
            # 计算损失
            loss = loss_fn(output, target)

            total_loss += loss.item() * input.shape[0]
            total_num += input.shape[0]
    # 返回平均损失
    return total_loss / total_num

# 评估
def evaluate(encoder, decoder, test_loader, loss_fn, device):
    # 验证
    encoder.eval()
    decoder.eval()
    total_loss = 0
    total_num = 0
    with torch.no_grad():
        for input, target in test_loader:
            input = input.to(device)
            target = target.to(device)
            # 前向传播
            embedding = encoder(input)
            output = decoder(embedding)
            # 计算损失
            loss = loss_fn(output, target)

            total_loss += loss.item() * input.shape[0]
            total_num += input.shape[0]
    # 返回平均损失
    return total_loss / total_num
