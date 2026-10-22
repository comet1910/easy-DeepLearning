import torch
from torch import nn


# 自定义神经网络类：基于CNN的去噪器
class ConvDenoiser(nn.Module):
    def __init__(self):
        super(ConvDenoiser, self).__init__()
        # 编码器
        # 卷积层
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 16, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(16, 8, kernel_size=3, padding=1)
        # 通用池化层
        self.pool = nn.MaxPool2d(2, 2)
        # 解码器
        # 转置卷积层
        self.conv_t1 = nn.ConvTranspose2d(8, 8, kernel_size=2, stride=2)
        self.conv_t2 = nn.ConvTranspose2d(8, 16, kernel_size=2, stride=2)
        self.conv_t3 = nn.ConvTranspose2d(16, 32, kernel_size=2, stride=2)
        # 输出普通卷积层
        self.conv_out = nn.Conv2d(32, 3, kernel_size=3, padding=1)

    # 前向传播
    def forward(self, x):
        # 编码
        x = torch.relu(self.conv1(x))
        # print(f"conv1: {x.shape}")
        x = self.pool(x)
        # print(f"pool1: {x.shape}")

        x = torch.relu(self.conv2(x))
        # print(f"conv2: {x.shape}")
        x = self.pool(x)
        # print(f"pool2: {x.shape}")

        x = torch.relu(self.conv3(x))
        # print(f"conv3: {x.shape}")
        x = self.pool(x)
        # print(f"pool3: {x.shape}")

        # 解码
        x = torch.relu(self.conv_t1(x))
        # print(f"conv_t1: {x.shape}")

        x = torch.relu(self.conv_t2(x))
        # print(f"conv_t2: {x.shape}")

        x = torch.relu(self.conv_t3(x))
        # print(f"conv_t3: {x.shape}")

        y = torch.sigmoid(self.conv_out(x))
        return y

if __name__ == '__main__':
    input = torch.randn(12, 3, 64, 64)
    model = ConvDenoiser()
    output = model(input)
    print(output.shape)