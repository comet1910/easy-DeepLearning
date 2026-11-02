import torch
import torch.nn as nn

# 分别定义编码器类和解码器类
class ConvEncoder(nn.Module):
    def __init__(self):
        super(ConvEncoder, self).__init__()
        # 卷积层
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        # 通用池化层
        self.pool = nn.MaxPool2d(2, 2)
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = torch.relu(self.conv3(x))
        x = self.pool(x)
        x = torch.relu(self.conv4(x))
        x = self.pool(x)
        x = torch.relu(self.conv5(x))
        x = self.pool(x)
        x = torch.relu(self.conv6(x))
        x = self.pool(x)
        # 压缩成向量形式（N, 512）返回
        x = x.squeeze(-1).squeeze(-1)
        return x

class ConvDecoder(nn.Module):
    def __init__(self):
        super(ConvDecoder, self).__init__()
        self.conv_t1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv_t2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv_t3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv_t4 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.conv_t5 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.conv_t6 = nn.ConvTranspose2d(16, 3, kernel_size=2, stride=2)
    def forward(self, x):
        # 恢复4维张量形状（N, 512, 1, 1）
        x = x.unsqueeze(-1).unsqueeze(-1)
        # 转置卷积前向传播
        x = torch.relu(self.conv_t1(x))
        x = torch.relu(self.conv_t2(x))
        x = torch.relu(self.conv_t3(x))
        x = torch.relu(self.conv_t4(x))
        x = torch.relu(self.conv_t5(x))
        x = torch.sigmoid(self.conv_t6(x))
        return x

if __name__ == '__main__':
    input = torch.randn(10, 3, 64, 64)
    encoder = ConvEncoder()
    decoder = ConvDecoder()
    # 前向传播
    embeddings = encoder(input)
    print("嵌入向量形状：", embeddings.shape)
    output = decoder(embeddings)
    print("重构图像形状：", output.shape)