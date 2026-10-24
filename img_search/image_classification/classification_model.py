import torch
import torch.nn as nn

class Classifier(nn.Module):
    # 初始化，传入分类数
    def __init__(self, n_classes=5):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Flatten(),
            nn.Linear(4096, n_classes)
        )

    # 前向传播
    def forward(self, x):
        return self.model(x)

if __name__ == '__main__':
    input = torch.rand(32, 3, 64, 64)
    model = Classifier()
    output = model(input)
    print(output.size())