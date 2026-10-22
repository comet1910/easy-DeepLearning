import torch
from torch.utils.data import DataLoader

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from common.utils import *
from denoising_config import *
from denoising_data import create_datasets
from denoising_model import ConvDenoiser
from denoising_engine import test_step

import matplotlib.pyplot as plt

def test_new_data(model, test_loader, device):
    # 1. 取一个批次的测试图像
    data_iter = iter(test_loader)
    noise_imgs, images = next(data_iter)

    # 2. 推理预测
    with torch.no_grad():
        inputs = noise_imgs.to(device)
        # 前向传播
        outputs = model(inputs)

    # 3. 转成ndarray，方便画图
    images_numpy = images.permute(0, 2, 3, 1).numpy()
    noise_imgs_numpy = noise_imgs.permute(0, 2, 3, 1).numpy()
    denoised_imgs_numpy = outputs.permute(0, 2, 3, 1).cpu().numpy()

    # 画图
    fig, axes = plt.subplots(3, 10, figsize=(25, 4), sharex=True, sharey=True)

    for ax_row, imgs in zip(axes, [images_numpy, noise_imgs_numpy, denoised_imgs_numpy]):
        for ax, img in zip(ax_row, imgs):
            ax.imshow(img)
            ax.set_axis_off()
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
    model = ConvDenoiser().to(device)
    state_dict = torch.load(DENOISER_MODEL_NAME)
    model.load_state_dict( state_dict )
    print("===========模型加载完成===========")

    # 5. 测试
    test_new_data(model, test_loader, device)
    test_loss = test_step(model, test_loader, loss_fn=torch.nn.MSELoss(), device=device)

    print("测试集误差：", test_loss)