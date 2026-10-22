# 数据预处理
##项目配置信息
from pathlib import Path

# ------------ 数据路径与预处理配置 ------------
# 以配置文件所在位置为基准定位数据集，避免受运行时工作目录影响
IMG_PATH = Path(__file__).resolve().parents[1] / "common" / "dataset"  # 原始图像存储根目录
IMG_H = 64
IMG_W = 64

# 随机性相关配置
SEED = 42
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15
NOISE_FACTOR = 0.5

# 超参数
LEARNING_RATE = 1e-3
TRAIN_BATCH_SIZE = 32
VAL_BATCH_SIZE = 32
TEST_BATCH_SIZE = 32
EPOCHS = 30

# 项目配置
PACKAGE_NAME = "image_denoising"
DENOISER_MODEL_NAME = "denoiser.pt"