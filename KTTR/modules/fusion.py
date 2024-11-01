import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.util import Hourglass, make_coordinate_grid, AntiAliasInterpolation2d
from .GDN import GDN
import math
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class FusionNetwork(nn.Module):
    def __init__(self, kp_detector, in_channels, num_blocks=3):
        super(FusionNetwork, self).__init__()
        self.kp_detector = kp_detector

        self.res_blocks = nn.Sequential(
            *[ResidualBlock(in_channels, in_channels) for _ in range(num_blocks)]
        )

    def forward(self, frames, weights):
        # 提取每个帧的多尺度特征
        features_list = [self.kp_detector(frame, scalefactor=1)[1] for frame in frames]

        # 将权重堆叠在一起并进行归一化
        weights = F.softmax(weights, dim=1)  # 假设权重已经传递过来

        # 融合特征
        fused_features = {}
        for scale in features_list[0].keys():
            weighted_sum = sum(weight * features[scale] for weight, features in zip(weights[0], features_list))
            fused_features[scale] = self.res_blocks(weighted_sum)

        return fused_features

# 示例用法
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 假设已经定义了KPDetector
kp_detector = KPDetector(
    block_expansion=64, num_kp=10, num_channels=3, num_ref=5, num_temporal=5,
    scale_range=[0.5, 1.0], max_features=256, num_blocks=3, temperature=0.1
).to(device)

# 定义融合网络
fusion_network = FusionNetwork(kp_detector=kp_detector, in_channels=10).to(device)

# 假设有动态数量的参考帧
num_frames = 4  # 动态数量，可以变化
frames = [torch.randn(1, 3, 256, 256).to(device) for _ in range(num_frames)]
weights = torch.randn(1, num_frames).to(device)  # 预先计算的权重

# 融合帧
fused_frame = fusion_network(frames, weights)
for scale, feature in fused_frame.items():
    print(f"{scale}: {feature.shape}")  # 输出各个尺度特征的形状





