#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Single model: PerturbAwareNet  —— 只保留“有群”版本
- 第一层：SteerableFirstLayer（根据 z/s 对卷积核逐样本转向）
- 主干：纯 CNN 堆叠到 1024 维特征
- 头部：
    1) 分类头（n_classes）
    2) 扰动感知联合训练所需的两个头：
       - z_head：5 维（二分类 logits，顺序 ['CFO','SCALE','GAIN','SHIFT','CHIRP']）
       - s_head：5 维（对应扰动的实数参数回归）
注意：
- forward(x, z, s) 需要同时提供 z/s（训练/验证）。若你要在“测试时不提供标签”，
  请在 CSR 里先用 z_head/s_head 预测得到 ẑ/ŝ 再喂给本模型第一层。
"""

from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from gconv import SteerableFirstLayer  # 与你的 gconv 文件保持一致

PERTURB_ORDER = ['CFO', 'SCALE', 'GAIN', 'SHIFT', 'CHIRP']
N_PERT = len(PERTURB_ORDER)


# ------------------------------
# 基础模块：1×k 的时序卷积块
# ------------------------------
class Conv2dBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int, stride: int = 1, pool: bool = False):
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, out_ch, kernel_size=(1, k), stride=(1, stride),
                      padding=(0, k // 2), bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        if pool:
            layers.append(nn.AvgPool2d(kernel_size=(1, 2), stride=(1, 2)))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


def _make_backbone() -> nn.Sequential:
    """与之前的 P4 主干等价的纯 CNN 堆叠，输出通道 1024。"""
    layers = []
    layers += [Conv2dBlock(32,  64, 11, pool=True)]
    layers += [Conv2dBlock(64, 128,  9, pool=True)]
    layers += [Conv2dBlock(128, 256, 7,  pool=True)]
    layers += [Conv2dBlock(256, 256, 7,  pool=True)]
    layers += [Conv2dBlock(256, 512, 5,  pool=True)]
    layers += [Conv2dBlock(512, 512, 5,  pool=True)]
    layers += [Conv2dBlock(512, 1024, 3, pool=True)]
    return nn.Sequential(*layers)


# ------------------------------
# 单一模型：PerturbAwareNet
# ------------------------------
class PerturbAwareNet(nn.Module):
    """
    输入:
        x: [B, 2, 1, L]
        z: [B, 5]   0/1 扰动激活标签（训练/验证必须提供）
        s: [B, 5]   扰动参数（训练/验证必须提供；未激活项可 NaN/任意）
    输出:
        logits: [B, n_classes]
        feat:   [B, 1024]   (分类前的全局特征)
        z_logit:[B, 5]      (联合训练用的扰动分类 logits)
        s_pred: [B, 5]      (联合训练用的扰动回归预测)
    """
    def __init__(self, n_classes: int, fs: float = 50e6):
        super().__init__()
        # 第一层：纯群等变（需要 z/s）
        self.first = SteerableFirstLayer(in_ch=2, out_ch=32, k=5, fs=fs)

        # 主干 & 池化
        self.features = _make_backbone()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(1024, 512), nn.ReLU(inplace=True), nn.Dropout(0.5),
            nn.Linear(512, n_classes),
        )

        # 扰动感知头（联合训练）
        # 先压到一个较小的共享表征，再分出 z/s 两个分支
        self._perturb_reducer = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(1, 3), padding=(0, 1), bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.z_head = nn.Linear(64, N_PERT)  # 二分类 logits（可用 BCEWithLogitsLoss）
        self.s_head = nn.Linear(64, N_PERT)  # 实数回归（可用 SmoothL1Loss/MSELoss）

    # —— 为了不改 CSR，保留这三个空接口（不做任何事）——
    def set_steer_disabled(self, flag: bool):  # 兼容旧调用；现在始终启用群
        return
    def set_first_debug(self, flag: bool):
        return
    def get_first_last_debug(self):
        return {}

    def forward(self, x: torch.Tensor, z: torch.Tensor, s: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # 1) 扰动感知辅助头：只看输入 x，提取第一层输入的全局表征
        #    注意：这是用来做联合训练监督的“感知器”，不参与卷积核转向。
        with torch.no_grad():
            pass  # 显式说明不需要这里对 x 额外处理；直接从 first 的输入侧取特征
        # 直接用一个浅层卷积对 x 提取共享表征（不依赖 z/s）
        red = self._perturb_reducer(self._x_as_32ch(x))  # [B,64,1,1]
        red = red.view(red.size(0), -1)                  # [B,64]
        z_logit = self.z_head(red)                       # [B,5]
        s_pred  = self.s_head(red)                       # [B,5]

        # 2) 核转向 + 主干分类
        x = self.first(x, z, s)          # 需要 z/s；测试若无标签，请先用 z_logit/s_pred 生成 ẑ/ŝ
        x = self.features(x)
        x = self.pool(x).view(x.size(0), -1)  # [B,1024]
        feat = x
        logits = self.classifier(x)
        return logits, feat, z_logit, s_pred

    @staticmethod
    def _x_as_32ch(x: torch.Tensor) -> torch.Tensor:
        """
        将 [B,2,1,L] 通过一个固定映射升到 32 通道，供扰动感知头使用。
        这里用 1×1 卷积快速升维，不影响第一层的群等变逻辑。
        """
        if not hasattr(PerturbAwareNet, "_lift32"):
            # 惰性创建：共享一个升维层（不参与第一层转向）
            PerturbAwareNet._lift32 = nn.Sequential(
                nn.Conv2d(2, 32, kernel_size=(1, 1), bias=False),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
            ).to(x.device)
        return PerturbAwareNet._lift32(x)


# ------------------------------
# 工厂：只保留“perturbawarenet”一个名字
# ------------------------------
def create(name: str, num_classes: int, fs: float = 50e6, **kwargs) -> PerturbAwareNet:
    key = (name or "").strip().lower()
    if key != "perturbawarenet":
        raise KeyError(f"Only 'perturbawarenet' is supported, got: {name}")
    return PerturbAwareNet(n_classes=num_classes, fs=fs)
