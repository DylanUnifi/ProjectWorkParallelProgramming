# models/cnn.py
# Version: 2.1 (feature extraction)

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=False):
        super(ResidualBlock, self).__init__()
        stride = 2 if downsample else 1
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = nn.Sequential()
        if downsample or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = self.downsample(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        return F.relu(out)


class CNNBinaryClassifier(nn.Module):
    def __init__(self, in_channels=1):
        super(CNNBinaryClassifier, self).__init__()
        self.layer1 = ResidualBlock(in_channels, 32)
        self.layer2 = ResidualBlock(32, 64, downsample=True)
        self.layer3 = ResidualBlock(64, 128, downsample=True)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(128, 1)
        self._feature_dim = 128

    def get_feature_dim(self):
        return self._feature_dim

    @torch.no_grad()
    def forward_features(self, x, use_dropout: bool = False):
        """
        It returns only the features (flattened after global pooling).
        By default, there is no dropout for deterministic features.
        """

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.global_pool(x)     # [B, 128, 1, 1]
        x = x.view(x.size(0), -1)   # [B, 128]
        if use_dropout and self.training:
            x = self.dropout(x)
        return x


    def forward(self, x, return_features: bool = False, apply_dropout: bool = True):

        """
        Classical forward (sigmoid probability) with the option to return features as well.
        """
        feats = self.forward_features(x, use_dropout=False)
        x = self.dropout(feats) if apply_dropout else feats
        logits = self.fc(x)
        probs = torch.sigmoid(logits)
        if return_features:
            return probs, feats
        return probs


    class CNNFeatureExtractor(nn.Module):

        """
        It directly exposes the features of the backbone, which is useful for SVM, kernels, etc.
        """
        def __init__(self, backbone: CNNBinaryClassifier, apply_dropout: bool = False):
            super().__init__()
            self.backbone = backbone
            self.apply_dropout = apply_dropout

        def get_feature_dim(self):
            return self.backbone.get_feature_dim()

        def forward(self, x):
            # Unless requested, deterministic features are enforced (no dropout).
            return self.backbone.forward_features(x, self.apply_dropout)