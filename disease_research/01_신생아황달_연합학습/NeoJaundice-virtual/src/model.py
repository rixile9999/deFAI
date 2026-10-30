import torch
import torch.nn as nn
from torchvision import models


class MultiModalNet(nn.Module):
    def __init__(self):
        super(MultiModalNet, self).__init__()

        # 1. Image Branch (ResNet18)
        try:
            weights = models.ResNet18_Weights.DEFAULT
            self.cnn = models.resnet18(weights=weights)
        except:
            self.cnn = models.resnet18(pretrained=True)

        # ★ 핵심 1: CNN 파라미터 동결 (학습 X -> 파라미터 폭발 방지)
        for param in self.cnn.parameters():
            param.requires_grad = False

        self.cnn_dim = self.cnn.fc.in_features
        self.cnn.fc = nn.Identity()

        # 2. Metadata Branch
        self.meta_mlp = nn.Sequential(
            nn.Linear(4, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU()
        )
        self.meta_dim = 32

        # 3. Fusion
        self.fusion = nn.Sequential(
            nn.Linear(self.cnn_dim + self.meta_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        # 4. Heads
        self.head_reg = nn.Linear(128, 1)
        self.head_cls = nn.Linear(128, 1)

    def forward(self, img, meta):
        # ★ 핵심 2: CNN은 항상 eval 모드 (BatchNorm 통계 고정)
        self.cnn.eval()
        feat_img = self.cnn(img)

        feat_meta = self.meta_mlp(meta)
        combined = torch.cat((feat_img, feat_meta), dim=1)
        shared = self.fusion(combined)

        # ★ 핵심 3: Sigmoid로 출력값을 0~1로 강제 (무한대 발산 방지)
        pred_blood = torch.sigmoid(self.head_reg(shared))
        pred_treat = self.head_cls(shared)

        return pred_blood, pred_treat