import torch
import torch.nn as nn
from torchvision import models


class MultiModalNet(nn.Module):
    def __init__(self, pretrained=True):
        super(MultiModalNet, self).__init__()

        # ResNet18 로드
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        self.cnn = models.resnet18(weights=weights)

        # ★ 수정: 파라미터 동결 해제 (실제 데이터 학습을 위해)
        # for param in self.cnn.parameters():
        #     param.requires_grad = False (주석 처리 또는 삭제)

        self.cnn_dim = self.cnn.fc.in_features
        self.cnn.fc = nn.Identity()

        # 메타데이터 처리
        self.meta_mlp = nn.Sequential(
            nn.Linear(4, 16),
            nn.LayerNorm(16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU()
        )
        self.meta_dim = 32

        # 통합 레이어
        self.fusion = nn.Sequential(
            nn.Linear(self.cnn_dim + self.meta_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        # 출력 헤드
        self.head_reg = nn.Linear(128, 1)
        self.head_cls = nn.Linear(128, 1)

    def train(self, mode=True):
        super().train(mode)
        # Use common pretrained statistics instead of mixing incompatible local BN moments.
        # LayerNorm above also permits the final singleton batch without dropping patients.
        for module in self.cnn.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()
        return self

    def forward(self, img, meta):
        feat_img = self.cnn(img)
        feat_meta = self.meta_mlp(meta)

        combined = torch.cat((feat_img, feat_meta), dim=1)
        shared = self.fusion(combined)

        # ★ 안전장치: Sigmoid로 출력을 0~1로 강제 (Dataset 스케일과 일치)
        # 값이 무한대로 튀는 것을 막아줌
        pred_blood = torch.sigmoid(self.head_reg(shared))
        pred_treat = self.head_cls(shared)

        return pred_blood, pred_treat
