import os
import cv2
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class ImagePreprocessor:
    @staticmethod
    def white_balance(img_bgr):
        """Gray World Assumption 기반 화이트 밸런스"""
        result = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype("float32")
        avg_a = np.mean(result[:, :, 1])
        avg_b = np.mean(result[:, :, 2])

        # a, b 채널을 128(무채색) 중심으로 이동
        result[:, :, 1] -= ((avg_a - 128) * (result[:, :, 0] / 255.0) * 1.1)
        result[:, :, 2] -= ((avg_b - 128) * (result[:, :, 0] / 255.0) * 1.1)

        result = np.clip(result, 0, 255).astype("uint8")
        return cv2.cvtColor(result, cv2.COLOR_LAB2RGB)

    @staticmethod
    def extract_skin_roi(img_bgr):
        """중앙 영역 추출 (컬러 차트 제외)"""
        h, w, _ = img_bgr.shape
        # 중앙 50% 영역만 사용
        crop_h, crop_w = int(h * 0.5), int(w * 0.5)
        cy, cx = h // 2, w // 2

        y1 = cy - crop_h // 2
        x1 = cx - crop_w // 2
        return img_bgr[y1:y1 + crop_h, x1:x1 + crop_w]


class JaundiceDataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.df = df
        self.img_dir = img_dir
        self.transform = transform

        # 메타데이터 정규화용 상수 (임의 설정, 실제론 전체 통계 사용 권장)
        self.stats = {'ga': 40.0, 'age': 30.0, 'weight': 4000.0}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # 1. 이미지 로드 및 전처리
        img_path = os.path.join(self.img_dir, row['image_idx'])
        image = cv2.imread(img_path)

        if image is None:
            image = np.zeros((224, 224, 3), dtype=np.uint8)
        else:
            # ROI 추출 -> 화이트 밸런스 -> RGB 변환
            image = ImagePreprocessor.extract_skin_roi(image)
            image = ImagePreprocessor.white_balance(image)

        image = Image.fromarray(image)
        if self.transform:
            image = self.transform(image)

        # 2. 메타데이터 (Feature Engineering)
        gender = 1.0 if str(row['gender']).strip().upper() == 'F' else 0.0
        ga = float(row['gestational_age']) / self.stats['ga']
        age = float(row['age(day)']) / self.stats['age']
        weight = float(row['weight']) / self.stats['weight']

        meta = torch.tensor([gender, ga, age, weight], dtype=torch.float32)

        # 3. 타겟 (Regression, Classification)
        # [수정 후] ★ 핵심! 30.0으로 나누어 0~1 사이로 스케일링
        raw_blood = float(row['blood(mg/dL)'])
        y_blood = torch.tensor([raw_blood / 30.0], dtype=torch.float32)
        y_treat = torch.tensor([float(row['Treatment'])], dtype=torch.float32)

        return (image, meta), (y_blood, y_treat)


def get_transforms(mode='train'):
    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    if mode == 'train':
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            normalize
        ])
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize
    ])