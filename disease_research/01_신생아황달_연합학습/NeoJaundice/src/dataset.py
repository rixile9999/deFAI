import os
import cv2
import torch
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class ImagePreprocessor:
    @staticmethod
    def adaptive_skin_crop(img_bgr):
        """
        [핵심 기술] 스마트 적응형 크롭
        이미지 중앙(피부)의 색상 통계를 기반으로 유사한 영역을 찾아내어
        컬러 차트(테두리)를 자동으로 제거합니다.
        """
        if img_bgr is None: return np.zeros((224, 224, 3), dtype=np.uint8)

        h, w = img_bgr.shape[:2]
        if h < 50 or w < 50: return img_bgr

        # 1. 중앙 10% 영역 샘플링 (여기는 무조건 피부라고 가정)
        cy, cx = h // 2, w // 2
        sample_h, sample_w = int(h * 0.1), int(w * 0.1)
        roi_sample = img_bgr[cy - sample_h:cy + sample_h, cx - sample_w:cx + sample_w]

        # 2. HSV 변환 후 피부색 통계 추출 (RGB보다 빛 반사에 강함)
        hsv_roi = cv2.cvtColor(roi_sample, cv2.COLOR_BGR2HSV)
        mean_color = np.mean(hsv_roi, axis=(0, 1))

        # 3. 허용 범위 설정 (평균 색상 위아래로 여유를 둠)
        # Hue(색상) +-20, Sat(채도)/Val(명도) +-60
        lower = np.clip(mean_color - np.array([20, 60, 60]), 0, 255)
        upper = np.clip(mean_color + np.array([20, 60, 60]), 0, 255)
        # HSV Hue 범위(0~180) 보정
        lower[0] = max(0, lower[0])
        upper[0] = min(180, upper[0])

        # 4. 전체 이미지에서 마스크 생성 (피부색이면 흰색, 차트면 검은색)
        hsv_full = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv_full, lower.astype(np.uint8), upper.astype(np.uint8))

        # 5. 노이즈 제거 (모폴로지 연산)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)

        # 6. 가장 큰 덩어리(Contour) 찾기 -> 그게 바로 피부 영역
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            # 면적이 가장 큰 덩어리 선택
            largest_cnt = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_cnt)

            # 너무 작으면(노이즈) 무시하고, 적당한 크기면 크롭
            if area > (h * w * 0.05):
                x, y, w_box, h_box = cv2.boundingRect(largest_cnt)

                # 결과물 반환
                return img_bgr[y:y + h_box, x:x + w_box]

        # [Fallback] 탐지 실패 시 안전하게 중앙 30%만 크롭
        scale = 0.3
        crop_h, crop_w = int(h * scale), int(w * scale)
        return img_bgr[cy - crop_h // 2:cy + crop_h // 2, cx - crop_w // 2:cx + crop_w // 2]


class JaundiceDataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.df = df
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def _safe_float(self, val, divisor=1.0):
        try:
            if pd.isna(val) or str(val).strip() == '': return 0.0
            val_float = float(val)
            if np.isinf(val_float) or np.isnan(val_float): return 0.0
            return val_float / divisor
        except:
            return 0.0

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_name = str(row['image_idx']).strip()
        img_path = os.path.join(self.img_dir, img_name)

        image = cv2.imread(img_path)
        if image is None:
            image = np.zeros((224, 224, 3), dtype=np.uint8)
        else:
            try:
                # [★] 스마트 크롭 적용 (차트 제거)
                image = ImagePreprocessor.adaptive_skin_crop(image)

                # [중요] 화이트 밸런스 제거: 피부만 남은 상태에서 Gray World를 쓰면
                # 전체를 회색으로 만들려는 성질 때문에 오히려 황달 정보가 사라집니다.
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            except:
                image = cv2.resize(image, (224, 224))
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = Image.fromarray(image)
        if self.transform:
            image = self.transform(image)

        # [최종 방어] NaN 제거
        if torch.isnan(image).any():
            image = torch.nan_to_num(image, nan=0.0)

        # 메타데이터 및 타겟 처리
        gender = 1.0 if str(row.get('gender', 'M')).strip().upper() == 'F' else 0.0
        ga = self._safe_float(row.get('gestational_age'), 40.0)
        age = self._safe_float(row.get('age(day)'), 30.0)
        weight = self._safe_float(row.get('weight'), 4000.0)

        meta = torch.tensor([gender, ga, age, weight], dtype=torch.float32)
        if torch.isnan(meta).any(): meta = torch.nan_to_num(meta, nan=0.0)

        blood = self._safe_float(row.get('blood(mg/dL)'), 30.0)
        treat = self._safe_float(row.get('Treatment'), 1.0)

        return (image, meta), (torch.tensor([blood]), torch.tensor([treat]))


def get_transforms(mode='train'):
    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    if mode == 'train':
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            # [추가] "같은 색도 다르게 찍힐 수 있다"는 점을 보완하기 위해
            # 밝기, 대비, 채도를 무작위로 흔들어주어 모델이 조명 변화에 강해지게 함
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            normalize
        ])
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize
    ])