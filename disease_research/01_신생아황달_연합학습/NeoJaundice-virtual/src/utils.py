import os
import pandas as pd
import numpy as np
import cv2
from sklearn.model_selection import train_test_split


def generate_dummy_data(data_dir="./data", num_patients=50):
    """실행 테스트를 위한 가상 데이터 생성 (노이즈 추가 버전)"""
    img_dir = os.path.join(data_dir, "images")
    os.makedirs(img_dir, exist_ok=True)

    data = []
    print(f"Generating Noisy dummy data in {data_dir}...")

    for i in range(num_patients):
        pid = f"{i:04d}"
        gender = np.random.choice(['M', 'F'])
        g_age = np.random.randint(36, 42)
        age = np.random.uniform(1, 14)
        weight = np.random.randint(2500, 4500)

        # 빌리루빈 수치 (2.0 ~ 20.0)
        blood = np.random.uniform(2.0, 20.0)
        treatment = 1 if blood > 13.0 else 0

        for j in range(1, 4):
            img_name = f"{pid}-{j}.jpg"
            img_path = os.path.join(img_dir, img_name)

            # [핵심] 랜덤 노이즈가 섞인 피부색 이미지 생성
            base_color = np.array([200, 200, 255])  # BGR
            jaundice_factor = blood / 20.0
            skin_color = base_color.copy()
            skin_color[0] -= int(100 * jaundice_factor)

            # 224x224 빈 캔버스
            img = np.zeros((224, 224, 3), dtype=np.uint8)

            # 1. 배경색 채우기
            img[:] = skin_color

            # 2. ★ 노이즈 추가 (분산 0 방지 -> NaN 해결의 열쇠)
            noise = np.random.randint(-20, 20, (224, 224, 3), dtype=np.int16)
            img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

            # 3. 컬러 차트 테두리
            cv2.rectangle(img, (0, 0), (20, 20), (0, 0, 255), -1)
            cv2.rectangle(img, (20, 0), (40, 20), (0, 255, 0), -1)

            cv2.imwrite(img_path, img)
            data.append([pid, img_name, gender, g_age, round(age, 1), weight, round(blood, 1), treatment])

    df = pd.DataFrame(data, columns=[
        'patient_id', 'image_idx', 'gender', 'gestational_age', 'age(day)', 'weight', 'blood(mg/dL)', 'Treatment'
    ])
    df.to_csv(os.path.join(data_dir, "metadata.csv"), index=False)
    return df


def load_partition(client_id, num_clients, csv_path):
    df = pd.read_csv(csv_path)
    df['patient_id'] = df['patient_id'].astype(str)
    unique_pids = df['patient_id'].unique()
    np.random.seed(42)
    np.random.shuffle(unique_pids)

    pid_chunks = np.array_split(unique_pids, num_clients)
    my_pids = pid_chunks[int(client_id)]

    client_df = df[df['patient_id'].isin(my_pids)].reset_index(drop=True)
    train_df, val_df = train_test_split(client_df, test_size=0.2, random_state=42)
    return train_df, val_df