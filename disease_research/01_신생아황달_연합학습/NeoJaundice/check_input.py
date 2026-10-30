import cv2
import os
import numpy as np


def debug_skin_crop(img_path, output_dir):
    img = cv2.imread(img_path)
    if img is None: return

    # 1. 중앙 샘플링 로직 시각화
    h, w = img.shape[:2]
    cy, cx = h // 2, w // 2
    sample_h, sample_w = int(h * 0.1), int(w * 0.1)

    # 2. 적응형 마스크 생성
    roi_sample = img[cy - sample_h:cy + sample_h, cx - sample_w:cx + sample_w]
    hsv_roi = cv2.cvtColor(roi_sample, cv2.COLOR_BGR2HSV)
    mean_c = np.mean(hsv_roi, axis=(0, 1))

    lower = np.clip(mean_c - [20, 60, 60], 0, 255)
    upper = np.clip(mean_c + [20, 60, 60], 0, 255)
    lower[0] = max(0, lower[0])
    upper[0] = min(180, upper[0])

    hsv_full = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_full, lower.astype(np.uint8), upper.astype(np.uint8))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    debug_img = img.copy()
    if contours:
        largest = max(contours, key=cv2.contourArea)
        x, y, w_box, h_box = cv2.boundingRect(largest)
        # 빨간색 네모: AI가 찾은 피부 영역
        cv2.rectangle(debug_img, (x, y), (x + w_box, y + h_box), (0, 0, 255), 3)

    filename = os.path.basename(img_path)
    cv2.imwrite(os.path.join(output_dir, f"check_{filename}"), debug_img)


if __name__ == "__main__":
    # debug_data의 이미지 10장을 테스트
    input_dir = "./debug_data/client_1/images"
    output_dir = "./check_visuals_smart"
    os.makedirs(output_dir, exist_ok=True)

    if os.path.exists(input_dir):
        print(f"🔍 {input_dir} 에서 이미지 검증 중...")
        files = [f for f in os.listdir(input_dir) if f.endswith(".jpg")]
        for f in files[:20]:
            debug_skin_crop(os.path.join(input_dir, f), output_dir)
        print(f"✅ {output_dir} 폴더를 확인해보세요. 빨간 네모가 피부만 잘 감싸고 있나요?")
    else:
        print("❌ debug_data 폴더가 없습니다. run_federated.py를 먼저 실행하세요.")