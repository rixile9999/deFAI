# 🚀 데모 환경 설정 (한 번만 실행)

## 📦 필수 패키지 설치

### 1. pywin32 설치 (자동 창 배치용)

```bash
# Anaconda 환경 활성화
conda activate flwr

# pywin32 설치
pip install pywin32
```

**설치 확인**:
```bash
python -c "import win32gui; print('✅ pywin32 installed successfully!')"
```

---

## 🧪 데모 테스트

### 1. 데이터 생성 (최초 1회)

```bash
python data_gen_hierarchical.py
```

**출력**:
```
✅ Created: data/districts/seoul_gangnam.xlsx
✅ Created: data/districts/seoul_seocho.xlsx
✅ Created: data/districts/gyeonggi_suwon.xlsx
✅ Created: data/districts/gyeonggi_seongnam.xlsx
```

### 2. 데이터 검증

```bash
python verify_hierarchical_data.py
```

**출력**:
```
✅ All 4 district files exist
✅ All data is valid
```

### 3. 키 생성

```bash
python setup_hierarchical.py
```

**출력**:
```
✅ Public key saved: keys/public_key.pkl
✅ Secret key saved: keys/secret_key.pkl
```

### 4. 시각화 영상 생성

```bash
python demo1_hierarchical_visualization.py
```

**출력**:
```
✅ Hierarchical Visualization Video Generated!
   File: demo1_hierarchical.mp4
   Duration: 10 seconds (300 frames @ 30fps)
```

---

## ✅ 환경 체크리스트

데모 실행 전에 다음을 확인하세요:

- [ ] pywin32 설치됨
- [ ] 4개 지역 데이터 파일 존재 (`data/districts/`)
- [ ] 암호화 키 생성됨 (`keys/`)
- [ ] 시각화 영상 존재 (`demo1_hierarchical.mp4`)
- [ ] 바탕화면 정리 (녹화용)
- [ ] OBS Studio 설치 (녹화 소프트웨어)

---

## 🎬 데모 실행

모든 설정이 완료되면:

```bash
# 방법 1: 배치 파일
launch_demo.bat

# 방법 2: Python 직접 실행
python demo_launcher.py
```

**모드 선택**:
- `1`: 단계별 실행 (설명 가능)
- `2`: 자동 실행 (녹화용) ⭐ **권장**
- `3`: 시각화만 재생

---

## 🔧 문제 해결

### pywin32 설치 실패

**증상**: `pip install pywin32` 오류

**해결**:
```bash
# Anaconda 전용 설치
conda install pywin32
```

또는:

```bash
# 관리자 권한으로 CMD 실행 후
pip install --user pywin32
python -m pywin32_postinstall -install
```

### 데이터 파일 없음

**증상**: `FileNotFoundError: seoul_gangnam.xlsx`

**해결**:
```bash
python data_gen_hierarchical.py
```

### 시각화 영상 없음

**증상**: `demo1_hierarchical.mp4 not found`

**해결**:
```bash
python demo1_hierarchical_visualization.py
```

---

## 📊 완성된 파일 구조

```
fcausal_hierarchical/
├── 📁 data/
│   └── districts/
│       ├── seoul_gangnam.xlsx      ✅
│       ├── seoul_seocho.xlsx       ✅
│       ├── gyeonggi_suwon.xlsx     ✅
│       └── gyeonggi_seongnam.xlsx  ✅
│
├── 📁 keys/
│   ├── public_key.pkl              ✅
│   └── secret_key.pkl              ✅
│
├── 🎬 demo1_hierarchical.mp4       ✅ (10초 시각화)
├── 🐍 demo_launcher.py             ✅ (통합 런처)
├── 📄 launch_demo.bat              ✅ (배치 파일)
│
├── 📖 README_DEMO.md               ✅ (빠른 가이드)
├── 📖 DEMO_RECORDING_GUIDE.md      ✅ (상세 가이드)
└── 📖 SETUP_DEMO.md                ✅ (이 파일)
```

---

## 🎯 다음 단계

1. ✅ 환경 설정 완료
2. 📹 `DEMO_RECORDING_GUIDE.md` 읽기
3. 🎬 데모 녹화 시작!

---

**작성**: 2026-01-01
**상태**: ✅ Setup Complete
