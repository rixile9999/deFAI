# 🎬 CES 2026 데모 빠른 실행 가이드

## ⚡ 빠른 시작 (1분 내)

### 방법 1: 배치 파일 (가장 간단)

1. **더블클릭**: `launch_demo.bat`
2. **선택**:
   - `2` 입력 → Enter (자동 실행)
3. **완료**: 30초 후 모든 창 실행됨

### 방법 2: Python 실행

```bash
python demo_launcher.py
```

---

## 📺 화면 배치 (3840x2160)

```
┌─────────────┬─────────────┬─────────────┐
│   (여백)    │   Tier 0    │ Seongnam    │
├─────────────┼─────────────┼─────────────┤
│   Seoul     │   (여백)    │  Gyeonggi   │
├─────────────┼─────────────┼─────────────┤
│  Gangnam    │   Seocho    │   Suwon     │
└─────────────┴─────────────┴─────────────┘
```

**7개 터미널 창 + 1개 시각화 영상**

---

## 🎯 데모 모드

### 1️⃣ Sequential (단계별)
- 각 단계마다 Enter 입력
- 설명 나레이션 가능
- 교육/발표용

### 2️⃣ Auto (자동)
- 5초 후 자동 실행
- 녹화 즉시 시작
- 전시/데모용 ⭐ **권장**

### 3️⃣ Quick Test
- 시각화 영상만 재생
- 빠른 확인용

---

## 📋 사전 준비

### 필수 파일 확인
```bash
python verify_hierarchical_data.py
```

✅ 4개 지역 데이터 존재 확인

### 시각화 영상 생성
```bash
python demo1_hierarchical_visualization.py
```

✅ `demo1_hierarchical.mp4` 생성

---

## 🎥 녹화 절차

1. **OBS Studio** 실행 (Display Capture)
2. **런처 실행**: `launch_demo.bat`
3. **모드 선택**: `2` (Auto)
4. **녹화 시작**: 5초 카운트다운 중
5. **대기**: 30초 (연합학습 완료)
6. **확인**: 시각화 영상 자동 재생
7. **종료**: Ctrl+C → OBS 녹화 중지

**총 소요 시간**: ~1분

---

## 📁 생성되는 파일

- `demo1_hierarchical.mp4`: 시각화 영상 (10초)
- `logs/`: 연합학습 로그 (자동)
- OBS 녹화본 (사용자 지정 경로)

---

## 🛠️ 문제 해결

### 창이 겹쳐요
```bash
pip install pywin32
```

### Tier 0가 대기 중이에요
- Seoul & Gyeonggi 모두 실행되었는지 확인
- 각 Tier 1은 2개 district 연결 필요

### 시각화 영상이 없어요
```bash
python demo1_hierarchical_visualization.py
```

---

## 📞 상세 가이드

자세한 내용은 `DEMO_RECORDING_GUIDE.md` 참조

---

**작성**: 2026-01-01
**버전**: v1.0
**상태**: ✅ Ready for CES 2026
