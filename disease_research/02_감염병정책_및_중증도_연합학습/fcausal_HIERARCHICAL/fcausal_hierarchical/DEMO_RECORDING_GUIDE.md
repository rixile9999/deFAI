# 🎬 CES 2026 데모 녹화 가이드

## 📋 개요

이 가이드는 CES 2026에서 발표할 계층형 연합학습 데모의 화면 녹화 절차를 설명합니다.

---

## 🖥️ 시스템 요구사항

### 하드웨어
- **주모니터**: 3840x2160 (4K UHD)
- **보조모니터**: 선택사항
- **메모리**: 최소 8GB RAM
- **CPU**: 멀티코어 권장 (7개 프로세스 동시 실행)

### 소프트웨어
- Windows 10/11
- Python 3.x with `flwr` environment
- Screen recording software (OBS Studio 권장)
- 선택사항: `pywin32` (자동 창 배치용)

```bash
pip install pywin32
```

---

## 📁 파일 구조

```
fcausal_hierarchical/
├── demo_launcher.py          ← 통합 런처 (메인)
├── launch_demo.bat            ← 배치 파일 (더블클릭 실행)
├── demo1_hierarchical.mp4     ← 시각화 영상
├── run_tier0.py               ← Tier 0 (KDCA)
├── run_tier1.py               ← Tier 1 (Provinces)
├── run_tier2.py               ← Tier 2 (Districts)
└── data/
    └── districts/
        ├── seoul_gangnam.xlsx
        ├── seoul_seocho.xlsx
        ├── gyeonggi_suwon.xlsx
        └── gyeonggi_seongnam.xlsx
```

---

## 🚀 빠른 시작 (Quick Start)

### 방법 1: 배치 파일 실행 (가장 간단)

1. `launch_demo.bat` 더블클릭
2. 메뉴에서 모드 선택:
   - `1`: 단계별 실행 (각 단계마다 Enter 입력)
   - `2`: 자동 실행 (녹화 즉시 시작)
   - `3`: 시각화 영상만 재생
3. 녹화 시작!

### 방법 2: Python 직접 실행

```bash
python demo_launcher.py
```

---

## 📺 화면 레이아웃 (3840x2160)

### 3x3 격자 배치 (각 셀: 1280x720)

```
┌─────────────────┬─────────────────┬─────────────────┐
│                 │   Tier 0 KDCA   │ Tier 2 Seongnam │
│     (Empty)     │    (Central)    │  (Gyeonggi)     │
│   [여백]        │   [중앙정부]     │   [성남시]       │
├─────────────────┼─────────────────┼─────────────────┤
│  Tier 1 Seoul   │                 │ Tier 1 Gyeonggi │
│   (Treatment)   │    (Empty)      │    (Control)    │
│   [서울시]      │   [여백]        │   [경기도]       │
├─────────────────┼─────────────────┼─────────────────┤
│ Tier 2 Gangnam  │ Tier 2 Seocho   │  Tier 2 Suwon   │
│    (Seoul)      │    (Seoul)      │   (Gyeonggi)    │
│   [강남구]      │   [서초구]      │   [수원시]       │
└─────────────────┴─────────────────┴─────────────────┘
```

### 창 위치 (픽셀)

| 노드 | 위치 (x, y) | 크기 (w x h) |
|------|------------|-------------|
| Tier 0 (KDCA) | (1280, 0) | 1280 x 720 |
| Tier 1 (Seoul) | (0, 720) | 1280 x 720 |
| Tier 1 (Gyeonggi) | (2560, 720) | 1280 x 720 |
| Tier 2 (Gangnam) | (0, 1440) | 1280 x 720 |
| Tier 2 (Seocho) | (1280, 1440) | 1280 x 720 |
| Tier 2 (Suwon) | (2560, 1440) | 1280 x 720 |
| Tier 2 (Seongnam) | (2560, 0) | 1280 x 720 |

**중앙 여백**: 시각화 영상 재생용 공간

---

## 🎥 녹화 절차

### 사전 준비 (5분)

1. **데이터 확인**
   ```bash
   python verify_hierarchical_data.py
   ```
   ✅ 4개 지역 데이터 파일 존재 확인

2. **시각화 영상 생성** (아직 없는 경우)
   ```bash
   python demo1_hierarchical_visualization.py
   ```
   ✅ `demo1_hierarchical.mp4` 생성 확인

3. **화면 녹화 소프트웨어 준비**
   - OBS Studio 실행
   - 소스: Display Capture (주모니터)
   - 해상도: 3840x2160 (4K) 또는 1920x1080 (FHD)
   - FPS: 30fps
   - 포맷: MP4 (H.264)

### 녹화 시작 (권장: 자동 모드)

#### 옵션 A: 자동 실행 (녹화 전문가용)

```bash
# 1. OBS Studio "녹화 시작" 대기 상태
# 2. 런처 실행
python demo_launcher.py

# 3. 메뉴에서 "2" 선택 (Auto Launch)
# 4. 5초 카운트다운 동안 OBS 녹화 시작
# 5. 모든 창이 자동으로 실행됨
```

**타임라인**:
- `0:00`: 런처 시작
- `0:05`: Tier 0 실행
- `0:07`: Tier 1 (Seoul, Gyeonggi) 실행
- `0:10`: Tier 2 (4개 지역) 실행
- `0:15`: 시각화 영상 재생

**총 예상 녹화 시간**: **약 30초** (연합학습 완료까지)

#### 옵션 B: 단계별 실행 (설명 포함)

```bash
# 1. 런처 실행
python demo_launcher.py

# 2. 메뉴에서 "1" 선택 (Sequential Launch)

# 3. 각 단계마다 Enter 입력
#    - Step 1: Tier 0 실행
#    - Step 2: Tier 1 실행
#    - Step 3: Tier 2 실행
#    - Step 4: 시각화 재생

# 4. 설명 나레이션을 하면서 진행 가능
```

**장점**: 각 단계를 설명하며 진행 가능
**단점**: 수동 입력 필요

### 녹화 중 체크리스트

- [ ] 모든 터미널 창이 화면에 보임
- [ ] 각 창의 출력이 명확히 읽힘
- [ ] Tier 2 → Tier 1 → Tier 0 순서로 데이터 흐름 확인
- [ ] 시각화 영상이 자동 재생됨
- [ ] 영상에서 DiD 그래프 결과 표시 확인

### 녹화 종료

1. 연합학습 완료 메시지 확인:
   ```
   ✅ DiD Analysis Complete!
   DiD Effect: -71.2% (t = -6.61, p < 0.001)
   ```

2. 시각화 영상 재생 (10초)

3. `Ctrl+C`로 모든 프로세스 종료

4. OBS 녹화 중지

---

## 🎨 녹화 팁 (Professional Quality)

### 화면 설정

1. **바탕화면 정리**
   - 아이콘 숨기기 (우클릭 → 보기 → 바탕화면 아이콘 표시 해제)
   - 배경화면: 단색 (진한 파랑 또는 검정 권장)

2. **작업 표시줄**
   - 자동 숨김 설정 (선택사항)
   - 알림 끄기 (Win+A → 집중 지원 ON)

3. **터미널 색상**
   - PowerShell 기본 파랑 배경이 깔끔함
   - 폰트 크기: 14pt 이상 (가독성)

### 나레이션 스크립트 (옵션)

```
[0:00] "안녕하세요, 계층형 연합학습 기반 COVID-19 정책 효과 분석 시스템을 소개합니다."

[0:05] "먼저 중앙정부 KDCA 서버가 시작됩니다."

[0:07] "다음으로 서울시와 경기도 광역지자체 서버가 연결됩니다."

[0:10] "이제 4개 시군구에서 암호화된 데이터를 전송합니다.
       강남구, 서초구는 서울시로, 수원시, 성남시는 경기도로 전송됩니다."

[0:20] "광역지자체는 동형암호 기술로 데이터를 복호화하지 않고 집계만 합니다."

[0:25] "중앙정부에서만 복호화하여 Difference-in-Differences 분석을 수행합니다."

[0:30] "분석 결과, 정책으로 감염이 71.2% 감소했음을 확인했습니다."
```

### 편집 포인트

- **인트로**: 시스템 다이어그램 오버레이 (3초)
- **메인**: 실시간 연합학습 실행 (30초)
- **시각화**: `demo1_hierarchical.mp4` 삽입 (10초)
- **아웃트로**: 결과 요약 슬라이드 (5초)

**총 영상 길이**: **45-50초**

---

## 🛠️ 문제 해결

### Issue 1: 창이 자동 배치되지 않음

**원인**: `pywin32` 미설치

**해결**:
```bash
pip install pywin32
```

재실행 후에도 문제가 있다면 수동으로 창 배치

### Issue 2: Tier 0가 계속 대기 중

**원인**: Tier 1 서버가 2개 연결되기를 기다림

**해결**:
- Seoul과 Gyeonggi 모두 실행되었는지 확인
- `--min-districts 2` 옵션 확인

### Issue 3: 시각화 영상이 재생되지 않음

**원인**: `demo1_hierarchical.mp4` 파일 없음

**해결**:
```bash
python demo1_hierarchical_visualization.py
```

### Issue 4: 데이터 파일 없음

**원인**: `data/districts/` 디렉토리에 xlsx 파일 누락

**해결**:
```bash
python data_gen_hierarchical.py
```

---

## 📊 출력 예시

### Tier 0 (KDCA)

```
======================================================================
✅ Tier 0: KDCA Central Government Started
======================================================================
Waiting for 2 provinces to connect...

✅ Seoul Province connected (treated=1)
✅ Gyeonggi Province connected (treated=0)

Starting DiD Analysis...
Treatment Effect: -1.24 (log scale)
DiD Effect: -71.2% reduction
Statistical Significance: t=-6.61 (p < 0.001)

✅ DiD Analysis Complete!
======================================================================
```

### Tier 1 (Seoul)

```
======================================================================
✅ Tier 1: Seoul Province (Treatment)
======================================================================
Port: 9001
Waiting for 2 districts to connect...

✅ Gangnam District connected
✅ Seocho District connected

Performing homomorphic aggregation (encrypted)...
Sending to Tier 0...

✅ Round 1 Complete!
======================================================================
```

### Tier 2 (Gangnam)

```
======================================================================
✅ Tier 2: Gangnam District
======================================================================
Data: seoul_gangnam.xlsx
Target: Seoul Province (127.0.0.1:9001)

Encrypting data with public key...
Sending encrypted data to Tier 1...

✅ Data sent successfully!
======================================================================
```

---

## 📹 OBS Studio 권장 설정

### 비디오 설정

- **출력 해상도**: 3840x2160 (4K) 또는 1920x1080 (FHD)
- **FPS**: 30
- **다운스케일 필터**: Lanczos (고품질)

### 인코더 설정 (H.264)

- **비트레이트**: 40000 Kbps (4K) / 6000 Kbps (FHD)
- **키프레임 간격**: 2초
- **프리셋**: Quality (고품질) 또는 Balanced

### 오디오 설정

- **마이크**: 나레이션용 (선택)
- **샘플링**: 48kHz
- **비트레이트**: 192 Kbps

---

## 🎯 최종 체크리스트

### 녹화 전

- [ ] 데이터 파일 확인 (`verify_hierarchical_data.py`)
- [ ] 시각화 영상 생성 (`demo1_hierarchical_visualization.py`)
- [ ] 바탕화면 정리
- [ ] 알림 끄기
- [ ] OBS Studio 설정 확인
- [ ] 마이크 테스트 (나레이션 녹음 시)

### 녹화 중

- [ ] 모든 7개 창 화면에 표시
- [ ] 터미널 출력 명확히 읽힘
- [ ] 데이터 흐름 시각적으로 확인
- [ ] 시각화 영상 자동 재생
- [ ] DiD 결과 표시 확인

### 녹화 후

- [ ] 영상 재생 확인
- [ ] 음성 동기화 확인
- [ ] 화질 확인 (텍스트 가독성)
- [ ] 파일 저장 (백업)

---

## 📦 데모 패키지 구성 (CES 전시용)

```
CES_2026_DEMO/
├── demo_launcher.py
├── launch_demo.bat
├── demo1_hierarchical.mp4       ← 시각화 영상
├── demo_recording.mp4            ← 녹화본
├── DEMO_RECORDING_GUIDE.md       ← 이 가이드
└── README_CES.md                 ← 부스 설명자료
```

---

## 💡 추가 개선 아이디어

### 1. 화면 오버레이

OBS에 다음 요소 추가:
- **로고**: 기관 로고 (우측 상단)
- **타이머**: 실행 경과 시간 (좌측 하단)
- **상태 표시**: "Encrypting...", "Aggregating...", "Analyzing..."

### 2. 슬로우 모션

중요 순간 강조:
- Tier 2 데이터 전송 시작
- DiD 결과 표시

### 3. 다국어 자막

영어/한국어 자막 추가 (CES 국제 관객용)

---

**작성**: 2026-01-01
**버전**: v1.0
**상태**: Production Ready

🎬 **Good luck with your CES 2026 demo!**
