# 📹 계층형 F-Causal 시각화 가이드

## 개요

이 문서는 3-Tier 계층형 연합학습 구조와 DiD 분석 결과를 시각화한 영상 생성 가이드입니다.

---

## 🎬 생성된 영상 정보

### 파일명
`demo1_hierarchical.mp4`

### 사양
- **해상도**: 1920x1080 (Full HD, 16:9)
- **길이**: 10초 (300 프레임)
- **프레임레이트**: 30 fps
- **코덱**: H.264 (libx264)
- **용량**: ~200KB

---

## 🏛️ 시각화 내용

### 실제 계층 구조

```
Tier 0 (중앙정부 - KDCA)
    ↑                 ↑
    |                 |
Tier 1 (광역자치단체)
서울시            경기도
(Treatment)       (Control)
    ↑                 ↑
    |                 |
Tier 2 (기초자치단체)
강남구, 서초구    수원시, 성남시
```

### 실제 테스트 결과 반영

#### 1. Treatment Group (서울시)
- **지역**: 강남구, 서초구
- **Pre-period**: log=4.45 (평균 86건/일)
- **Post-period**: log=4.89 (평균 134건/일)
- **변화**: +0.44 (+55.3%)

#### 2. Control Group (경기도)
- **지역**: 수원시, 성남시
- **Pre-period**: log=5.49 (평균 243건/일)
- **Post-period**: log=7.18 (평균 1,308건/일)
- **변화**: +1.68 (+438.4%)

#### 3. DiD 효과
- **계수**: -1.24
- **해석**: 정책으로 인해 서울시 감염이 **71.2% 감소**
- **통계적 유의성**: t = -6.61 (p < 0.001) ***

---

## 📊 영상 구성 (Phase별)

### Phase 1: 데이터 생성 (0 ~ 3.5초)
**내용**:
- Tier 2 (시군구)에서 데이터 생성
- 파란색 입자: 서울 (강남구, 서초구)
- 빨간색 입자: 경기 (수원시, 성남시)

**연출**:
- 하단 노드에서 입자가 상승
- 중간 Tier 1 노드로 이동

**자막**:
```
Phase 1: 데이터 생성 (Tier 2)
시군구 단위 데이터 암호화 및 전송
```

---

### Phase 2: 동형암호 집계 (3.5 ~ 5초)
**내용**:
- Tier 1에서 동형암호 연산 수행
- **핵심**: 광역지자체는 데이터 내용을 볼 수 없음

**연출**:
- 입자가 황금색(#ffd700)으로 변화
- "ENCRYPTED" 라벨 표시 (검은 배경, 황금 테두리)
- Pulsing 효과로 암호화 강조

**자막**:
```
Phase 2: 동형암호 집계 (Tier 1)
광역지자체는 데이터 내용을 볼 수 없음 (Blind Aggregation)
```

**보안 특징**:
- Public Key만 사용
- 복호화 불가 (No Secret Key)
- Homomorphic Addition만 수행

---

### Phase 3: 중앙 전송 (5 ~ 7초)
**내용**:
- 암호화된 집계 데이터를 Tier 0으로 전송
- 여전히 황금색 유지 (암호화 상태)

**연출**:
- Tier 1 → Tier 0 이동
- ENCRYPTED 라벨 유지

**자막**:
```
Phase 3: 중앙 전송 (Tier 1 → Tier 0)
암호화된 집계 데이터만 중앙정부로 전송
```

---

### Phase 4: DiD 분석 결과 (7 ~ 10초)
**내용**:
- Tier 0에서 Secret Key로 복호화
- DiD 그래프 팝업으로 결과 표시

**연출**:
- 입자 사라짐
- 우측 하단에 DiD 그래프 등장
- 3개 라인 애니메이션:
  1. **검은 점선**: Counterfactual (정책이 없었다면)
  2. **파란 실선**: Treatment (서울, 실제)
  3. **빨간 실선**: Control (경기)

**그래프 요소**:
- X축: Time (0~10일)
- Y축: Log(Cases)
- 세로선: 정책 개입 시점 (Day 5)
- 범례: 상단 좌측

**결과 텍스트** (우측 상단):
```
DiD Effect: -71.2%
(t = -6.61, p < 0.001)
```

**자막**:
```
Phase 4: DiD 분석 결과
정책 효과: -71.2% (통계적으로 유의)
```

---

## 🎨 색상 코드

| 요소 | 색상 | 의미 |
|------|------|------|
| 서울시 노드 | `#1976d2` (파랑) | Treatment Group |
| 경기도 노드 | `#d32f2f` (빨강) | Control Group |
| KDCA 노드 | `#1a237e` (진한 남색) | 중앙정부 |
| 암호화 입자 | `#ffd700` (금색) | Encrypted Data |
| Raw Data (서울) | `#1976d2` (파랑) | 처리 전 데이터 |
| Raw Data (경기) | `#d32f2f` (빨강) | 처리 전 데이터 |
| 배경 | `#f8f9fa` (밝은 회색) | 학술적 톤 |

---

## 🚀 실행 방법

### 요구사항
```bash
pip install numpy matplotlib
```

### 영상 생성
```bash
python demo1_hierarchical_visualization.py
```

### 출력
```
======================================================================
✅ 계층형 시각화 영상 생성 완료!
======================================================================
   파일: demo1_hierarchical.mp4
   해상도: 1920x1080 (16:9)
   길이: 10초 (300 프레임 @ 30fps)
   내용:
     - Tier 2: 강남구, 서초구, 수원시, 성남시
     - Tier 1: 서울시 (Treatment), 경기도 (Control)
     - Tier 0: KDCA 중앙정부
     - DiD 효과: -71.2% (정책으로 감염 71.2% 감소)
     - 통계적 유의성: t = -6.61 (p < 0.001)
======================================================================
```

---

## 📝 MagicLight 편집 가이드

### Nano Banana 모델 프롬프트 예시

**Context**:
```
High-tech hierarchical network diagram showing South Korean COVID-19
policy analysis. Three-tier architecture with districts (Gangnam, Seocho,
Suwon, Seongnam) at bottom, provinces (Seoul, Gyeonggi) in middle,
and central government (KDCA) at top. Data flows upward with golden
encrypted particles. Clean white background with blue/red color scheme.
```

**Keywords**:
```
3-Tier Architecture, Federated Learning, Lock Icon, Golden Data Packets,
Network Topology, White Blueprint, DiD Graph, Privacy-Preserving,
Homomorphic Encryption, South Korea Map, Medical Data Security
```

**Style**:
```
Academic visualization, minimalist design, technical diagram,
IEEE/Nature publication quality, sans-serif fonts
```

---

## 🔊 오디오 추천 (SFX)

### Phase 1 (데이터 생성)
- **사운드**: 가벼운 디지털 신호음
- **예시**: `Blip-blip-blip` (데이터 전송)
- **음량**: -15dB

### Phase 2 (암호화)
- **사운드**: 자물쇠 잠기는 소리 + 중후한 Humming
- **예시**: `Click-lock` + `Hummmmm`
- **음량**: -10dB
- **강조점**: "볼 수 없음"을 청각적으로 표현

### Phase 3 (전송)
- **사운드**: 디지털 전송음
- **예시**: `Whoosh` (부드러운 바람 소리)
- **음량**: -12dB

### Phase 4 (결과)
- **사운드**: 웅장한 Bass Drop → 명쾌한 성공음
- **예시**: `Boom` → `Chime-ding`
- **음량**: -8dB
- **강조점**: 중요한 발견의 순간

---

## 📐 기술 세부사항

### 노드 크기
- Tier 0 (KDCA): 0.5 반경
- Tier 1 (광역): 0.4 반경
- Tier 2 (시군구): 0.25 반경

### 입자 애니메이션
- 총 8개 입자 (각 district당 2개)
- Phase 1: Linear interpolation (선형 이동)
- Phase 2: Pulsing effect (sin 함수 활용)
- Phase 3: Linear interpolation
- Phase 4: Fade out (alpha=0)

### DiD 그래프
- Inset 위치: [0.62, 0.15, 0.35, 0.35]
- 애니메이션: 70%~100% 구간에서 progressive drawing
- 결과 텍스트: 85% 이후 표시

---

## 🎯 핵심 메시지

### Privacy-Preserving Aggregation
> **"Tier 1 광역지자체는 암호화된 데이터를 집계만 하고,
> 내용은 전혀 볼 수 없습니다!"**

### Causal Inference
> **"정책으로 인해 서울시 COVID-19 감염이 71.2% 감소했으며,
> 이는 통계적으로 매우 유의미합니다 (p < 0.001)"**

---

## 📚 참고 자료

### 실제 실행 결과
```bash
# 테스트 로그 참조
🏙️ Province: Seoul (treated=1, 2 districts)
   Pre:  n=57, log_mean=4.4546, std=0.8613
   Post: n=148, log_mean=4.8946, std=0.8427
   Change: +0.4400

🏙️ Province: Gyeonggi (treated=0, 2 districts)
   Pre:  n=57, log_mean=5.4929, std=0.8749
   Post: n=149, log_mean=7.1763, std=0.7876
   Change: +1.6834

DiD Coefficient: -1.2434
Interpretation: Policy reduced cases by 71.2%
t-statistic: -6.61
Significance: *** (p < 0.05)
```

### 관련 파일
- `demo1_hierarchical_visualization.py`: 시각화 스크립트
- `run_tier0.py`, `run_tier1.py`, `run_tier2.py`: 실제 FL 구현
- `verify_hierarchical_data.py`: 데이터 검증 스크립트

---

## 💡 추가 개선 아이디어

1. **사운드 디자인**: 각 Phase별 distinct SFX 추가
2. **3D 효과**: 노드에 그림자 추가하여 입체감 강화
3. **통계 수치 애니메이션**: CountUp 효과로 숫자 표시
4. **지도 배경**: 한반도 실루엣을 배경에 희미하게 표시
5. **QR 코드**: 논문/GitHub 링크를 우측 하단에 삽입

---

**생성일**: 2026-01-01
**버전**: 1.0
**작성자**: F-Causal Research Team
