# 빠른 시작 가이드

## 1. 데이터 생성 (1분)
```bash
python data_gen_hierarchical.py
python verify_hierarchical_data.py
```

## 2. 키 생성 (10초)
```bash
python setup_hierarchical.py
```

## 3. 실행 (3개 터미널)

### Terminal 1: 중앙정부
```bash
python run_tier0.py
```

### Terminal 2: 서울시
```bash
python run_tier1.py --port 9001 --region Seoul --treated --min-districts 2
```

### Terminal 3: 강남구
```bash
python run_tier2.py --data data/districts/seoul_gangnam.xlsx --target 127.0.0.1:9001
```

### Terminal 4: 서초구
```bash
python run_tier2.py --data data/districts/seoul_seocho.xlsx --target 127.0.0.1:9001
```

## 4. 결과 확인

Tier 0 화면에서 DiD 분석 결과 확인:
```
3️⃣ DiD Effect: -0.XXXX
   ✅ STRONG EFFECT: Policy reduced cases by XX%
```

## 전체 구조 테스트 (6개 이상 터미널)

상세한 가이드는 README.md 참조
