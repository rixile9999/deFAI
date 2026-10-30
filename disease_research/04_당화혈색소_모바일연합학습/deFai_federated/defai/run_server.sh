#!/bin/bash

echo "🌸 단일 레이어 HbA1c 예측 연합학습 서버 실행 🌸"
echo "포트: 1501"
echo "=" * 50

# Python 가상환경 확인 및 활성화
if [ -d "venv" ]; then
    echo "🔧 가상환경 활성화 중..."
    source venv/bin/activate
else
    echo "🔧 가상환경 생성 중..."
    python3 -m venv venv
    source venv/bin/activate
    
    echo "📦 필요한 패키지 설치 중..."
    pip install flask tensorflow numpy
fi

echo "🚀 서버 시작 중..."
python3 simple_server.py 