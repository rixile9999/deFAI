#!/bin/bash

echo "🌸 단일 레이어 HbA1c 예측 연합학습 클라이언트 실행 🌸"
echo "=" * 50

# Gradle 빌드
echo "🔧 프로젝트 빌드 중..."
./gradlew build

if [ $? -eq 0 ]; then
    echo "✅ 빌드 성공"
    echo "🚀 클라이언트 실행 중..."
    ./gradlew run
else
    echo "❌ 빌드 실패"
    exit 1
fi 