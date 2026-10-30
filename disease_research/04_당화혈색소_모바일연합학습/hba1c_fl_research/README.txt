당화혈색소 모바일 연합학습 자료 안내
정리 시각: 2026.10.16 14:01 (KST)
상태: 연구 단계(실전 미적용)

연구 루트: H:\Work\2026_deFAI\질병관련연합학습\04_당화혈색소_모바일연합학습
기존 모바일 앱·서버 구현: H:\Work\2026_deFAI\질병관련연합학습\04_당화혈색소_모바일연합학습\deFai_federated
데이터 확보·연구설계 묶음: H:\Work\2026_deFAI\질병관련연합학습\04_당화혈색소_모바일연합학습\hba1c_fl_research

우선 읽을 파일
1. 연구설계 보고서: H:\Work\2026_deFAI\질병관련연합학습\04_당화혈색소_모바일연합학습\hba1c_fl_research\report\HbA1c_FL_데이터확보_연구설계.md
2. Word 보고서: H:\Work\2026_deFAI\질병관련연합학습\04_당화혈색소_모바일연합학습\hba1c_fl_research\report\HbA1c_FL_데이터확보_연구설계.docx
3. 후보 36개 목록: H:\Work\2026_deFAI\질병관련연합학습\04_당화혈색소_모바일연합학습\hba1c_fl_research\report\후보_데이터셋_전체목록.xlsx
4. 변수사전: H:\Work\2026_deFAI\질병관련연합학습\04_당화혈색소_모바일연합학습\hba1c_fl_research\report\변수사전.xlsx
5. 현재 파일 경로·출처·체크섬: H:\Work\2026_deFAI\질병관련연합학습\04_당화혈색소_모바일연합학습\hba1c_fl_research\data\manifest_current_paths.csv

폴더별 역할
- data\public: 확보한 실제 공개 데이터. 출처별 원본 구조를 유지했다.
- data\synthetic: 합성 자료. 임상 성능 근거로 사용하지 않는다.
- docs: NHANES 공식 설명 문서와 KoGES 코드북. KoGES 참여자 원자료가 아니다.
- report: 보고서, 후보 목록, 변수사전, 기존 조사에서 생성한 요약 집계.
- scripts: 전달받은 분석·목록 생성 스크립트 원본.
- migration: 이전 전 파일 목록, 다운로드 목록 대조 결과, 임시 사본 중복 확인 기록.

경로와 재현성
- 원본 108개 파일은 내용 변경 없이 복사했다. Markdown·Word·표 파일의 역사적 경로 표기도 보존했다.
- 보고서의 R:\hba1c_fl_research 경로는 위의 현재 데이터 확보·연구설계 묶음 경로로 읽는다.
- data\manifest.csv는 수집 당시 원본 목록이다. R_path 열은 당시 위치를 나타낸다.
- data\manifest_current_paths.csv의 current_path 열은 현재 절대경로다. 원본 출처·체크섬·당시 경로도 보존했다.
- scripts의 일부 파일에는 /workspace/hba1c_fl_research 경로가 남아 있다. 이번 작업에서는 실행환경 이식과 재실행을 하지 않았다.
- 보고서의 표본 수, 접근 조건, 예측 오차는 기존 조사 결과다. 이번 작업은 파일 보존을 검증했으며 연구 결과 자체는 재검증하지 않았다.
- 탐색용 선형모형 평균절대오차 0.39 %p는 성능 입증으로 취급하지 않는다. %p는 퍼센트포인트다.

이전 검증 및 원본 보존
- R: 원본 108개 파일, 690,560,416바이트의 복사본과 SHA256(파일 내용 일치 확인용 해시)을 전수 대조했다.
- data\manifest.csv의 93개 항목은 파일 크기와 SHA256이 모두 일치했다.
- Downloads 임시 사본 17개 중 16개는 복사된 파일과 동일했다.
- 추가 KoGES_codebooks.zip 1개는 migration\staging_originals에 보존했다.
- CGMacros LICENSE.txt는 원본부터 0바이트였고 그대로 보존했다. 이 파일 자체로 이용허락 내용을 확인할 수는 없다.
- 자동 승인 검토가 원본·임시 폴더 삭제를 포함한 명령을 '정책에 의해 차단됨'으로 거부했다.
- 따라서 R:\hba1c_fl_research 및 C:\Users\Haneol\Downloads\hba1c_fl_research\_staging 은 남아 있다.

한 것 / 안 한 것
한 것: 전체 복사, 전수 해시 대조, 현재 절대경로 목록 생성, 안내문 작성, 임시 중복 확인 및 추가 압축파일 보존.
안 한 것: 원본·임시 사본 삭제(정책 차단), 연구 결과 재계산, 외부 접근 조건 재조사, 모델 검증, 분석 스크립트 수정·실행.
