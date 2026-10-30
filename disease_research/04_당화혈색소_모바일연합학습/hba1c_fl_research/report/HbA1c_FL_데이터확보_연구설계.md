# HbA1c 예측 모바일 연합학습 연구: 데이터 확보와 연구 설계

- 작성일: 2026-10-16 (KST)
- 작업 폴더: `R:\hba1c_fl_research\` (box 원본: `/workspace/hba1c_fl_research/`)
- 함께 낸 파일: `report/후보_데이터셋_전체목록.xlsx`(36개 후보, 27개 항목), `report/변수사전.xlsx`, `data/manifest.csv`(파일 출처·크기·SHA256), 분석 스크립트 `scripts/`
- 확인수준 표기
  - **원자료확인**: 이번에 원자료를 직접 받아 집계했다.
  - **공식변수사전확인**: 공식 코드북이나 데이터 사전에서 변수를 확인했다. 원자료는 받지 않았다.
  - **문헌확인**: 원논문이나 공식 안내문의 서술로 확인했다.
  - **미확인**: 이번 조사에서 근거를 찾지 못했다.
- 이 보고서의 표본수·변수·접근조건은 모두 출처가 있는 값이다. 근거가 없는 항목은 '미확인'으로 적었다. 예측 성능이나 임상 유효성은 주장하지 않는다.

---

## ① 연구 목적과 확보 전략

### 1.1 배경과 현재 시제품
- **HbA1c(당화혈색소, glycated hemoglobin A1c)**: 적혈구 혈색소에 포도당이 결합한 비율이다. 지난 수개월의 평균 혈당을 반영하는 검사값이다.
- **연합학습(FL, Federated Learning)**: 원자료는 각 기기에 둔다. 기기에서 모델을 학습하고, 서버는 모델 업데이트만 모아 집계한다.
- 현재 시제품
  - 입력 2개(혈당, 체중), 출력 1개(HbA1c 연속값)인 단일 선형층이다.
  - 앱은 사용자 입력 1건으로 학습한다.
  - 예제 데이터는 합성 데이터이며, 임상 성능은 입증되지 않았다.
- 확장 계획
  - **CGM(연속혈당측정, Continuous Glucose Monitoring)**, 체중 변화, 걸음수·활동량 시계열을 추가한다.
  - 개인별 분포 차이(non-IID)를 고려한 연합학습과 개인화를 적용한다.
  - 암호화·**보안집계(secure aggregation)**·개인화는 아직 구현·검증 여부가 확인되지 않았다. §7에서 별도로 점검한다.

### 1.2 연구 질문 구분

| 질문 | 정의 | 데이터 필수조건 |
|---|---|---|
| **A. 현재 HbA1c 추정** | 같은 사람의 실측 HbA1c 검사값을, 그 검사 직전 기간의 혈당·체중 입력과 연결해 추정한다 | 동일인 연결, 검사일과 입력 관측일의 근접성 |
| **B. 미래 HbA1c 예측** | 예측 시점 t0까지의 입력(관측기간 W)으로 t0+h(예측 간격) 시점의 HbA1c를 예측한다 | 동일인 반복 HbA1c, t0 이후 정보 차단(누출 방지) |
| **C. 모바일 개인별 FL·개인화** | 사람(기기) 단위로 반복 관측이 쌓이며 개인 모델이 개선되는지 본다 | 사람별 기록 수와 반복 HbA1c 횟수. 1회 검사 자료로는 개인화 '평가'를 할 수 없다 |

표의 의미: A는 횡단면 자료(①분류)로도 검증할 수 있다. B와 C는 **반복 HbA1c가 있는 종단 자료**가 있어야 한다. 이번에 즉시 받은 공개 자료는 모두 1인 1회 HbA1c였다. 따라서 B·C에는 신청형 자료나 신규 수집이 반드시 필요하다.

### 1.3 확보 전략과 우선순위
우선순위는 지정된 순서를 따랐다: (1) 실제 검사값 → (2) 동일인 연결 → (3) 시간정보 → (4) 반복관측 → (5) 접근가능성 → (6) 모바일 활용성.

1. **즉시 확보(공개)**: 이용조건이 허용하는 원본과 변수사전을 받았다. 실제로 쓸 수 있는 표본수(N)를 직접 계산했다.
   - 받은 자료: NHANES 2주기, ShanghaiT1DM/T2DM, CGMacros, Hall 2018, BIG IDEAs
2. **간이 신청형**: 회원가입이나 약관 동의만 필요하다. 개인정보 제출이 따르므로 사용자 본인이 직접 해야 한다.
   - KNHANES, JAEB 공개 데이터셋, AI-READI, CHARLS, CHNS
3. **정식 신청형**: IRB(기관생명윤리위원회) 승인, DUA(데이터 이용 계약), 심의, 비용이 필요하다.
   - KoGES, NHIS, UK Biobank, All of Us, NIDDK-CR, BioLINCC, Vivli, MIMIC-IV
4. **신규 수집**: 기존 자료로 채우기 어려운 조건(한국인, 모바일 자가입력, 반복 HbA1c + CGM + 활동량이 동일인에게 시간 연결된 자료)을 위한 최소안과 확장안을 설계했다.

### 1.4 핵심 결론 요약
- **A 질문(현재 모델)에 바로 쓸 수 있는 원자료**: NHANES 2021–2023의 HbA1c + 공복혈당 + 체중 연결 표본 **3,631명**, 2017–2018은 **2,840명**이다(원자료확인). 다만 정맥 공복 혈장포도당이므로 앱의 자가혈당(SMBG) 입력과 측정 맥락이 다르다.
- **국내 종단 최우선 후보**: KoGES 안산·안성 코호트. 코드북 15종을 확인한 결과, 기반조사(2001–02)부터 10차 추적(2021–22)까지 모든 차수에 HbA1c(%), 공복혈당, 실측 체중이 있다(공식변수사전확인). 신청이 필요하다.
- **CGM에서 HbA1c로 확장하는 연구의 최우선 공개원**: JAEB 공개 데이터셋(특히 GMI Public Dataset, REPLACE-BG, DCLP3, CITY, WISDM)과 AI-READI. 둘 다 약관 동의나 신청이 필요하다.
- **개인 반복관측(C)을 충족하는 즉시 확보 공개자료는 없다.** 그래서 신규 수집의 확장안이 필요하다.
- 현 시제품은 **1건 학습으로 만든 단일 선형층 업데이트에서 서버가 입력값과 HbA1c를 거의 그대로 역산할 수 있는 구조**다(§7.6). 원자료를 전송하지 않는다는 것만으로 프라이버시가 완성되지 않는다.

---

## ② 조사한 출처 유형과 탐색하지 않은 영역

| 출처 유형 | 이번에 조사한 대표 출처 | 비고 |
|---|---|---|
| 국가 건강·영양조사, 건강검진 | NHANES(미국), KNHANES(한국), NHIS 검진코호트, 일본 NDB, CHNS·CHARLS·ELSA·HRS | NHANES만 원자료 확보 |
| 청구·공공 빅데이터 | NHIS 표본코호트, HIRA, 보건의료 빅데이터 통합 플랫폼 | HbA1c 결과값 없음(③) 또는 미확인 |
| 병원 EMR(전자의무기록)·다기관 | MIMIC-IV, eICU, K-MIMIC, 국내 OMOP CDM(공통데이터모델) 분산연구망, All of Us | 모두 자격인증·신청형 |
| 당뇨 종단 코호트·임상시험 | KoGES, UK Biobank, Maastricht, NIDDK-CR(DCCT/EDIC, Look AHEAD, DPP, TODAY), BioLINCC ACCORD, T1DEXI(Vivli) | 반복 HbA1c 보유. 모두 신청형 |
| CGM·자가혈당 | JAEB 공개 데이터셋 약 40종, OhioT1DM, ShanghaiT1DM/T2DM, D1NAMO, CGMacros, Hall 2018, BIG IDEAs | 공개 4종 확보 |
| 모바일·웨어러블 | All of Us Fitbit, AI-READI(Garmin), CGMacros(Fitbit), BIG IDEAs(Empatica) | — |
| 논문 부속자료·공식 저장소 | PLOS(Hall), figshare(Shanghai), PhysioNet, Zenodo | — |
| ML 예제 저장소 | Kaggle, UCI | ③·④로 분류해 제외 |

**이번에 탐색하지 않은 영역** (따라서 여기서 "적합 자료 없음"이라고 결론 내리지 않는다)
- 국내
  - 개별 대학병원과의 직접 데이터 협약(병원 자체 CGM·HbA1c 레지스트리)
  - 대한당뇨병학회 레지스트리, 1형 당뇨 레지스트리
  - 국내 CGM·혈당측정기 제조사와 헬스앱의 데이터
  - KNHANES 추적조사 연계자료
  - 국민건강보험 맞춤형 DB의 세부 변수
- 해외
  - 북유럽 국가 레지스트리(스웨덴 NDR 등), FinnGen, 에스토니아 바이오뱅크
  - 대만 NHIRD, 싱가포르, 인도 코호트
  - 유럽 IMI 컨소시엄(예: DIRECT, Hypo-RESOLVE)
  - YODA, ClinicalStudyDataRequest, Project Data Sphere 등 제약 임상시험 공유 플랫폼(Vivli는 T1DEXI만 확인)
  - NHANES 과거 주기(1999–2016)와 가속도계 모듈
  - Glucose-ML 카탈로그(사이트 로딩 실패로 목록 미확인)
  - Awesome-CGM 목록(페이지 본문 미수집)
  - 중국어 문헌 데이터베이스(CNKI)
- 상업 데이터: Optum, IQVIA, TriNetX(유료·계약형이라 이번 범위에서 제외)

---

## ③ 후보 비교표 (요약)
열 27개짜리 전체 표는 `report/후보_데이터셋_전체목록.xlsx`에 있다. 분류 기준은 다음과 같다.
- ① HbA1c·혈당·체중이 동일인에 연결됨
- ② HbA1c + 일부 입력만 있음
- ③ HbA1c 없음(보조용)
- ④ 합성·출처불명·검증불가

| ID | 데이터셋 | 분류 | 확인수준 | 실측 HbA1c / 혈당 종류 / 체중 | 반복 HbA1c | 접근 | A | B | C | 공식 근거 |
|---|---|---|---|---|---|---|---|---|---|---|
| D01 | NHANES 2021–2023 | ① | 원자료 | LBXGH % / 공복 혈장 LBXGLU / BMXWT | 없음 | 공개 | 높음 | 불가 | 불가 | wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/GHB_L.htm |
| D02 | NHANES 2017–2018 | ① | 원자료 | 동일 | 없음 | 공개 | 높음 | 불가 | 불가 | …/2017/DataFiles/GHB_J.htm |
| D03 | KNHANES | ① | 문헌 | HE_HbA1c / HE_glu 공복 / HE_wt | 없음 | 회원가입·서약 | 높음(한국인) | 불가 | 불가 | knhanes.kdca.go.kr |
| D04 | KoGES 안산·안성 | ① | 공식변수사전 | ASn_HBA1C % / ASn_GLU0 공복 / ASn_WEIGHT | 최대 11회(약 2년 간격) | IRB+분양심의 | 높음 | 높음 | 중간 | nih.go.kr menuNo=300566/300577 |
| D05 | KoGES HEXA·CAVAS | ② | 공식변수사전 | 일부 연도만 HbA1c | 2회 내외 | 동일 | 중간 | 중간 | 낮음 | 동일 |
| D06 | NHIS 표본·검진코호트 | ③ | 문헌 | HbA1c 없음 / 공복혈당 / 체중 | — | IRB+심의+유료 | 불가 | 불가 | 불가 | nhiss.nhis.or.kr |
| D07 | HIRA 환자표본 | ③ | 문헌 | 검사 시행 코드만 | — | 심의+수수료 | 불가 | 불가 | 불가 | opendata.hira.or.kr |
| D08 | 복지부 통합플랫폼·CDM 분산망 | ②(미확인) | 문헌 | OMOP MEASUREMENT 가능성 | 가능(미확인) | IRB+심의, 결과만 반출 | 미확인 | 잠재 높음 | 기관단위 | hcdl.mohw.go.kr |
| D09 | K-MIMIC | ③/미확인 | 문헌 | 미확인 | — | IRB+약정 | 낮음 | 불가 | 불가 | khdp.net |
| D10 | UK Biobank | ① | 공식변수사전(30750) | mmol/mol HPLC / 비공복 혈청(미검증) / 체중 | 2회(14,257명) | 신청+£3–9천 | 높음 | 제한 | 낮음 | biobank.ndph.ox.ac.uk/ukb/field.cgi?id=30750 |
| D11 | All of Us | ① | 문헌 | EHR + Fitbit | EHR 반복 | 기관 DURA+교육 | 중간 | 높음 | 높음(모의) | researchallofus.org |
| D12 | MIMIC-IV | ① | 문헌 | itemid 50852 / 혼재 / OMR | 일부 | 자격인증+DUA | 중간 | 중간 | 낮음 | physionet.org/content/mimiciv |
| D13 | eICU | ② | 문헌 | lab HbA1c / ICU | — | 자격인증+DUA | 낮음 | 불가 | 기관단위 | physionet.org/content/eicu-crd/2.0 |
| D14 | DCCT/EDIC | ① | 문헌 | 중앙검사실 / 7점 SMBG / 체중 | 분기→연 1회 | NIDDK-CR | 높음 | 매우 높음 | 높음 | repository.niddk.nih.gov/study/16 |
| D15 | Look AHEAD | ① | 문헌 | 있음 / 공복 / 체중(주결과) | 연 1회(미확인) | NIDDK-CR | 높음 | 높음 | 중간 | repository.niddk.nih.gov/study/61 |
| D16 | DPP/DPPOS, TODAY | ① | 문헌 | 있음 / 공복·OGTT / 체중 | 반복 | NIDDK-CR | 높음 | 높음 | 중간 | repository.niddk.nih.gov |
| D17 | ACCORD | ① | 데이터사전(문헌 경유) | hba1c(MaskID, Visit) | 방문별 | BioLINCC | 높음 | 높음 | 중~높 | biolincc.nhlbi.nih.gov/studies/accord |
| D18 | JAEB 공개(REPLACE-BG·DCLP3·CITY·WISDM·GMI 등) | ② | 공식목록 / 파일 변수 미확인 | 중앙검사실(연구별) / CGM / 미확인 | 연구별 2–4회 | 성명·소속·목적 입력+약관 | 높음 | 높음 | 중간 | public.jaeb.org/datasets/diabetes |
| D19 | T1DEXI | ② | 문헌 | 기저 / Dexcom G6 / 미확인 | 미확인 | Vivli DUA | 중간 | 낮음 | 중간 | search.vivli.org (PR00008428) |
| D20 | OhioT1DM | ③ | 문헌 | 없음 / CGM | — | DUA | 불가 | 불가 | 보조 | (charlotte.edu OhioT1DM 페이지) |
| D21 | ShanghaiT1DM/T2DM | ① | 원자료 | mmol/mol(변환값 추정) / 공복·식후2h·CGM / 체중 | 극소수 | 공개 CC BY | 중간 | 낮음 | 낮음 | doi.org/10.6084/m9.figshare.20444397.v3 |
| D22 | D1NAMO | ③ | 문헌 | 미확인 / CGM | — | 조건 불일치 | 불가 | 불가 | 보조 | zenodo.org/records/5651217 |
| D23 | CGMacros | ① | 원자료 | % (사전 단위 오기) / 공복+CGM 2종 / 체중 lb | 1회 | 공개 CC BY-NC-SA | 중~높 | 불가 | 제한 | physionet.org/content/cgmacros/1.0.0 |
| D24 | BIG IDEAs | ② | 원자료 | % / Dexcom G6 / 없음 | 1회 | 공개 ODC-By | 낮음 | 불가 | 제한 | physionet.org/content/big-ideas-glycemic-wearable/1.1.3 |
| D25 | Hall 2018 Glucotypes | ① | 원자료 | A1C % / FBG+CGM / Weight | 1회 | 공개 CC BY | 중간 | 불가 | 제한 | doi.org/10.1371/journal.pbio.2005143 |
| D26 | Maastricht Study | ① | 문헌 | 있음 / OGTT·CGM / 있음 | 미확인 | 신청+관리비 | 높음 | 미확인 | 중간 | demaastrichtstudie.nl/research/data-guidelines |
| D27 | AI-READI | ① | 문헌 | "HbA1c (%)" / Dexcom G6 / 미확인 | 1회 | 로그인+라이선스 | 높음 | 불가 | 제한 | docs.aireadi.org |
| D28 | CHARLS | ① | 문헌 | 2011·2015 / 혈장 / 체중 | 2회 | 등록 | 높음 | 중간 | 낮음 | charls.pku.edu.cn |
| D29 | CHNS | ① | 문헌 | 2009 / 공복 / 체중 | 미확인 | 등록 | 높음 | 미확인 | 낮음 | chns.cpc.unc.edu |
| D30 | ELSA | ② | 문헌 | 있음 / 미확인 / 체중 | 약 4년 간격 | UKDS 라이선스 | 중간 | 중간 | 낮음 | elsa-project.ac.uk |
| D31 | HRS DBS / CLHLS | 미확인 | 미확인 | DBS HbA1c(검증 실패) | — | 신청 | — | — | — | hrs.isr.umich.edu |
| D32 | 일본 NDB 등 | ③(오픈데이터) | 문헌 | 집계표 | — | 개인자료는 제3자제공 신청 | 불가 | — | — | mhlw.go.jp |
| D33 | Pima | ③ | 문헌 | 없음 | — | 공개 | 불가 | 불가 | 불가 | kaggle(uciml) |
| D34 | Kaggle diabetes prediction | ④ | 출처불명 | HbA1c_level(검증불가) | — | 공개 | 불가 | 불가 | 불가 | kaggle(iammustafatz) |
| D35 | UCI 130-US hospitals | ③ | 문헌 | A1Cresult 범주형 | — | 공개 | 불가 | 불가 | 불가 | archive.ics.uci.edu/dataset/296 |
| D36 | JAEB CGMND / GLAM | 미확인 | 공식목록 | 미확인 | — | 약관 | — | — | — | public.jaeb.org |

표의 의미
- **현재 모델(A) 검증용**: 원자료가 이미 확보된 NHANES(D01·D02)를 1차로 쓴다. 한국인 외부검증은 KNHANES(D03)로 하며, 사용자 본인의 회원가입과 서약이 필요하다.
- **미래 예측(B)**: KoGES(D04)와 NIDDK-CR·ACCORD(D14–D17)만이 사람당 여러 번의 HbA1c를 제공한다. 모두 IRB와 계약이 필요하다.
- **CGM 확장**: JAEB(D18)와 AI-READI(D27)가 가장 접근성이 좋다. 그래도 약관 동의 단계가 있다.
- **③·④ 자료(D06, D07, D20, D22, D32–D35)는 정답 데이터로 쓰지 않는다.** 서로 다른 데이터셋을 임의로 조합하는 것도 하지 않는다(동일인이 아니기 때문).

---

## ④ 실제 확보한 자료와, 신청·추가확인이 필요한 자료

### 4.1 확보 완료 (모두 `data/manifest.csv`에 SHA256 기록)

| 데이터셋 | 받은 파일 | 크기 | 이용조건 | 무결성 | 저장 위치 |
|---|---|---|---|---|---|
| NHANES 2021–2023 | GHB_L, GLU_L, BMX_L, DEMO_L, DIQ_L, CBC_L, FASTQX_L (.xpt) | 8.2 MB | 공공 이용자료(재식별 금지) | SHA256 기록(공식 체크섬 없음) | `data\public\NHANES_2021-2023\` |
| NHANES 2017–2018 | 같은 구성 _J | 11.7 MB | 동일 | 동일 | `data\public\NHANES_2017-2018\` |
| NHANES 코드북 | 위 14개 파일의 .htm | 0.8 MB | 공개 문서 | — | `docs\NHANES_*.htm` |
| KoGES 코드북 | 지역사회 기반~10차, HEXA, CAVAS 15종 .xlsx | 4.5 MB | 공개 문서(원자료는 신청형) | — | `docs\KoGES\` |
| CGMacros v1.0.0 | CGMacros_dateshifted365.zip(627 MB), DataDictionary(.pdf/.csv), LICENSE, SHA256SUMS, 압축에서 꺼낸 bio.csv | 658 MB | CC BY-NC-SA 4.0 | **공식 SHA256SUMS 7/7 일치** | `data\public\CGMacros_1.0.0\` |
| BIG IDEAs v1.1.3 | Demographics.csv, 16명 Dexcom·Food_Log CSV, LICENSE, SHA256SUMS | 2.4 MB | ODC-By 1.0 | **공식 SHA256 34/34 일치** | `data\public\BIG_IDEAs_1.1.3\` |
| Hall 2018 | S1_Data_CGM.tsv.gz, S5_Data_clinical.sqlite | 0.8 MB | CC BY 4.0 | SHA256 기록 | `data\public\Hall2018_glucotypes\` |
| ShanghaiT1DM/T2DM v3 | data.zip | 3.4 MB | CC BY 4.0 | **figshare MD5 일치** | `data\public\Shanghai_T1DM_T2DM\` |
| 합성 데이터(별도 관리) | synthetic_fl_smoketest.csv, 생성 스크립트, README | 12 KB | — | — | `data\synthetic\` |

대용량 자료는 우선순위대로 처리했다.
- **BIG IDEAs**: 전체 34.1 GB(zip 4.7 GB)의 대부분이 Empatica 원신호다(ACC 879 MB, BVP 1.37 GB 등/1인). HbA1c와 CGM이 들어 있는 소형 파일만 먼저 받았다.
- **CGMacros**: 627 MB zip의 대부분은 식사 사진이다. 분석에 필요한 bio.csv와 1인당 CSV는 압축 안에 있어서 zip 전체를 받았다.
- **ShanghaiT1DM/T2DM**: figshare가 box의 IP를 차단해, 사용자 PC에서 공식 URL로 직접 받았다.

### 4.2 원자료확인 결과 (실제 사용가능 N, 결측, 반복)

**NHANES** (스크립트 `scripts/nhanes_linkage.py`, 결과 `report/NHANES_linkage_summary.json`)

| 항목 | 2021–2023 (L) | 2017–2018 (J) |
|---|---|---|
| 인구학 파일 인원 / 검진 인원 | 11,933 / 8,860 | 9,254 / 8,704 |
| HbA1c 비결측(12세 이상 검진 대상) | 6,715 (결측 6.7%) | 6,045 (결측 5.6%) |
| 공복 하위표본 배정 / FPG 비결측 | 3,996 / 3,672 (결측 8.1%) | 3,036 / 2,891 (결측 4.8%) |
| **HbA1c + 공복혈당 + 체중 모두 있음** | **3,631** | **2,840** |
| 그중 공복 8–24시간 확인 | 3,324 | 2,665 |
| 18세 이상 / 12–17세 | 3,290 / 341 | 2,491 / 349 |
| 자가보고 당뇨 / 인슐린 사용 | 440 / 130 | 408 / 119 |
| HbA1c ≥6.5% / ≥8.0% | 400 / 138 | 368 / 125 |
| 임신(RIDEXPRG=1) | 24 | 20 |
| 빈혈(Hb: 남 <13, 여 <12 g/dL) | 366 (Hb 결측 15) | 286 |
| 사람당 기록 수 | 1 (SEQN 고유) | 1 |
| HbA1c 평균±SD (범위) | 5.70±1.00 % (3.2–13.7) | 5.79±1.06 % (4.1–15.2) |

2021–2023 연결표본의 집단별 HbA1c 평균(전체는 `report/NHANES_group_distribution.csv`)
- 성별: 여 5.65 (n=1,996), 남 5.77 (n=1,635)
- 연령: 12–17세 5.22, 18–39세 5.30, 40–59세 5.83, 60세 이상 5.97
- 인종: 비히스패닉 백인 5.63, 비히스패닉 흑인 5.90, 멕시코계 5.83, 비히스패닉 아시아계 5.75 (n=208)
- 자가보고 당뇨: 7.38±1.58 (n=440), 비당뇨: 5.45±0.56

> 탐색적 실현가능성 점검(임상성능 아님): 18세 이상, 가중치 미적용, 사람 단위 무작위 5분할 교차검증으로 계산했다.
> - 평균예측: MAE 0.62, RMSE 1.03 %p
> - 공복혈당+체중 선형회귀: MAE 0.39, RMSE 0.60 %p
> - 자가보고 당뇨군만: 선형회귀 MAE 0.85 %p (오차 절대값의 SD 0.88)
>
> 이 수치는 같은 조사 안의 횡단면 분할이다. 모바일 자가입력 환경, 다른 인구, 미래 시점, 연합학습 성능을 보여주지 않는다. 표본수 계산(§5.5)의 가정 입력값으로만 쓴다.

**ShanghaiT1DM/T2DM** (`report/Shanghai_linkage_summary.json`)
- T1DM: 12명 16기간. HbA1c·FPG·체중이 모두 있는 것은 **15기간/11명**이다. 3기간을 가진 2명은 기간별로 다른 HbA1c를 갖는다.
- T2DM: 100명 109기간. 모두 있는 것은 **101기간/92명**이다(HbA1c 결측 8, FPG 결측 3).
  - 2–3기간을 가진 8명 중 4명은 HbA1c가 기간 간 **같은 값으로 반복**된다. 같은 검사값을 재사용한 것일 수 있으므로 반복검사로 보지 않는다.
- HbA1c 단위는 mmol/mol로 표기되어 있다. 그러나 값이 %를 마스터방정식으로 변환한 형태(예: 115.311 → 12.7%)라서, 원래 측정 단위는 %로 추정된다.
- 검사일은 제공되지 않는다. 논문은 "CGM 전후 6개월 이내"라고만 기술한다. 따라서 **A에서도 시간 근접성이 보장되지 않는다.**
- CGM은 15분 간격이며, 파일당 판독 수 중앙값은 T1 941, T2 1,218이다.
- zip 안에 Office 임시파일(`~$*.xlsx`) 3개가 섞여 있다. 논문의 기간 수 서술(T1 '8명 1기간')과 실제 파일(10명)이 다르다.

**CGMacros**
- 45명 전원이 HbA1c·공복혈당·체중을 갖는다(**사용가능 45명**).
- 데이터 사전은 HbA1c 단위를 'mmol/mol'로 적었지만, 실제 값은 4.6–8.5다. **%로 판단되며 사전 오기로 보인다.**
- 체중은 lb 단위다.
- CGM은 1인 평균 10.9일이다. Libre는 1분 행에 값이 꽉 차 있어 **보간값으로 보인다**(원 측정은 15분 간격).
- 걸음수(Steps)는 45명 중 1명만 있다. METs는 34명에게 있다.
- HbA1c는 1회뿐이다.

**Hall 2018**
- 57명. A1C·FBG·Weight가 모두 있는 사람은 **46명**이다(체중 결측 10, A1C 결측 2, FBG 결측 1).
- CGM(S1)은 57명 전원에게 있다. 분석창 기준 1인 판독 수 중앙값은 1,822, 기간 중앙값은 7일이다.
- 1명은 기간이 423일로 계산된다. 날짜 이상치이므로 점검이 필요하다.

**BIG IDEAs**
- 16명 모두 HbA1c(5.3–6.4%)와 Dexcom CGM이 있다. 포착률 중앙값은 98.8%, 범위는 75–100%다.
- **체중이 없어서 현 모델 입력을 충족하지 못한다.**
- `report/BIG_IDEAs_per_person_summary.csv`의 GMI 열은 파생 추정치다. **정답으로 쓰지 않는다.**

### 4.3 신청이나 추가 확인이 필요한 자료 (이번에 계정 생성·서명·신청·결제는 하지 않음)

| 자료 | 절차 | 준비서류 | 담당 주체 | 비고 |
|---|---|---|---|---|
| KNHANES | 누리집 회원가입 → 원시자료 다운로드 → 이용 서약 | 개인정보, 서약 | 연구자 본인 | 무료. 이용지침서로 HbA1c 측정대상·연도 확인 |
| KoGES 안산·안성 | CODA 온라인 신청 → 월 1회 분양심의 → 방문·원격분석 또는 제공 | 연구계획서, IRB 심의결과서, 신청서 | 책임연구자·소속기관 | 비용·제공형태(반출 가능 여부)는 미확인 |
| JAEB 공개 데이터셋 | 데이터셋 클릭 → 성명·소속·사용목적(500자) 입력 → 약관 체크 | 성명, 소속 | 연구자 본인 | 무료. 출처표기 의무 |
| AI-READI | FAIRhub 로그인 → Data License 동의(제한필드는 DTUA+DAC) | 계정, 소속 | 연구자 | — |
| CHARLS, CHNS | 웹 등록 → 승인 | 등록정보 | 연구자 | 무료 |
| NIDDK-CR (DCCT/EDIC, Look AHEAD, DPP, TODAY) | R4R 데이터 요청 | 기관 상근 PI, IRB 승인/면제, 연구·분석계획, 보안대책 | 소속기관 | — |
| BioLINCC ACCORD | 등록 → 데이터 요청 | IRB, 계획 | 소속기관 | — |
| UK Biobank | 기관·연구자 등록 → 신청 승인 → UKB-RAP | 연구계획, 기관 서명 | 기관 | £3,000–9,000/3년 |
| All of Us | 기관 DURA → 신원확인 → RCR 교육 → Workbench | 기관 계약 | 기관 | 국외 기관 가능 여부 미확인 |
| MIMIC-IV, eICU | PhysioNet 자격인증 → CITI 교육 → DUA 서명 | 교육 수료증, 추천인 | 연구자 | 무료 |
| T1DEXI | Vivli 요청 → 심사 → DUA → 교육 | 계획서, 통계 담당자 | 기관 | — |
| NHIS 표본코호트(③) | IRB → 신청 → 심의(25일) → 비용 → 원격분석 | IRB, 계획서 | 기관 | HbA1c가 없어 정답으로 못 씀. 보조용 |
| Maastricht | 포털 요청 → 승인 → 관리비 | 계획서 | 기관 | — |
| OhioT1DM, D1NAMO | DUA / 서명 양식 | — | 기관 | HbA1c가 없어 우선순위 낮음 |

표의 의미
- 간이 절차(KNHANES, JAEB, AI-READI, CHARLS, CHNS)는 **1–2주 안에** 사용자 본인이 처리할 수 있는 범위로 보인다.
- 국내 종단 자료(KoGES)와 미국 시험자료(NIDDK-CR)는 IRB가 선행되어야 한다. 지금부터 IRB 준비를 시작해야 B 질문이 일정상 가능해진다.

---

## ⑤ 최소 수집안과 확장 수집안 비교

### 5.1 기존 자료로 충족되는 요구와 부족한 요구

| 요구 | 기존 자료로 충족? | 근거 |
|---|---|---|
| 실측 HbA1c–혈당–체중의 동일인 횡단면 연결 | 충족(NHANES 확보, KNHANES는 신청) | §4.2 |
| 한국인 반복 HbA1c(B) | 신청 시 충족 가능(KoGES, 약 2년 간격) | 코드북 |
| 3개월 단위 반복 HbA1c + 일상 혈당 | 신청 시 일부 충족(DCCT, ACCORD, JAEB 시험: 미국·임상시험 인구) | 문헌 |
| **모바일 앱 자가입력(SMBG·체중) ↔ 검사실 HbA1c 연결** | **부족**: 기존 자료의 혈당은 대부분 정맥 혈장이나 CGM | — |
| **한국인 CGM + 활동량 + 체중 + 반복 HbA1c를 동일인에서 함께** | **부족** | — |
| 기기 단위(cross-device) FL을 실제 기기에서 운영한 기록(업데이트·통신·탈락) | **부족**: 어떤 데이터셋도 없음 | — |
| 수기입력 오류율, 검사값 출처확인 비율 | **부족** | — |

### 5.2 두 안 비교

| 항목 | 최소안: 현재 혈당·체중 모델 검증(A) | 확장안: 반복검사·CGM·생활기록으로 예측·개인화(B·C) |
|---|---|---|
| 목적 | 앱 입력(자가혈당·체중) → 검사실 HbA1c 추정 오차 추정. 앱·서버 FL 흐름 실증 | 미래 HbA1c 예측, 개인화 FL의 장기 효과, CGM·활동 기여 평가 |
| 대상 | 19세 이상 성인. 정상·전당뇨·2형·1형을 **층화 모집**(사전 제외 없음) | 동일. 반복 추적이 가능한 사람(당뇨군 비중을 더 높이는 층화) |
| 모집경로 | 대학병원·의원 내분비내과 외래, 건강검진센터, 지역사회 공고(정상·전당뇨) | 동일 + 당뇨 교육실 + CGM 사용자 모임 |
| 포함 기준 | 스마트폰 사용 가능, 혈당측정기 사용에 동의, 서면·전자 동의 | 최소안 + 12개월 추적 가능 + CGM 부착 동의 |
| 제외 기준과 근거 | 동의 능력 없음, 연구 참여 시 안전 위험. **빈혈·신질환·임신 등은 제외하지 않고 기록**한다(층화·민감도 분석). HbA1c 해석에 영향이 커서 분석 단계에서 다룬다 | 최소안 + CGM 접착제 알레르기(안전) |
| 관찰기간·방문 | 2–4주 가정 측정 → 방문 1회(V1: HbA1c, 정맥 공복혈당, 실측 체중). 선택적으로 3개월 뒤 V2 | 12개월. V0(기저), V1 3개월, V2 6개월, V3 9개월, V4 12개월 |
| HbA1c 검사시점 근거 | 1회(V1). 측정 직전 입력기간 14–28일 | 3개월 간격. ADA 2026: 목표 미달이나 치료 변경 시 3개월마다. KDA(대한당뇨병학회) 2025: 2–3개월 간격 |
| 혈당 측정 | SMBG 공복 매일(최소 주 5회) + 식후 2시간 주 2회(유형 태그 필수) | SMBG 동일 + **각 HbA1c 직전 14일 CGM**(국제합의: ≥14일, 포착률 ≥70%) |
| 체중 | 블루투스 체중계로 매일 아침(최소 주 3회) | 매일 |
| 생활기록 | 없음(선택 설문) | 스마트폰·워치 걸음수와 활동(연속), 수면(선택), 투약 변경 이벤트 |
| 검사값 출처확인 | 검사실 LIS(검사정보시스템) 직접 전송 또는 결과지 사진 + 코디네이터 이중대조 | 동일 + 기관 간 측정법 기록(NGSP 인증) |
| 수기입력 오류 점검 | 측정기 메모리 업로드와 앱 수기값 대조(불일치율 산출), 범위·단위 검사 | 동일 + CGM과 SMBG 동시값 비교 |
| 검사일–입력 연결 규칙 | label = V1 채혈일의 HbA1c. 입력 = 채혈 전 [−28일, −1일]의 측정값 요약(평균·개수) | t0 = 마지막 입력일, 관측기간 W(예: 90일), 예측간격 h(예: 90일). 레이블 허용창 ±14일 |
| 누락·측정중단 | 창 안의 유효 측정 일수 기준(예: SMBG ≥10일) 미달 시 '불충분'으로 표시, 해당 레코드는 주분석 제외 + 민감도 분석 | 동일 + CGM 포착률 <70%는 CGM 특성을 결측 처리 |
| 기기 교체 | device_id를 새로 발급하고 교체일 기록. 같은 참여자 ID에 연결 | 동일 + 센서 세대 변경 공변량 |
| 약물 변경 | 이벤트 기록. 창 안에 변경이 있으면 플래그 | 예측창과 겹치면 층화·민감도 분석. 변경 자체를 특성으로 쓸 때는 t0 이전 정보만 사용 |
| 참여자 부담 | 2–4주, 하루 2–3분 + 방문 1회 | 12개월, 방문 5회, CGM 5회 부착 |
| 비용 항목(금액 미산정) | HbA1c·공복혈당 검사, 혈당측정기·스트립, 체중계, 사례비, 코디네이터, 앱·서버, 보안점검 | 최소안 + CGM 센서(14일×5회), 웨어러블, 장기 사례비, 탈락 방지 연락 |
| 현장운영 | 등록·교육 30분, 앱 설치 지원 | 정기 리마인더, 3개월 방문, 센서 부착 교육 |
| IRB·동의 | 생명윤리법·개인정보보호법(가명정보)에 따른 IRB. 설명문에 FL 구조와 업데이트 전송 사실을 명시 | 동일 + 장기 보관 |
| 철회·보관·삭제 | 철회 시 기기와 서버의 원자료·로그 삭제. **이미 집계된 글로벌 모델에서 개인 기여를 제거할 수 없다는 점을 동의서에 명시**. 보관기간은 기관 IRB 규정(생명윤리법상 기록 보관 기간 확인 필요) | 동일 |

표의 의미
- 최소안은 기존 자료에 없는 **'앱 자가입력 ↔ 검사실 HbA1c' 연결과 실제 기기 FL 운영 기록**을 만드는 가장 작은 단위다.
- 확장안은 B·C 질문에 필요한 **사람 내 반복 HbA1c**(사람당 5회)를 만든다.
- 반복 HbA1c는 기존 신청형 자료(KoGES 등)로도 일부 얻을 수 있다. 그래서 확장안은 그 결과를 보고 규모를 정하는 것이 효율적이다.

### 5.3 시간 연결 규칙 (공통)
1. 기준점은 **HbA1c 채혈일**이다. 결과 보고일이나 앱 업로드일이 아니다.
2. 입력 특성은 **채혈일보다 앞선 창** 안의 측정만으로 만든다. 채혈 당일 측정은 시각이 채혈보다 앞설 때만 넣는다.
3. 다른 시점의 값을 한 행으로 임의 결합하지 않는다.
   - 예: 3개월 전 체중 + 오늘 HbA1c를 '현재값'으로 묶는 것.
   - 창 정의와 요약 방식(평균, 마지막 값, 개수, 경과일)을 변수로 함께 저장한다.
4. HbA1c는 대략 지난 2–3개월의 혈당을 반영한다. 따라서 B 질문의 W·h 조합은 사전에 등록하고, 사후에 바꾸지 않는다. 예: W=90일, h∈{90, 180}일.

### 5.4 예비연구와 본연구 구분
- **예비연구(최소안 형식)**: 실행가능성 지표를 추정한다.
  - 동의율, 4주 순응도, 수기입력 불일치율, 결과지 확인 가능 비율, 탈락률, ICC(사람 간·내 분산)
  - 지표 비율을 ±15%p 정밀도로 추정하려면 약 40명이 필요하다(p=0.5일 때 1.96×√(0.25/40)=0.155). 이 값으로 규모를 정한다.
- **본연구 규모는 예비연구 결과로 산정한다.** 아래 계산은 가정을 보이기 위한 예시다.

### 5.5 표본수 산정 틀 (고정값 제시가 아니라 가정 제시)
1. **평가 정밀도(MAE의 95% 신뢰구간 반폭 d)**: n ≈ (1.96·SD|e| / d)²
   - NHANES 자가보고 당뇨군의 탐색값 SD|e|=0.88 %p를 가정 입력으로 쓰면:
     - d=0.10 %p → 약 296명
     - d=0.15 %p → 약 132명
   - 비당뇨군은 SD|e|가 작아(0.28) 필요 수가 크게 줄어든다.
   - **앱 자가입력에서는 SD가 더 클 수 있으므로 예비연구로 갱신한다.**
2. **반복측정**: 사람당 m회 HbA1c, 사람 내 상관을 ICC라 하면 설계효과는 DE = 1+(m−1)·ICC다. 유효 표본은 n·m/DE이다.
   - ICC는 KoGES·NIDDK-CR 신청 자료나 예비연구로 추정한다.
3. **탈락률 r**: 모집 수 = 필요 수/(1−r). 12개월 추적의 r은 예비연구와 유사 연구에서 가져온다(미확인).
4. **집단별 오차 보고**(당뇨 유형, 연령, 빈혈 등): 하위집단마다 최소 정밀도를 만족하도록 층화 할당한다.
5. **FL 비교**(개인화 대 전역): 사람별 오차 차이의 SD로 짝지은 비교 검정력을 계산한다. 이 SD는 예비연구로 추정해야 한다.

---

## ⑥ 변수사전과 시간 연결 구조
전체 42개 변수(필수·권장·선택, 자료형, 단위, 출처, 결측표시, 품질규칙)는 `report/변수사전.xlsx`에 있다. 핵심만 요약한다.

| 범주 | 필수 | 권장 | 선택 |
|---|---|---|---|
| 식별자 | participant_pid(가명), device_id, visit_id, lab_test_id | site_id(기관; cross-silo에서는 필수) | — |
| HbA1c | hba1c_value_raw + unit_raw(원본 보존), hba1c_pct_std(파생), hba1c_date(채혈일), hba1c_source, hba1c_verification, is_estimated_a1c | hba1c_method(HPLC 등, NGSP 인증) | — |
| 혈당 | glucose_value_raw + unit_raw, glucose_mgdl_std, glucose_datetime(+시간대), glucose_type(공복/식후/무작위/SMBG/CGM/혈장) | meal_relation_min | — |
| 체중 | weight_kg, weight_datetime | weight_source | — |
| 인구·임상 | age, sex, diabetes_type, treatment, med_change_date | height, diabetes_duration, ethnicity | — |
| 시계열 | — | — | cgm_glucose_series(확장안에서는 필수), steps_daily, activity_minutes, sleep |
| 해석 영향 임상상태 | pregnancy_status(가임 여성) | hemoglobin/anemia, eGFR/CKD, transfusion_3mo, hemoglobinopathy | — |

**단위 변환** (원본 값과 단위는 항상 보존하고, 표준화 값은 별도 열에 둔다)
- HbA1c: NGSP % = 0.09148 × IFCC(mmol/mol) + 2.152, IFCC = 10.93 × NGSP − 23.50
  - NGSP/IFCC 마스터방정식, ngsp.org/ifcc.asp
  - NGSP: 미국 당화혈색소 표준화 프로그램, IFCC: 국제임상화학연맹
- 혈당: mg/dL = mmol/L × 18.016 (포도당 분자량 180.16 g/mol). NHANES는 mg/dL × 0.05551로 mmol/L를 만든다(코드북 확인).
- **eAG(추정 평균혈당)와 GMI(혈당관리지표, CGM 평균으로 추정한 A1c)는 파생 추정값이다. 실측 HbA1c와 같은 열에 섞지 않는다.**
  - is_estimated_a1c=TRUE인 값은 label로 쓰지 않는다.
  - GMI(%) = 3.31 + 0.02392 × 평균 CGM 혈당(mg/dL) (Bergenstal 2018). ADA 2026은 GMI가 A1C와 항상 같지 않다고 명시한다.
- 확보 자료에서 발견한 단위 문제
  - CGMacros: 사전은 mmol/mol, 실제 값은 %
  - Shanghai: mmol/mol로 변환되어 제공됨
  - UK Biobank: mmol/mol
  - → 적재할 때 **값의 범위로 단위를 교차검증하는 규칙**을 둔다(예: 3–20이면 %, 20–200이면 mmol/mol, 경계값은 수동 검토).

**임상상태 처리** (일괄 제외 금지)
- **빈혈**: 철결핍은 HbA1c를 높이고, 용혈·출혈은 낮추는 방향이다. 방향이 서로 반대이므로 플래그와 Hb 값을 남긴다. 주분석은 전체로, 민감도 분석은 빈혈 제외로 한다.
- **신질환(eGFR 저하), 최근 3개월 수혈, 임신(임신 중·산후), 혈색소병증(측정법별 간섭)**: 층화와 민감도 분석을 사전에 명시한다.
- 각 상태를 실제로 확인할 수 있는지 데이터셋별로 정리했다.
  - NHANES: Hb와 임신 확인 가능. 수혈·혈색소병증은 미확인
  - Shanghai: eGFR과 동반질환 확인 가능
  - CGMacros·BIG IDEAs·Hall: 대부분 확인 불가

**시간 연결 구조**
- participant(1)
  - visit(N): 체중, 기관
  - lab_test(HbA1c, N): 채혈일 = t0 기준
- device(N): glucose_obs(다수, 시각), activity_daily(일)
- medication_episode(기간)
- feature_window(HbA1c 1건당 1행, 생성 규칙 버전 포함)

이 구조의 원칙은 두 가지다.
- 서로 다른 시점의 값을 한 행에 넣는 일은 **feature_window 생성 규칙을 통해서만** 한다.
- 모든 파생 행은 근거가 된 원 관측 ID 목록을 추적할 수 있어야 한다.

**날짜 비식별화의 영향**
- 확보한 자료는 날짜가 이동되어 있다. MIMIC, CGMacros(±365–720일), BIG IDEAs가 그렇다.
- 사람마다 같은 이동량을 쓰면 **사람 내 간격과 순서는 보존된다.** 그러나 계절, 요일, 달력 효과와 사람 간 동시기 비교는 왜곡된다.
- NHANES는 검사일이 없고 반기 정보(RIDEXMON)만 있다. 그래서 시간 창을 만들 수 없다.
- 신규 수집에서는 원 날짜를 키 보관소에만 둔다. 분석용으로는 개인별 일관 이동 + 이동량 기록(date_shift_days)을 쓴다.

---

## ⑦ 품질점검, 누출방지, FL 평가계획

### 7.1 품질점검 규칙
- 범위
  - HbA1c 3.0–20.0 %
  - 혈당 20–600 mg/dL. CGM 표시 상·하한값은 별도 코드로 둔다.
  - 체중 20–300 kg. 1일 변화가 3 kg을 넘으면 플래그한다.
- 단위 교차검증, 중복 업로드 탐지(같은 값·같은 시각), 기기 시계 오차 점검(서버 수신시각 대비)
- 결과지 이중대조 불일치율, 측정기 메모리 대비 수기입력 불일치율을 보고한다.
- 데이터셋별 이상: Shanghai 임시파일·동일 HbA1c 반복, CGMacros 단위 오기·Libre 보간, Hall 423일 기간 등. 이런 이상은 적재 로그에 남긴다.

### 7.2 정보 누출 차단
1. **사람 단위 분할**: 같은 사람의 기록이 학습과 평가에 동시에 들어가지 않게 한다(신규 사용자 평가).
2. **시간 분할**: 기존 사용자 평가는 각 사람의 마지막 HbA1c를 평가에, 그 이전을 학습에 둔다. t0 이후 관측은 특성에서 제외한다.
3. 정규화 평균·분산과 결측대체 값은 **학습 자료에서만** 추정한다. FL에서는 클라이언트 로컬 통계만 쓰거나, 연합 집계 통계를 쓴다.
4. 레이블 유래 특성을 금지한다: 이전 HbA1c는 t0 이전 것만, GMI·eAG는 레이블과 같은 창에서 계산하지 않는다.
5. 약물 변경 같은 사후 정보는 t0 이후 것을 쓰지 않는다.

### 7.3 연합학습 단위 구분
- **개인단위(cross-device)**: 한 사람의 휴대폰 1대가 클라이언트 1개다. 사람당 표본이 적고(현재 1건), 참여가 간헐적이다.
- **기관단위(cross-silo)**: 병원이나 코호트 1곳이 클라이언트 1개다. KoGES HEXA·CAVAS의 검사기관 코드, eICU hospitalid, CDM 분산망처럼 **실제 기관 구분이 있는 자료**에서만 실제 이질성을 평가할 수 있다.
- **모의 참여단위**: NHANES처럼 1인 1기록 자료를 사람이나 집단으로 임의 분할해 만든 '가짜 클라이언트'다. 통신·집계 코드의 점검과 non-IID 민감도 분석에만 쓴다.
  - **횡단면 분할 실험만으로 장기 개인학습 효과를 주장하지 않는다.**

### 7.4 비교 조건과 지표
- 비교할 모델
  1. 평균예측: 학습 평균
  2. 일반 회귀: 중앙집중 선형. **이용조건이 데이터 통합을 허용할 때만** 수행한다. 예: 원격분석 환경 안에서만 허용되는 자료는 그 안에서만.
  3. 개인별 로컬 모델(반복 HbA1c가 있을 때)
  4. 연합(FedAvg 등)
  5. 연합+개인화(로컬 미세조정 등)
  - 모든 조건에서 입력, 전처리, 분할을 같게 한다.
- 지표
  - MAE, RMSE(단위 %p)
  - **사람별 오차 분포**(중앙값, 90백분위)
  - 집단별 오차: 당뇨 유형, 연령, 성별, 빈혈·신질환·임신 층
  - **사람 단위 클러스터 부트스트랩 95% 신뢰구간**
- 신규 사용자(학습에 없던 사람)와 기존 사용자의 미래 시점은 **따로** 보고한다.
- 사람별 표본수와 검사 횟수 분포, 집단 간 분포 차이(입력·레이블 분포 비교)를 결과와 함께 낸다.

### 7.5 데이터셋별 FL 실험 적합성

| 자료 | 실제 참여단위 | 가능한 실험 | 불가능한 주장 |
|---|---|---|---|
| NHANES | 없음(1인 1기록) | A 오차 추정, 모의 cross-device 통신 점검 | 개인화·장기 학습 효과 |
| KoGES 안산·안성 | 사람(반복 11회), 지역 2곳 | B, 사람 단위 시간분할, 지역 2-silo | 일상 모바일 입력 효과(검진 공복혈당) |
| JAEB·DCCT·ACCORD | 사람, 임상기관 | B, CGM→HbA1c, 기관 cross-silo | 한국인 일반화 |
| 신규 확장안 | 실제 기기 | C 전체, 기기 단위 FL 운영 | — |

### 7.6 프라이버시: 원자료 미전송 ≠ 프라이버시 완성
- 현 시제품은 입력 x=(혈당, 체중), 출력 y, 단일 선형층이며 1건으로 학습한다. 제곱오차 기준 업데이트는 다음과 같다.
  - ∂L/∂w = (ŷ−y)·x
  - ∂L/∂b = (ŷ−y)
- 따라서 서버는 **x = (∂L/∂w)/(∂L/∂b)로 입력을 정확히 복원**할 수 있다. ŷ는 서버가 아는 모델로 계산할 수 있으므로 **y(HbA1c)도 복원**된다.
- 즉 현 구조에서 업데이트를 보내는 것은 원자료를 보내는 것과 사실상 같다.
- 필요한 대책(구현·검증 여부는 별도 확인)
  - 보안집계: 서버는 합계만 본다
  - 로컬 다회 학습·배치화
  - 차분프라이버시(업데이트 클리핑 + 잡음)
  - 최소 참여자 수 이상일 때만 집계
- 구분해서 관리할 것
  - **업데이트**: 전송 내용, 보관 여부
  - **로그**: 접속시각, 기기정보, IP는 그 자체가 건강정보 이용 사실을 드러냄
  - **보관자료**: 백업, 크래시 리포트
  - 각각의 보관기간과 삭제절차를 IRB 서류에 기재한다.

---

## ⑧ 단계별 실행 순서와 남은 결정사항

| 단계 | 할 일 | 산출물 | 주체 |
|---|---|---|---|
| 0 (완료) | 공개자료 확보, 체크섬, 연결 N 산출, 후보표·변수사전 | 이 보고서, manifest | 연구원 |
| 1 (1–2주) | NHANES로 A 기준선 재현(가중치·층화 반영), 합성데이터로 FL 파이프라인 점검 | 기준선 보고 | 연구원 |
| 2 (1–2주) | 간이 신청: KNHANES, JAEB(GMI·REPLACE-BG·DCLP3·CITY·WISDM), AI-READI, CHARLS/CHNS | 원자료 + 변수 확인 갱신 | **사용자 본인**(계정·약관) |
| 3 (1–3개월) | IRB 준비 → KoGES CODA 신청. NIDDK-CR/BioLINCC 중 1–2개 선택 | 분양 승인 | 책임연구자·기관 |
| 4 | 최소안 예비연구 IRB, 앱 보안 점검(§7.6 대책) | 예비연구 프로토콜 | 연구팀 |
| 5 | 예비연구 수행 → ICC, 탈락, 입력오류 추정 → 본연구 표본수 확정 | 표본수 보고 | 연구팀 |
| 6 | 확장안 본연구(12개월, CGM·활동) | B·C 평가 | 연구팀 |

**남은 결정사항**
1. 1차 대상 집단의 층화 비율(정상·전당뇨·2형·1형). 앱의 의도된 사용자에 따라 정한다.
2. 앱의 혈당 입력 정의: 공복 SMBG만 받을지, 유형 태그를 붙인 다중 입력을 받을지. NHANES(정맥 공복)와 대응 가능성이 달라진다.
3. B 질문의 W·h 조합(사전등록).
4. 상업적 이용 여부. CGMacros(CC BY-NC-SA)는 비상업 조건이라 사용 여부가 갈린다.
5. 기관단위 FL을 병행할지(KoGES 지역·검사기관, CDM 분산망).
6. 보안집계·차분프라이버시 구현 범위와 검증 방법.
7. 철회 시 글로벌 모델 처리 방침(재학습 주기 등).

---

## ⑨ 한 것과 안 한 것

| 구분 | 내용 | 결과 |
|---|---|---|
| 한 것 | NHANES 2021–2023·2017–2018 HbA1c·공복혈당·체중·인구·당뇨설문·CBC·공복시간 XPT와 코드북 다운로드 | 완료(20 MB) |
| 한 것 | NHANES 연결 N, 결측률, 집단별 분포, 임상상태 수, 탐색적 기준선 계산 | 3,631 / 2,840명 |
| 한 것 | CGMacros(627 MB zip 포함)·BIG IDEAs(소형 파일)·Hall S1/S5·Shanghai 다운로드와 공식 체크섬 검증 | 공식 해시 전부 일치 |
| 한 것 | 4개 CGM 자료의 사람별 연결 N, 반복 HbA1c, CGM 포착률, 단위 이상 점검 | §4.2 |
| 한 것 | KoGES 공개 코드북 15종 다운로드, HbA1c·공복혈당·체중 변수 확인 | 전 차수 확인 |
| 한 것 | 36개 후보 공식 페이지·문헌 확인과 분류 | xlsx |
| 한 것 | 변수사전, 시간연결구조, 단위변환 시트 | xlsx |
| 한 것 | 합성 데이터를 별도 폴더에 생성(FL 흐름 점검용) | data\synthetic |
| 안 한 것 | 계정 생성, 회원가입, 약관 동의(JAEB 이름·목적 입력 포함), DUA 서명, IRB 신청, 결제, 개인정보 제출 | 원칙상 미수행 |
| 안 한 것 | BIG IDEAs의 Empatica 원신호(4.7 GB zip) | 우선순위 낮음 |
| 안 한 것 | D1NAMO 다운로드 | 라이선스 문구 불일치, HbA1c 없음 |
| 안 한 것 | Kaggle·Pima·UCI 다운로드 | ③·④ |
| 안 한 것 | 데이터셋 간 결합(서로 다른 사람) | 금지 원칙 |
| 실패·제약 | figshare가 box IP 차단(403) | PC에서 직접 받아 해결 |
| 실패·제약 | PhysioNet 직접 다운로드 속도 저하 | 공식 S3 공개 미러로 받고 SHA256 일치 확인 |
| 실패·제약 | Glucose-ML 카탈로그 로딩 실패, Awesome-CGM 본문 미수집, Zenodo 페이지 504 | 검색 요약으로 대체 |
| 미확인으로 남김 | UK Biobank 혈당·체중 필드번호, All of Us 인원, JAEB 파일 내 HbA1c·체중 변수, KNHANES 연도별 HbA1c 측정대상, KoGES 비용·반출 형태, HRS DBS HbA1c | 신청·문서 열람 후 갱신 |

---

## 부록 A. 주요 공식 근거 링크
- NHANES GHB_L 코드북: https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/GHB_L.htm (GLU_L, BMX_L 등 같은 경로)
- KNHANES: https://knhanes.kdca.go.kr/knhanes/ / 제10기 계획: https://www.e-epih.org/journal/view.php?number=1641
- KoGES 자료분양: https://www.nih.go.kr/ko/main/contents.do?menuNo=300566 / 코드북: menuNo=300577 / CODA: https://coda.nih.go.kr
- NHIS 빅데이터: https://nhiss.nhis.or.kr/ / HIRA: https://opendata.hira.or.kr/ / 복지부 통합플랫폼: https://hcdl.mohw.go.kr/
- UK Biobank Field 30750: https://biobank.ndph.ox.ac.uk/ukb/field.cgi?id=30750 / 비용: https://www.ukbiobank.ac.uk/use-our-data/fees/
- All of Us: https://www.researchallofus.org/data-tools/workbench/
- MIMIC-IV: https://physionet.org/content/mimiciv/ / eICU: https://physionet.org/content/eicu-crd/2.0/
- NIDDK-CR: https://repository.niddk.nih.gov/ (EDIC study/16, Look AHEAD study/61) / ACCORD: https://biolincc.nhlbi.nih.gov/studies/accord/
- JAEB: https://public.jaeb.org/datasets/diabetes / T1DEXI: https://search.vivli.org/doiLanding/studies/PR00008428/isLanding
- OhioT1DM: https://webpages.charlotte.edu/rbunescu/data/ohiot1dm/OhioT1DM-dataset.html
- ShanghaiT1DM/T2DM: https://doi.org/10.6084/m9.figshare.20444397.v3 / 논문: https://www.nature.com/articles/s41597-023-01940-7
- D1NAMO: https://zenodo.org/records/5651217
- CGMacros: https://physionet.org/content/cgmacros/1.0.0/ / BIG IDEAs: https://physionet.org/content/big-ideas-glycemic-wearable/1.1.3/
- Hall 2018: https://doi.org/10.1371/journal.pbio.2005143
- Maastricht: https://www.demaastrichtstudie.nl/research/data-guidelines / AI-READI: https://docs.aireadi.org/
- CHARLS: https://charls.pku.edu.cn/en/ / CHNS: https://chns.cpc.unc.edu/data/datasets/biomarker-data/ / 일본 NDB: https://www.mhlw.go.jp/stf/seisakunitsuite/bunya/0000177182.html
- 지침
  - ADA Standards of Care 2026 Section 6: https://doi.org/10.2337/dc26-s006
  - 대한당뇨병학회 2025 진료지침: https://doi.org/10.4093/jkd.2025.26.3.152
  - CGM 임상시험 지표 국제합의(Battelino 2023): https://pubmed.ncbi.nlm.nih.gov/36493795/
  - GMI(Bergenstal 2018): https://pmc.ncbi.nlm.nih.gov/articles/PMC6196826/
  - NGSP/IFCC 마스터방정식: https://ngsp.org/ifcc.asp

## 부록 B. 산출 파일
- `report/` 아래
  - HbA1c_FL_데이터확보_연구설계.md / .docx
  - 후보_데이터셋_전체목록.xlsx / .csv
  - 변수사전.xlsx
  - NHANES_linkage_summary.json, NHANES_group_distribution.csv
  - Shanghai_linkage_summary.json, CGM_sets_summary.json
  - CGMacros_per_person_summary.csv, BIG_IDEAs_per_person_summary.csv
- `data/manifest.csv`: 모든 파일의 출처 URL, 버전, 이용조건, 접근일, 크기, SHA256, 공식 체크섬 대조, R: 경로
- `scripts/`: nhanes_linkage.py, candidates.py, build_xlsx.py
- `data/synthetic/`: 합성 데이터, 생성 스크립트, README(정답·성능 근거로 사용 금지)
