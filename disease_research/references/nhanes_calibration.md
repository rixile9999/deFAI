# NHANES inputs used by current cohort calibration

National Health and Nutrition Examination Survey (NHANES), CDC/NCHS, 2017-2018.
cohort.py merges DEMO_J, GHB_J, GLU_J, BMX_J and FASTQX_J on SEQN, then selects
complete adults with 8 <= fasting hours < 24 and positive WTSAF2YR weights.

- Glycohemoglobin: https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/GHB_J.htm
- Fasting plasma glucose: https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/GLU_J.htm
- Demographics: https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/DEMO_J.htm
- Body measures: https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/BMX_J.htm
- Fasting questionnaire: https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/FASTQX_J.htm

Use matching .xpt files from those official pages under data/public/NHANES_2017-2018.
LBXGH is reported in %, LBXGLU in mg/dL, BMXWT in kg. SEQN joins people within a
survey cycle; it is not a cross-dataset identity key. Repeated synthetic visits
are simulated, not observed NHANES follow-up. Existing code uses fasting weights
for calibration but does not establish design-based national inference.
The 2021-2023 cycle is used by exploratory linkage scripts, not by this calibration.
