> Version 0.2 integration: use ../../docs/user_guide_ko.md and the package optional dependencies. This note below records the original 2026-10-29 publication; current modified-source hashes are in defai/research/catalog.json.

# Mobile HbA1c federated-learning research

Research stage; not deployed.

Entry points: deFai_federated/defai_server_python/server.py; deFai_federated/defai/app/src/main; hba1c_fl_research/experiments/fl_comparison/cohort.py; learning.py.

The current synthetic-cohort calibration reads NHANES 2017-2018 DEMO/GHB/GLU/BMX/FASTQX and Hall2018 S5 clinical data. NHANES 2021-2023 and other collected datasets are not automatically inputs to that calibration. Preserve source-person development/held-out separation before creating virtual clones. A virtual cohort size is not a count of observed clinical participants. Android source variants are retained; build outputs, local.properties, the defai_run copy, and participant files are omitted. Full Android compilation is unverified.

See ../references for source data and ../VALIDATION.md for publication-time checks. Source snapshots are byte-identical to the authorized workspace. Some inherited scripts contain machine-specific paths and need local configuration before use.
