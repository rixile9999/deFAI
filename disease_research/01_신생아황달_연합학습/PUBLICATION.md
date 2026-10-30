> Version 0.2 integration: use ../../docs/user_guide_ko.md and the package optional dependencies. This note below records the original 2026-10-29 publication; current modified-source hashes are in defai/research/catalog.json.

# Neonatal jaundice research

Research stage; not deployed.

Entry points: NeoJaundice/run_federated.py; NeoJaundice-virtual/run_federated.py; GAN_research/prepare_data.py; GAN_research/run_algorithm_study.py.

Restore NeoJaundice/data/images and metadata.csv from the official archive. GAN preparation creates patient-disjoint partitions and quarantines conflicting or zero targets. Existing split/result files and trained models are intentionally omitted. Synthetic institutions and brightness perturbations are experimental scenarios, not verified hospital or skin-tone labels.

See ../references for source data and ../VALIDATION.md for publication-time checks. Source snapshots are byte-identical to the authorized workspace. Some inherited scripts contain machine-specific paths and need local configuration before use.
