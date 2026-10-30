> Version 0.2 integration: use ../../docs/user_guide_ko.md and the package optional dependencies. This note below records the original 2026-10-29 publication; current modified-source hashes are in defai/research/catalog.json.

# Federated generation-interval estimation

Research stage; not deployed.

Entry points: abm_fl_simulation/fl_pipeline/paillier.py; fl_pipeline/federated_burr.py; experiments/e1_correctness.py; experiments/e2_blind_topology.py.

Self-tests create synthetic intervals. ABM input settings are included, but initial_vals/agents_updated.xml and places.xml are excluded. Their upstream population/place provenance has not been established; do not label them verified public data. Deterministic small-key encryption demonstrations are research tests, not production cryptographic deployments. The existing README payload estimate describes plaintext numeric content, not total encrypted network traffic.

See ../references for source data and ../VALIDATION.md for publication-time checks. Source snapshots are byte-identical to the authorized workspace. Some inherited scripts contain machine-specific paths and need local configuration before use.
