# Disease-related federated learning research sources

Research stage; not deployed clinical or public-health systems.

This publication preserves source files from the four numbered disease-research
directories as they existed on 2026-10-29. Source files are copied byte for byte.
Commits record today's publication of existing work, not historical development
dates. Each source/configuration/document file has its own publication commit.
The original workspace remains unchanged.

| Directory prefix | Included work | Data and execution boundary |
|---|---|---|
| 01 | Neonatal image models, simulated clients, generative augmentation, algorithm and skin-bias studies | NeoJaundice images must be obtained separately; clients are simulated institutions |
| 02 | Pandemic severity models, hierarchical encrypted policy statistics, synthetic-region validation | Included generators produce synthetic records; these do not establish real policy effects |
| 03 | Agent-based simulation, federated Burr estimation, Paillier sum aggregation | Full simulation needs separately obtained population/place input files; public provenance of those local files is unresolved |
| 04 | Android/Python prototype, public-data linkage, calibrated synthetic longitudinal cohorts, algorithm comparison | NHANES/Hall calibration is distinct from real longitudinal clinical validation |

Each topic's PUBLICATION.md explains its entry points and limits. The references
directory identifies public data and distinguishes actual calibration inputs
from collected/exploratory datasets. source_manifest.json records original
relative paths, file sizes and SHA-256 hashes.

## Scope and interpretation

Excluded: unpublished methods, cross-topic reports containing unpublished work,
archive snapshots, duplicate build/run trees, environments, cryptographic keys,
photographs, participant-level tables, trained models and generated results.
Historical demo narratives are preserved as source documentation; their numerical
claims are not newly validated by this publication. Current research limitations
in the topic READMEs take precedence over earlier demo narratives.

Tests that require omitted data/results need those inputs restored locally first.
Source tests and small self-tests are described in VALIDATION.md. Syntax checks
do not imply end-to-end or clinical validation. Android builds, full training,
full population simulations and real multi-institution deployments were not run
as part of this publication.
