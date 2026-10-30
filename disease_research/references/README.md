# Public data reference registry

neojaundice.md verifies the image dataset against official archive metadata.
nhanes_calibration.md and hall_calibration.md identify inputs actually read by
the current cohort generator. public_inventory_*.csv preserves the existing
local download registry grouped by dataset, with machine-specific paths removed.
Its access dates and license descriptions are historical acquisition records,
not a fresh verification of every source or its current legal terms.

Collected, exploratory and calibration inputs are different categories. Consult
cohort.py for the actual current calibration dependencies. The broader candidate
catalog in scripts/candidates.py includes sources that were never acquired.
No public origin is asserted for synthetic pandemic patient records or for the
unresolved population/place inputs of the agent-based model.
