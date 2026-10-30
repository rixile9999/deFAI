# ⚠️ Warning Suppression for CES Demo

## 🎯 Problem

When running `demo_simple.bat`, each terminal window showed long deprecation warnings from Flower:

```
WARNING :   DEPRECATED FEATURE: flwr.client.start_numpy_client() is deprecated.
        Instead, use `flwr.client.start_client()` by ensuring you first call the `.to_client()` method as shown below:
        flwr.client.start_client(
                server_address='<IP>:<PORT>',
                client=FlowerClient().to_client(), # <-- where FlowerClient is of type flwr.client.NumPyClient object
        )
        Using `start_numpy_client()` is deprecated.

            This is a deprecated feature. It will be removed
            entirely in future versions of Flower.

WARNING :   DEPRECATED FEATURE: flwr.client.start_client() is deprecated.
        Instead, use the `flower-supernode` CLI command to start a SuperNode as shown below:

                $ flower-supernode --insecure --superlink='<IP>:<PORT>'

        To view all available options, run:

                $ flower-supernode --help

        Using `start_client()` is deprecated.

            This is a deprecated feature. It will be removed
            entirely in future versions of Flower.
```

### Impact on Demo

- ❌ Clutters terminal output
- ❌ Distracts from actual progress messages
- ❌ Makes demo look unprofessional
- ❌ Hides important information like DiD results

---

## ✅ Solution

### Method 1: Python Logging Level (in scripts)

Added **logging configuration** at the **top of each script** (before importing Flower):

```python
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', message='.*DEPRECATED.*')

import logging

# Suppress Flower deprecation warnings (BEFORE importing flwr)
logging.getLogger("flwr").setLevel(logging.ERROR)

import flwr as fl  # Now warnings are suppressed
```

### Method 2: Environment Variables (in batch file)

Added **environment variables** in `demo_simple.bat` for each terminal:

```batch
set PYTHONWARNINGS=ignore::DeprecationWarning
set FLWR_TELEMETRY_ENABLED=0
```

### Files Updated

```
✅ run_tier0.py       (Lines 9-16: warnings + logging)
✅ run_tier1.py       (Lines 11-19: warnings + logging)
✅ run_tier2.py       (Lines 10-18: warnings + logging)
✅ demo_simple.bat    (Lines 8-9 + all start commands: env vars)
```

---

## 🎬 Before vs After

### Before (Cluttered)

```
WARNING :   DEPRECATED FEATURE: flwr.client.start_numpy_client() is deprecated.
        Instead, use `flwr.client.start_client()` by ensuring you first call the `.to_client()` method...
        [... 15 more lines ...]

WARNING :   DEPRECATED FEATURE: flwr.client.start_client() is deprecated.
        Instead, use the `flower-supernode` CLI command...
        [... 12 more lines ...]

======================================================================
[TIER 2] Gangnam: Encrypting data with homomorphic encryption...
======================================================================
```

### After (Clean)

```
======================================================================
[TIER 2] Gangnam: Encrypting data with homomorphic encryption...
======================================================================
   District: Gangnam (Seoul)
   Pre-period mean: 125.4
   Post-period mean: 194.8
   Encrypting with TenSEAL CKKS...
✅ [TIER 2] Gangnam: Data sent successfully!
```

---

## 🔍 Technical Details

### Why This Works

1. **warnings.filterwarnings('ignore', category=DeprecationWarning)**
   - Suppresses all deprecation warnings from any library
   - Targets the standard Python `DeprecationWarning` class

2. **warnings.filterwarnings('ignore', message='.*DEPRECATED.*')**
   - Additional catch-all for messages containing "DEPRECATED"
   - Uses regex pattern matching
   - Ensures no deprecation messages slip through

3. **logging.getLogger("flwr").setLevel(logging.ERROR)**
   - **Critical**: Flower uses Python logging, not just warnings
   - Sets Flower's logger to only show ERROR level and above
   - Suppresses WARNING level messages (where deprecation warnings appear)

4. **PYTHONWARNINGS=ignore::DeprecationWarning** (Environment Variable)
   - System-wide suppression at Python interpreter level
   - Applied before any Python code runs

5. **FLWR_TELEMETRY_ENABLED=0** (Environment Variable)
   - Disables Flower telemetry
   - Reduces additional log output

### Placement is Critical

```python
# ✅ CORRECT - logging.getLogger BEFORE importing flwr
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

import logging
logging.getLogger("flwr").setLevel(logging.ERROR)  # CRITICAL!

import flwr as fl  # Warnings already suppressed

# ❌ WRONG - After importing flwr
import flwr as fl  # Warnings will still show

import logging
logging.getLogger("flwr").setLevel(...)  # Too late!
```

---

## 🎯 Impact on Demo

### Clean Terminal Output

**Tier 2 (Districts)** now shows:
```
======================================================================
[TIER 2] Gangnam: Encrypting data with homomorphic encryption...
======================================================================
   District: Gangnam (Seoul)
   Sending to Province: Seoul (127.0.0.1:9001)
   Status: Connected

✅ [TIER 2] Gangnam: Data sent successfully!
======================================================================
   Status: ✅ Complete
   This window will close in 5 seconds...
======================================================================
```

**Tier 1 (Provinces)** now shows:
```
======================================================================
[TIER 1] Seoul: Performing blind aggregation (cannot see individual data)
======================================================================
   Received: 2 districts
   Homomorphic aggregation: ✅
   Forwarding to KDCA...

✅ [TIER 1] Seoul: Mission complete!
======================================================================
   Phase 1: ✅ Aggregated district data (blind)
   Phase 2: ✅ Forwarded to central government
   Status: All tasks completed successfully
======================================================================
```

**Tier 0 (KDCA)** now shows:
```
======================================================================
✅ [TIER 0] KDCA: Analysis Complete!
======================================================================
   ✅ Received data from all provinces
   ✅ Decrypted aggregated statistics
   ✅ Performed DiD causal analysis
   ✅ Policy effect: -71.2% (statistically significant)
======================================================================
```

---

## 📝 Notes

### Future Flower Versions

These warnings exist because Flower is transitioning to a new API:
- Old: `start_numpy_client()` → New: `.to_client()`
- Old: `start_client()` → New: `flower-supernode` CLI

For production deployment (post-CES), consider:
- [ ] Migrating to new Flower API
- [ ] Using `flower-supernode` CLI instead
- [ ] Updating to latest Flower version

### For CES Demo

Current approach is acceptable because:
- ✅ Clean, professional terminal output
- ✅ No functional impact (just warning suppression)
- ✅ Code still works perfectly
- ✅ Easy to remove filters later

---

## 🧪 Testing

### Quick Test

Run one terminal to verify warnings are gone:

```bash
# Test Tier 2
python run_tier2.py --data data/districts/seoul_gangnam.xlsx --target 127.0.0.1:9001

# Should NOT show deprecation warnings
# Should ONLY show clean status messages
```

### Full Demo Test

```bash
demo_simple.bat

# All 7 windows should show:
# ✅ Clean output
# ✅ No deprecation warnings
# ✅ Clear status messages
# ✅ Professional appearance
```

---

**Updated**: 2026-01-02
**Version**: 1.0 (Warning Suppression)
**Status**: ✅ Complete
**Impact**: Professional, clean demo output

---

# ✨ Result

CES 2026 demo now presents **clean, professional terminal output**
without distracting deprecation warnings. All focus is on the
actual federated learning process and DiD analysis results! 🚀
