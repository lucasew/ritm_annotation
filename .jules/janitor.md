# Janitor's Journal

This journal is a record of critical learnings related to code quality and refactoring in this repository. Its purpose is to help maintain consistency and avoid repeating past mistakes.

## 2025-12-31 - Remove Commented-Out Code
**Issue:** The file `ritm_annotation/cli/annotate/annotator.py` contained several blocks of commented-out code.
**Root Cause:** The commented code appeared to be remnants of previous debugging or experimental code that was no longer needed but was never removed.
**Solution:** I removed the unused, commented-out code blocks to clean up the file and improve readability.
**Pattern:** Periodically scan for and remove dead or commented-out code. Such code adds clutter, can be confusing to new developers, and provides no functional value. Keeping the codebase clean of such artifacts is a simple but effective maintenance practice.

## 2026-01-29 - Remove Dead Logging Code
**Issue:** The file `ritm_annotation/utils/log.py` contained commented-out logging setup code and an unused `add_logging` function. `ritm_annotation/utils/exp.py` also contained commented-out calls to this function.
**Root Cause:** Likely leftovers from a previous logging implementation or debugging session that was disabled but not cleaned up.
**Solution:** Removed the commented-out code in both files to improve cleanliness and readability. Also fixed unused imports in `ritm_annotation/utils/test_misc.py`.
**Pattern:** Remove commented-out code ("dead code") to prevent confusion and reduce maintenance overhead.
