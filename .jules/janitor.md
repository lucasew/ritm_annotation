# Janitor's Journal

This journal is a record of critical learnings related to code quality and refactoring in this repository. Its purpose is to help maintain consistency and avoid repeating past mistakes.

## 2025-12-31 - Remove Commented-Out Code
**Issue:** The file `ritm_annotation/cli/annotate/annotator.py` contained several blocks of commented-out code.
**Root Cause:** The commented code appeared to be remnants of previous debugging or experimental code that was no longer needed but was never removed.
**Solution:** I removed the unused, commented-out code blocks to clean up the file and improve readability.
**Pattern:** Periodically scan for and remove dead or commented-out code. Such code adds clutter, can be confusing to new developers, and provides no functional value. Keeping the codebase clean of such artifacts is a simple but effective maintenance practice.

## 2026-01-23 - Cleanup Utils/Misc
**Issue:** `ritm_annotation/utils/misc.py` contained debug print statements, redundant imports, and inefficient or flawed helper functions (`incrf`, `try_tqdm`).
**Root Cause:** Helper functions were likely added quickly for specific needs without checking for standard libraries (`itertools`) or best practices. Debug prints were left behind.
**Solution:** Refactored `incrf` to use `itertools.count(1)`, simplified `try_tqdm` to use `tqdm.auto.tqdm` directly without forcing list materialization, and removed debug prints.
**Pattern:** Prefer standard library tools (`itertools`) over custom reimplementations. Ensure utility functions do not have side effects like printing to stdout unless intended. Use `tqdm.auto` for environment-agnostic progress bars.
